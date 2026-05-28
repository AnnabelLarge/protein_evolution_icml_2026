#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 04:33:00 2025


models:
=======
FragAndSiteClasses
FragAndSiteClassesLoadAll

"""
import pickle
import numpy as np

from flax import linen as nn
import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm
from jax.scipy.special import logsumexp

from utils.BaseClasses import ModuleBase
from latent_class_mixtures.emission_models import (EqulDistLogprobsFromCounts,
                                                              EqulDistLogprobsPerClass,
                                                              EqulDistLogprobsFromFile,
                                                              GTRLogprobs,
                                                              GTRLogprobsFromFile,
                                                              RateMultipliersPerClass,
                                                              IndpRateMultipliers,
                                                              RateMultipliersPerClassFromFile,
                                                              IndpRateMultipliersFromFile,
                                                              F81Logprobs,
                                                              F81LogprobsFromFile)
from latent_class_mixtures.transition_models import (TKF92TransitionLogprobs,
                                                                TKF92TransitionLogprobsFromFile)
from latent_class_mixtures.model_functions import (bound_sigmoid,
                                                              safe_log,
                                                              cond_logprob_emit_at_match_per_mixture,
                                                              joint_logprob_emit_at_match_per_mixture,
                                                              lse_over_equl_logprobs_per_mixture,
                                                              lse_over_match_logprobs_per_mixture,
                                                              # joint_only_forward,
                                                              # all_loglikes_forward,
                                                              marginalize_over_times,
                                                              write_matrix_to_npy,
                                                              maybe_write_matrix_to_ascii)

from latent_class_mixtures.one_dim_forward_joint_loglikes import joint_only_one_dim_forward as joint_only_forward
from latent_class_mixtures.one_dim_forward_all_loglikes import all_loglikes_one_dim_forward as all_loglikes_forward

class FragAndSiteClasses(ModuleBase):
    """
    pairHMM that finds joint loglikelihood of alignments, P(Anc, Desc, Align),
      given different hidden fragment classes; each discrete site class has
      its own equilibrium distribution, rate multiplier, and tkf extension 
      probability
    
    if using one time per sample, and wanting to QUANTIZE the time, need a 
        different model
    
    
    C_dom: number of mixtures associated with nested TKF92 models (domain-level)
        > here, C_dom = 1
    C_frag: number of mixtures associated with TKF92 fragments (fragment-level)
    C_trans: C_dom * C_frag = C_frag
    B = batch size; number of samples
    T = number of branch lengths; this could be: 
        > an array of times for all samples (T; marginalize over these later)
        > an array of time per sample (T=B)
        > a quantized array of times per sample (T = T', where T' <= T)
    S: number of transition states (4 here: M, I, D, start/end)
    A: emission alphebet size (20 for proteins)
    
    
    Initialize with
    ----------------
    config : dict
        config['num_domain_mixtures'] : int
            number of larger TKF92 domain mixtures
            (one domain type here)
    
        config['num_fragment_mixtures'] : int
            number of TKF92 fragment mixtures 
        
        config['num_site_mixtures'] : int
            number of mixtures associated with the EMISSIONS
        
        config['subst_model_type'] : {gtr, f81}
            which substitution model
        
        config['exponential_dist_param'] : int, optional
            There is an exponential prior over time; this provides the
            parameter for this during marginalization over times
            Default is 1
        
        config['times_from'] : {geometric, t_array_from_file, t_per_sample}
        
    name : str
        class name, for flax
    
    Main methods here
    -----------------
    setup
    
    __call__
        unpack batch and calculate logP(anc, desc, align)
    
    calculate_all_loglikes
        calculate logP(anc, desc, align), logP(anc), logP(desc), and
        logP(desc, align | anc)
    
    write_params
        write parameters to files
    
    
    Other methods
    --------------
    _get_emission_scoring_matrices
        get the emission matrices, specifically
    
    _get_scoring_matrices
        get the transition and emission matrices
    
    
    Methods inherited from neural_models.model_utils.BaseClasses.ModuleBase
    -----------------------------------------------------------------
    sow_flax_intermeds
        for tensorboard logging
    """
    config: dict
    name: str
    
    def setup(self):
        assert self.config['num_domain_mixtures'] == 1
        
        ###################
        ### read config   #
        ###################
        # required
        self.num_domain_mixtures = 1
        self.num_fragment_mixtures = self.config['num_fragment_mixtures']
        self.num_site_mixtures = self.config['num_site_mixtures']
        self.num_rate_mults = self.config['k_rate_mults']
        self.num_transit_mixtures = ( self.config['num_fragment_mixtures'] *
                                      self.config['num_domain_mixtures'] )# C_tr

        self.subst_model_type = self.config['subst_model_type']
        self.times_from = self.config['times_from']
        
        # optional
        self.indp_rate_mults = self.config.get('indp_rate_mults', False)
        self.gap_idx = self.config.get('gap_idx', 43)
        self.norm_reported_loss_by = 'desc_len'
        self.exponential_dist_param = 1
        
        
        ########################################################
        ### module for transition probabilities, and the       #
        ### fragment-level mixture weights P(c_frag | c_dom)   #
        ########################################################
        self.transitions_module = TKF92TransitionLogprobs(config = self.config,
                                                 name = f'tkf92 indel model')
        
        
        ###############################################################
        ### probability of having a particular subsitution rate       #
        ### rate multiplier, and the rate multipliers themselves      #
        ###############################################################
        if not self.indp_rate_mults:
            self.rate_mult_module = RateMultipliersPerClass(config = self.config,
                                                      name = f'get rate multipliers')
        
        elif self.indp_rate_mults:
            self.rate_mult_module = IndpRateMultipliers(config = self.config,
                                                      name = f'get rate multipliers')
        
        
        ###############################################################
        ### module for equilibrium distribution, and the site-level   # 
        ### mixture weights P(c_sites | c_frag)                       #
        ###############################################################
        if (self.num_transit_mixtures * self.num_site_mixtures) == 1:
            self.equl_dist_module = EqulDistLogprobsFromCounts(config = self.config,
                                                       name = f'get equilibrium')
        elif (self.num_transit_mixtures * self.num_site_mixtures) > 1:
            self.equl_dist_module = EqulDistLogprobsPerClass(config = self.config,
                                                     name = f'get equilibrium')
        
        
        ###########################################
        ### module for substitution rate matrix   #
        ###########################################
        if self.subst_model_type == 'gtr':
            self.logprob_subst_module = GTRLogprobs( config = self.config,
                                                  name = f'gtr subst. model' )
            
        elif self.subst_model_type == 'f81':
            self.logprob_subst_module = F81Logprobs( config = self.config,
                                                     name = f'f81 subst. model' )

    def __call__(self,
                 batch,
                 t_array,
                 sow_flax_intermeds: bool):
        """
        Use this during active model training
        
        
        C_dom: number of mixtures associated with nested TKF92 models (domain-level)
            > here, C_dom = 1
        C_frag: number of mixtures associated with TKF92 fragments (fragment-level)
        C_trans: C_dom * C_frag = C_frag
        B = batch size; number of samples
        T = number of branch lengths; this could be: 
            > an array of times for all samples (T; marginalize over these later)
            > an array of time per sample (T=B)
            > a quantized array of times per sample (T = T', where T' <= T)
        S: number of transition states (4 here: M, I, D, start/end)
        A: emission alphebet size (20 for proteins)
        
        
        returns:
            - loss: average across the batch, based on joint log-likelihood
                    
            - aux_dict: has the following keys and values
              1.) 'joint_neg_logP': sum down the length
              2.) 'joint_neg_logP_length_normed': sum down the length,  
                  normalized by descendant sequence length
              3.) align_length_for_normalization : desired length for reporting
        """
        aligned_inputs = batch[0]
        
        # which times to use for scoring matrices
        if self.times_from =='t_per_sample':
            times_for_matrices = batch[1] #(B,)
            unique_time_per_sample = True
        
        elif self.times_from in ['geometric','t_array_from_file']:
            times_for_matrices = t_array #(T,)
            unique_time_per_sample = False
        
        # scoring matrices
        # 
        # scoring_matrices_dict has the following keys (when return_intermeds is False)
        #   logprob_emit_at_indel, (C_frag, A)
        #   joint_logprob_emit_at_match, (T, C_frag, A, A)
        #   all_transit_matrices, dict, with joint transit matrix being (T, C_frag, C_frag, S, S)
        scoring_matrices_dict = self._get_scoring_matrices(t_array=times_for_matrices,
                                                           sow_flax_intermeds=sow_flax_intermeds,
                                                           return_all_matrices = False,
                                                           return_intermeds = False)
        
        
        ### calculate joint loglike using 1D forward algorithm over latent site 
        ###   classes
        logprob_emit_at_indel = scoring_matrices_dict['logprob_emit_at_indel'] #(C_frag, A)
        joint_logprob_emit_at_match = scoring_matrices_dict['joint_logprob_emit_at_match'] #(T, C_frag, A, A) 
        joint_logprob_transit =  scoring_matrices_dict['all_transit_matrices']['joint'] #(T, C_frag, S, S)
        joint_logprob_perSamp_maybePerTime = joint_only_forward(aligned_inputs = aligned_inputs,
                                                 joint_logprob_emit_at_match = joint_logprob_emit_at_match,
                                                 logprob_emit_at_indel = logprob_emit_at_indel,
                                                 joint_logprob_transit = joint_logprob_transit,
                                                 unique_time_per_sample = unique_time_per_sample,
                                                 return_all_intermeds = False)  #(T, B)  or (B,)
        
        ### marginalize over times where needed
        if (not unique_time_per_sample) and (t_array.shape[0] > 1):
            joint_neg_logP = -marginalize_over_times(logprob_perSamp_perTime = joint_logprob_perSamp_maybePerTime,
                                                      exponential_dist_param = self.exponential_dist_param,
                                                      t_array = times_for_matrices) #(B,)
             
        elif (not unique_time_per_sample) and (t_array.shape[0] == 1):
            joint_neg_logP = -joint_logprob_perSamp_maybePerTime[0,:] #(B,)
        
        elif unique_time_per_sample:
            joint_neg_logP = -joint_logprob_perSamp_maybePerTime #(B,)
            
            
        ### for REPORTING ONLY (not the objective function), normalize by length
        if self.norm_reported_loss_by == 'desc_len':
            # where descendant is not pad or gap
            banned_toks = np.array( [0,1,2,self.gap_idx] )
            
        elif self.norm_reported_loss_by == 'align_len':
            # where descendant is not pad (but could be gap)
            banned_toks = np.array( [0,1,2] )
        
        mask = ~jnp.isin( aligned_inputs[...,1], banned_toks)
        length_for_normalization = mask.sum(axis=1)
        joint_neg_logP_length_normed = joint_neg_logP / length_for_normalization
        del mask
        
        
        ### compile final outputs
        # aux dict
        aux_dict = {'joint_neg_logP': joint_neg_logP,
                    'joint_neg_logP_length_normed': joint_neg_logP_length_normed,
                    'align_length_for_normalization': length_for_normalization}
        
        # final loss
        loss = jnp.mean( aux_dict['joint_neg_logP'] )
        
        return loss, aux_dict
    
    
    def calculate_all_loglikes(self,
                               batch,
                               t_array):
        """
        Use this during final eval
        
        
        C_dom: number of mixtures associated with nested TKF92 models (domain-level)
            > here, C_dom = 1
        C_frag: number of mixtures associated with TKF92 fragments (fragment-level)
        C_trans: C_dom * C_frag = C_frag
        B = batch size; number of samples
        T = number of branch lengths; this could be: 
            > an array of times for all samples (T; marginalize over these later)
            > an array of time per sample (T=B)
            > a quantized array of times per sample (T = T', where T' <= T)
        S: number of transition states (4 here: M, I, D, start/end)
        A: emission alphebet size (20 for proteins)
        
        
        returns all four loglikelihoods in a dictionary:
        
        1.) 'joint_neg_logP': P(anc, desc, align)
        2.) 'joint_neg_logP_length_normed': P(anc, desc, align), normalized 
            by desired length (set by self.norm_by)
        3.) 'anc_neg_logP': P(anc)
        4.) 'anc_neg_logP_length_normed': P(anc), normalized by ancestor 
            length
        5.) 'desc_neg_logP': P(desc)
        6.) 'desc_neg_logP_length_normed': P(desc), normalized by descendant 
            length
        7.) 'cond_neg_logP': P(desc, align | anc)
        8.) 'cond_neg_logP_length_normed': P(desc, align | anc), normalized 
            by desired length (set by self.norm_by)
        
        Calculate joint and sequence marginals in one jax.lax.scan operation
        """
        aligned_inputs = batch[0]
        
        # which times to use for scoring matrices
        if self.times_from =='t_per_sample':
            times_for_matrices = batch[1] #(B,)
            unique_time_per_sample = True
        
        elif self.times_from in ['geometric','t_array_from_file']:
            times_for_matrices = t_array #(T,)
            unique_time_per_sample = False
            
        # get lengths, not including <bos> and <eos>
        align_len = ~jnp.isin( aligned_inputs[...,0], np.array( [0,1,2] ) )
        anc_len = ~jnp.isin( aligned_inputs[...,0], np.array( [0,1,2,self.gap_idx] ) )
        desc_len = ~jnp.isin( aligned_inputs[...,1], np.array( [0,1,2,self.gap_idx] ) )
        align_len = align_len.sum(axis=1)
        anc_len = anc_len.sum(axis=1)
        desc_len = desc_len.sum(axis=1)
        
        # score matrices
        # 
        # scoring_matrices_dict has the following keys (when return_intermeds is False)
        #   logprob_emit_at_indel, (C_frag, A)
        #   joint_logprob_emit_at_match, (T, C_frag, A, A)
        #   all_transit_matrices, dict, with joint transit matrix being (T, C_frag, C_frag, S, S) 
        scoring_matrices_dict = self._get_scoring_matrices( t_array=times_for_matrices,
                                                            sow_flax_intermeds=False,
                                                            return_all_matrices = True,
                                                            return_intermeds = False)
        
        
        ### get all log-likelihoods
        logprob_emit_at_indel = scoring_matrices_dict['logprob_emit_at_indel']
        joint_logprob_emit_at_match = scoring_matrices_dict['joint_logprob_emit_at_match']
        all_transit_matrices =  scoring_matrices_dict['all_transit_matrices']
        out = all_loglikes_forward( aligned_inputs = aligned_inputs,
                                    joint_logprob_emit_at_match = joint_logprob_emit_at_match,
                                    logprob_emit_at_indel = logprob_emit_at_indel,
                                    all_transit_matrices = all_transit_matrices,
                                    unique_time_per_sample = unique_time_per_sample )
        
        # for joint loglike: marginalize over times where needed
        if (not unique_time_per_sample) and (t_array.shape[0] > 1):
            joint_logprob_perSamp_maybePerTime = out['joint_neg_logP']  #(T,B)
            overwrite_joint_neg_logP = -marginalize_over_times(logprob_perSamp_perTime = -joint_logprob_perSamp_maybePerTime,
                                                     exponential_dist_param = self.exponential_dist_param,
                                                     t_array = times_for_matrices) #(B,)
            out['joint_neg_logP'] = overwrite_joint_neg_logP #(B,)
             
        elif (not unique_time_per_sample) and (t_array.shape[0] == 1):
            out['joint_neg_logP'] = out['joint_neg_logP'][0,:] #(B,)
        
        
        ### conditional comes from joint / anc
        cond_neg_logP = - (-out['joint_neg_logP'] - -out['anc_neg_logP'])
        
        
        ### for REPORTING ONLY (not the objective function), normalize by length
        anc_neg_logP_length_normed = out['anc_neg_logP'] / anc_len
        desc_neg_logP_length_normed = out['desc_neg_logP'] / desc_len
        
        if self.norm_reported_loss_by == 'desc_len':
            joint_neg_logP_length_normed = out['joint_neg_logP'] / desc_len
            cond_neg_logP_length_normed = cond_neg_logP / desc_len
        
        elif self.norm_reported_loss_by == 'align_len':
            joint_neg_logP_length_normed = out['joint_neg_logP'] / align_len
            cond_neg_logP_length_normed = cond_neg_logP / align_len
        
        to_add = { 'joint_neg_logP_length_normed': joint_neg_logP_length_normed,
                   'anc_neg_logP_length_normed': anc_neg_logP_length_normed,
                   'desc_neg_logP_length_normed': desc_neg_logP_length_normed,
                   'cond_neg_logP': cond_neg_logP,
                   'cond_neg_logP_length_normed': cond_neg_logP_length_normed
                }
        
        return {**out, **to_add}
    
    
    def _get_emission_scoring_matrices( self,
                                        log_transit_class_probs,
                                        t_array,
                                        sow_flax_intermeds: bool,
                                        return_all_matrices: bool,
                                        return_intermeds: bool ):
        """
        Matrices needed to score emissions: substitution rate matrices, 
            equilibrium distributions, etc.
        
        
        C_dom: number of mixtures associated with nested TKF92 models (domain-level)
        C_frag: number of mixtures associated with TKF92 fragments (fragment-level)
        C_tr: number of transition mixtures, C_dom * C_frag = C_tr
        
        B = batch size; number of samples
        T = number of branch lengths; this could be: 
            > an array of times for all samples (T; marginalize over these later)
            > an array of time per sample (T=B)
            > a quantized array of times per sample (T = T', where T' <= T)
        A: emission alphabet size (20 for proteins)
        S: number of transition states (4 here: M, I, D, start/end)
           
        
        Arguments
        ----------
        log_transit_class_probs : ArrayLike, (C_tr,)
            P(C_tr, c_dom); MUST be a vector!!! (i.e. 1D)
        
        t_array : ArrayLike, (T,)
            branch lengths, times for marginalizing over
        
        return_all_matrices : bool
            if false, only return the joint (used for model training)
            if true, return joint, conditional, and marginal matrices
        
        return_intermeds : bool
            return other intermediates
        
        sow_flax_intermeds : bool
            switch for tensorboard logging
          
        Returns
        -------
        out_dict : dict
        
            always returns:
                out_dict['logprob_emit_at_indel'] : (C_tr, A)
                out_dict['joint_logprob_emit_at_match'] : (T, C_tr, A, A)
            
            if return_all_matrices:
                out_dict['cond_logprob_emit_at_match'] : (T, C_tr, A, A)
            
            if return_intermeds:
                out_dict['log_equl_dist_per_mixture'] : (C_tr, C_sites, A)
                out_dict['rate_multipliers'] : (C_tr, C_sites, C_subrate)
                out_dict['rate_matrix'] : (C_tr, C_sites, C_subrate)
                out_dict['exchangeabilities'] : (A, A)
                out_dict['cond_subst_logprobs_per_mixture'] : (T, C_tr, C_sites, C_subrate, A, A)
                out_dict['joint_subst_logprobs_per_mixture'] : (T, C_tr, C_sites, C_subrate, A, A)
                out_dict['log_site_class_probs'] : (C_tr, C_sites)
                out_dict['log_rate_mult_probs'] : (C_tr, C_sites, C_subrate)
        """
        ###########################################################
        ### build log-transformed equilibrium distribution; get   #
        ### site-level mixture probability P(c_site | c_tr)       #
        ###########################################################
        # log_site_class_probs is (C_tr, C_sites)
        # log_equl_dist_per_mixture is (C_tr, C_sites, A)
        out = self.equl_dist_module(sow_flax_intermeds = sow_flax_intermeds) 
        log_site_class_probs, log_equl_dist_per_mixture = out
        del out
        
        # P(x) = \sum_c P(c) * P(x|c)
        logprob_emit_at_indel = lse_over_equl_logprobs_per_mixture( log_site_class_probs = log_site_class_probs,
                                                                    log_equl_dist_per_mixture = log_equl_dist_per_mixture ) #(C_tr, A)
        
        
        ####################################################
        ### site rate multipliers, and probabilities for   #
        ### selecting a rate multiplier from the mixture   #
        ####################################################
        # Substitution rate multipliers
        # both are (C_tr=C_frag, C_sites, K)
        # log_frag_class_probs needed in case you're normalizing rate multipliers
        log_rate_mult_probs, rate_multipliers = self.rate_mult_module( sow_flax_intermeds = sow_flax_intermeds,
                                                                       log_site_class_probs = log_site_class_probs,
                                                                       log_transit_class_probs = log_transit_class_probs ) 
        
        
        ####################################################
        ### build substitution log-probability matrix      #
        ### use this to score emissions from match sites   #
        ####################################################
        # cond_logprobs_per_mixture is (T, C_tr, C_sites, C_subrate, A, A) 
        # subst_module_intermeds is a dictionary of intermediates
        out = self.logprob_subst_module( log_equl_dist = log_equl_dist_per_mixture,
                                         rate_multipliers = rate_multipliers,
                                         t_array = t_array,
                                         sow_flax_intermeds = sow_flax_intermeds,
                                         return_cond = True,
                                         return_intermeds = return_intermeds )        
        cond_subst_logprobs_per_mixture, subst_module_intermeds = out
        del out
        
        # get the joint probability
        # joint_subst_logprobs_per_mixture is (T, C_tr, C_sites, C_subrate, A, A)
        joint_subst_logprobs_per_mixture = joint_logprob_emit_at_match_per_mixture( cond_logprob_emit_at_match_per_mixture = cond_subst_logprobs_per_mixture,
                                                                              log_equl_dist_per_mixture = log_equl_dist_per_mixture ) 
        
        # marginalize over classes and possible rate multipliers
        joint_logprob_emit_at_match = lse_over_match_logprobs_per_mixture(log_site_class_probs = log_site_class_probs,
                                                            log_rate_mult_probs = log_rate_mult_probs,
                                                            logprob_emit_at_match_per_mixture = joint_subst_logprobs_per_mixture) # (T, C_tr, A, A)
        
        if return_all_matrices:
            cond_logprob_emit_at_match = lse_over_match_logprobs_per_mixture(log_site_class_probs = log_site_class_probs,
                                                            log_rate_mult_probs = log_rate_mult_probs,
                                                            logprob_emit_at_match_per_mixture = cond_subst_logprobs_per_mixture) # (T, C_tr, A, A)
            
        #####################
        ### decide output   #
        #####################
        # always returned (at training, at final eval, etc.)
        out_dict = {'logprob_emit_at_indel': logprob_emit_at_indel, #(C_tr, A)
                    'joint_logprob_emit_at_match': joint_logprob_emit_at_match } #(T, C_tr, A, A)
        
        # returned if you need conditional and marginal logprob matrices
        if return_all_matrices:
            out_dict['cond_logprob_emit_at_match'] = cond_logprob_emit_at_match #(T, C_tr, A, A)
        
        # all intermediates
        if return_intermeds:
            to_add = {'log_equl_dist_per_mixture': log_equl_dist_per_mixture, #(C_tr, C_sites, A)
                      'rate_multipliers': rate_multipliers, #(C_tr, C_sites, C_subrate)
                      'rate_matrix': subst_module_intermeds.get('rate_matrix',None), #(C_tr, C_sites, A, A) or None
                      'exchangeabilities': subst_module_intermeds.get('exchangeabilities',None), #(A,A) or None
                      'cond_subst_logprobs_per_mixture': cond_subst_logprobs_per_mixture, #(T, C_tr, C_sites, C_subrate, A, A)
                      'joint_subst_logprobs_per_mixture': joint_subst_logprobs_per_mixture, #(T, C_tr, C_sites, C_subrate, A, A)
                      'log_site_class_probs': log_site_class_probs, #(C_tr, C_sites)
                      'log_rate_mult_probs': log_rate_mult_probs } #(C_tr, C_sites, C_subrate)
            out_dict = {**out_dict, **to_add}
        
        return out_dict
        
    
    def _get_scoring_matrices( self,
                               t_array,
                               sow_flax_intermeds: bool,
                               return_all_matrices: bool,
                               return_intermeds: bool):
        """
        C_dom: number of mixtures associated with nested TKF92 models (domain-level)
            > here, this is one
        C_frag: number of mixtures associated with TKF92 fragments (fragment-level)
        C_tr: number of transition mixtures, C_dom * C_frag = C_tr
        
        B = batch size; number of samples
        T = number of branch lengths; this could be: 
            > an array of times for all samples (T; marginalize over these later)
            > an array of time per sample (T=B)
            > a quantized array of times per sample (T = T', where T' <= T)
        A: emission alphabet size (20 for proteins)
        S: number of transition states (4 here: M, I, D, start/end)
           
        
        Arguments
        ----------
        t_array : ArrayLike, (T,)
            branch lengths, times for marginalizing over
        
        return_all_matrices : bool
            if false, only return the joint (used for model training)
            if true, return joint, conditional, and marginal matrices
        
        return_intermeds : bool
            return other intermediates
        
        sow_flax_intermeds : bool
            switch for tensorboard logging
          
        Returns
        -------
        out_dict : dict
        
            always returns:
                out_dict['logprob_emit_at_indel'] : (C_frag, A)
                out_dict['joint_logprob_emit_at_match'] : (T, C_frag, A, A)
                out_dict['all_transit_matrices'] : dict
            
            if return_all_matrices:
                out_dict['cond_logprob_emit_at_match'] : (T, C_frag, A, A)
            
            if return_intermeds:
                out_dict['log_equl_dist_per_mixture'] : (C_frag, C_sites, A)
                out_dict['rate_multipliers'] : (C_frag, C_sites, K)
                out_dict['rate_matrix'] : (C_frag, C_sites, K)
                out_dict['exchangeabilities'] : (A, A)
                out_dict['cond_subst_logprobs_per_mixture'] : (T, C_frag, C_sites, C_subrate, A, A)
                out_dict['joint_subst_logprobs_per_mixture'] : (T, C_frag, C_sites, C_subrate, A, A)
                out_dict['log_fragment_class_probs'] : (C_dom, C_frag)
                out_dict['log_site_class_probs'] : (C_frag, C_sites)
                out_dict['log_rate_mult_probs'] : (C_frag, C_sites, K)
            
        """
        ######################################
        ### scoring matrix for TRANSITIONS   #
        ######################################
        out = self.transitions_module( t_array = t_array,
                                       return_all_matrices = return_all_matrices,
                                       sow_flax_intermeds = sow_flax_intermeds ) 
        # P(c_fragment | c_domain)
        log_frag_class_probs = out[0] #(C_dom=1, C_frag)
        
        # all_transit_matrices['joint']: (T, C_dom=1, C_{frag_from}, C_{frag_to}, S_from, S_to)
        # 
        # if return_all_matrices is True, also include:
        # all_transit_matrices['conditional']: (T, C_dom=1, C_{frag_from}, C_{frag_to}, S_from, S_to)
        # all_transit_matrices['marginal']: (C_dom=1, C_{frag_from}, C_{frag_to}, 2, 2)
        all_transit_matrices = out[1]
        
        # remove unused dims, since C_dom=1
        log_frag_class_probs = log_frag_class_probs[0,...] #(C_frag,)
        all_transit_matrices['joint'] = all_transit_matrices['joint'][:,0,...] # (T, C_{frag_from}, C_{frag_to}, S_from, S_to)
        if return_all_matrices:
            all_transit_matrices['conditional'] = all_transit_matrices['conditional'][:,0,...] # (T, C_{frag_from}, C_{frag_to}, S_from, S_to)
            all_transit_matrices['marginal'] = all_transit_matrices['marginal'][0,...] # (C_{frag_from}, C_{frag_to}, S_from, S_to)
            
        
        ####################################
        ### scoring matrix for EMISSIONS   #
        ####################################
        # always returns:
        #     out_dict['logprob_emit_at_indel'] : (C_tr, A)
        #     out_dict['joint_logprob_emit_at_match'] : (T, C_tr, A, A)
        
        # if return_all_matrices:
        #     out_dict['cond_logprob_emit_at_match'] : (T, C_tr, A, A)
        
        # if return_intermeds:
        #     out_dict['log_equl_dist_per_mixture'] : (C_tr, C_sites, A)
        #     out_dict['rate_multipliers'] : (C_tr, C_sites, C_subrate)
        #     out_dict['rate_matrix'] : (C_tr, C_sites, C_subrate)
        #     out_dict['exchangeabilities'] : (A, A)
        #     out_dict['cond_subst_logprobs_per_mixture'] : (T, C_tr, C_sites, C_subrate, A, A)
        #     out_dict['joint_subst_logprobs_per_mixture'] : (T, C_tr, C_sites, C_subrate, A, A)
        #     out_dict['log_site_class_probs'] : (C_tr, C_sites)
        #     out_dict['log_rate_mult_probs'] : (C_tr, C_sites, C_subrate)
        out_dict = self._get_emission_scoring_matrices( log_transit_class_probs = log_frag_class_probs,
                                                        t_array = t_array,
                                                        sow_flax_intermeds = sow_flax_intermeds,
                                                        return_all_matrices = return_all_matrices, 
                                                        return_intermeds = return_intermeds)
        
        
        ##################################
        ### add to out_dict and return   #
        ##################################
        out_dict['all_transit_matrices'] = all_transit_matrices
        
        if return_intermeds:
            out_dict['tkf_param_dict'] = out[2]
            out_dict['log_frag_class_probs'] = log_frag_class_probs #(C_frag,)
        
        return out_dict
    
    
    def write_params(self,
                     t_array,
                     out_folder: str,
                     prefix: str,
                     write_time_static_objs: bool):
        #########################################################
        ### only write once: activations_times_used text file   #
        #########################################################
        if write_time_static_objs:
            with open(f'{out_folder}/activations_and_times_used.tsv','w') as g:
                if self.times_from in ['geometric','t_array_from_file']:
                    g.write(f't_array for all samples; possible marginalized over them\n')
                    g.write(f'{t_array}')
                    g.write('\n')
                
                elif self.times_from == 't_per_sample':
                    g.write(f'one branch length for each sample; times used for {prefix}:\n')
                    g.write(f'{t_array}')
                    g.write('\n')
        
        
        ###################################
        ### always write: Full matrices   #
        ###################################
        out = self._get_scoring_matrices(t_array=t_array,
                                        sow_flax_intermeds=False,
                                        return_all_matrices=True,
                                        return_intermeds=True)
        
        # final conditional and joint prob of match (before and after LSE over site/rate mixtures)
        for key in ['joint_logprob_emit_at_match',
                    'cond_subst_logprobs_per_mixture',
                    'joint_subst_logprobs_per_mixture']:
            mat = np.exp ( out[key] )
            new_key = f'{prefix}_{key}'.replace('log','')
            write_matrix_to_npy( out_folder, mat, new_key )
            maybe_write_matrix_to_ascii( out_folder, mat, new_key )
            del mat, new_key
            
        # transition matrix
        for loss_type in ['joint','conditional','marginal']:
            mat = np.exp(out['all_transit_matrices'][loss_type]) 
            new_key = f'{prefix}_{loss_type}_prob_transit_matrix'
            write_matrix_to_npy( out_folder, mat, new_key )
            maybe_write_matrix_to_ascii( out_folder, mat, new_key )
            del mat, new_key
        
        
        #####################################################################
        ### only write once: parameters, things that don't depend on time   #
        #####################################################################
        if write_time_static_objs:
            ### class probs
            # fragment class probs; P(c_frag | c_dom)
            if self.num_fragment_mixtures > 1:
                log_frag_class_probs = out['log_frag_class_probs'] #(C_dom, C_frag,)
                mat = np.exp( log_frag_class_probs )
                key = f'{prefix}_frag_class_probs'
                write_matrix_to_npy( out_folder, mat, key )
                maybe_write_matrix_to_ascii( out_folder, mat, key )
                del key, mat, log_frag_class_probs
            
            # site class probs; P( C_site | C_transit )
            if self.num_site_mixtures > 1:
                site_class_probs = np.exp(out['log_site_class_probs']) #(C_tr, C_sites)
                key = f'{prefix}_site_class_probs'
                write_matrix_to_npy( out_folder, site_class_probs, key )
                maybe_write_matrix_to_ascii( out_folder, site_class_probs, key )
                del key, site_class_probs
            
            # probability of substitution rate multiplier;
            # P( C_subrate | C_site, C_transit )
            if self.num_rate_mults > 1:
                rate_mult_probs = np.exp(out['log_rate_mult_probs']) #(C_tr, C_sites, C_subrate)
                key = f'{prefix}_rate_mult_probs'
                write_matrix_to_npy( out_folder, rate_mult_probs, key )
                maybe_write_matrix_to_ascii( out_folder, rate_mult_probs, key )
                del key            
            
                
            ### substitution rate matrix (BEFORE scaling by rate multipliers!)
            rate_matrix = out['rate_matrix'] #(C_frag, C_sites, A, A) or None
            if rate_matrix is not None:
                key = f'{prefix}_rate_matrix'
                write_matrix_to_npy( out_folder, rate_matrix, key )
                del key

                for c_fr in range(rate_matrix.shape[0]):
                    for c_s in range(rate_matrix.shape[1]):
                        mat_to_save = rate_matrix[c_fr, c_s, ...]
                        key = f'{prefix}_frag-class-{c_fr}_site-class-{c_s}_rate_matrix'
                        maybe_write_matrix_to_ascii( out_folder, mat_to_save, key )
                        del mat_to_save, key
                        
            
            ### equilibrium distribution
            # (BEFORE marginalizing over site clases)
            equl_dist = np.exp(out['log_equl_dist_per_mixture']) #(C_tr, C_sites, A)
            key = f'{prefix}_equilibriums-per-site-class'
            write_matrix_to_npy( out_folder, equl_dist, key )
            maybe_write_matrix_to_ascii( out_folder, equl_dist, key )
            del key, equl_dist
            
            # AFTER marginalizing out site and rate mixtures
            mat = np.exp( out['logprob_emit_at_indel'] ) #(C_frag, A)
            new_key = f'{prefix}_logprob_emit_at_indel'.replace('log','')
            write_matrix_to_npy( out_folder, mat, new_key )
            maybe_write_matrix_to_ascii( out_folder, mat, new_key )
            del mat, new_key
            
        
            ### rate multipliers 
            # \rho_{c,k} or \rho_k
            if not self.rate_mult_module.use_unit_rate_mult:
                rate_multipliers = out['rate_multipliers'] #(C_frag, C_sites, K)
                key = f'{prefix}_rate_multipliers'
                write_matrix_to_npy( out_folder, rate_multipliers, key )
                maybe_write_matrix_to_ascii( out_folder, rate_multipliers, key )
                del key
                
            
            ### exchangeabilities, if gtr or hky85
            exchangeabilities = out['exchangeabilities'] #(A, A) or None
            
            if self.subst_model_type == 'gtr':
                key = f'{prefix}_gtr-exchangeabilities'
                write_matrix_to_npy( out_folder, exchangeabilities, key )
                maybe_write_matrix_to_ascii( out_folder, exchangeabilities, key )
                del key
                
            elif self.subst_model_type == 'hky85':
                ti = exchangeabilities[0, 2]
                tv = exchangeabilities[0, 1]
                arr = np.array( [ti, tv] )
                key = f'{prefix}_hky85_ti_tv'
                write_matrix_to_npy( out_folder, arr, key )
                
                with open(f'{out_folder}/ASCII_{prefix}_hky85_ti_tv.txt','w') as g:
                    g.write(f'transition rate, ti: {ti}\n')
                    g.write(f'transition rate, tv: {tv}')
                del key, arr
              
                
            ### indel parameters
            if self.config['load_all']:
                lam = self.transitions_module.param_dict['lambda']
                mu = self.transitions_module.param_dict['mu']
                offset = 1 - (lam/mu)
                
            elif not self.config['load_all']:
                # lambda and mu
                mu_min_val = self.transitions_module.mu_min_val #float
                mu_max_val = self.transitions_module.mu_max_val #float
                offs_min_val = self.transitions_module.offs_min_val #float
                offs_max_val = self.transitions_module.offs_max_val #float
                mu_offset_logits = self.transitions_module.tkf_mu_offset_logits #(1,2)
            
                mu = bound_sigmoid(x = mu_offset_logits[0,0],
                                   min_val = mu_min_val,
                                   max_val = mu_max_val).item() #float
                
                offset = bound_sigmoid(x = mu_offset_logits[0,1],
                                         min_val = offs_min_val,
                                         max_val = offs_max_val).item() #float
                lam = mu * (1 - offset)  #float
            
            
            with open(f'{out_folder}/ASCII_{prefix}_tkf92_indel_params.txt','w') as g:
                g.write(f'insert rate, lambda: {lam}\n')
                g.write(f'deletion rate, mu: {mu}\n')
                g.write(f'offset: {offset}\n\n')
            
            out_dict = {'lambda': np.array(lam), # shape=()
                        'mu': np.array(mu), # shape=()
                        'offset': np.array(offset)} # shape=()
                            
            # tkf92 r_ext param
            if self.config['load_all']:
                r_extend = self.transitions_module.param_dict['r_extend']
                
            elif not self.config['load_all']:
                r_extend_min_val = self.transitions_module.r_extend_min_val
                r_extend_max_val = self.transitions_module.r_extend_max_val
                r_extend_logits = self.transitions_module.r_extend_logits #(C_dom=1,C_frag)
                
                r_extend = bound_sigmoid(x = r_extend_logits,
                                         min_val = r_extend_min_val,
                                         max_val = r_extend_max_val) #(C_dom=1,C_frag)
                
            mean_indel_lengths = 1 / (1 - r_extend) #(C_dom=1,C_frag)
            
            with open(f'{out_folder}/ASCII_{prefix}_tkf92_indel_params.txt','a') as g:
                g.write(f'extension prob, r: ')
                [g.write(f'{elem}\t') for elem in r_extend.flatten()]
                g.write('\n')
                g.write(f'mean indel length: ')
                [g.write(f'{elem}\t') for elem in mean_indel_lengths]
                g.write('\n')
            
            out_dict['r_extend'] = r_extend #(C_dom=1,C_frag)
        
            with open(f'{out_folder}/PARAMS-DICT_{prefix}_tkf92_indel_params.pkl','wb') as g:
                pickle.dump(out_dict, g)
            del out_dict

class FragAndSiteClassesLoadAll(FragAndSiteClasses):
    """
    same as FragAndSiteClasses, but load all parameters from files (excluding time, 
        exponential distribution parameter)
    
    
    Initialize with
    ----------------
    config : dict    
        config['num_fragment_mixtures'] :  int
            number of fragment classes (for transitions)
    
        config['num_site_mixtures'] :  int
            number of emission site classes
        
        config['subst_model_type'] : {gtr, f81}
            which substitution model
        
        config['norm_loss_by'] :  {desc_len, align_len}, optional
            what length to normalize loglikelihood by
            Default is 'desc_len'
        
        config['exponential_dist_param'] : int, optional
            There is an exponential prior over time; this provides the
            parameter for this during marginalization over times
            Default is 1
        
        config['times_from'] : {geometric, t_array_from_file, t_per_sample}
        
        config['filenames'] : files of parameters to load
        
    name : str
        class name, for flax
    
    Main methods here
    -----------------
    setup
    
    
    Inherited from FragAndSiteClasses
    ----------------------------------
    __call__
        unpack batch and calculate logP(anc, desc, align)
    
    calculate_all_loglikes
        calculate logP(anc, desc, align), logP(anc), logP(desc), and
        logP(desc, align | anc)
    
    write_params
        write parameters to files
        
    _get_emission_scoring_matrices
    
    _get_scoring_matrices
    
    
    Methods inherited from neural_models.model_utils.BaseClasses.ModuleBase
    -----------------------------------------------------------------
    sow_flax_intermeds
        for tensorboard logging
    """
    config: dict
    name: str
    
    def setup(self):
        assert self.config['num_domain_mixtures'] == 1
        
        ###################
        ### read config   #
        ###################
        # required
        self.num_domain_mixtures = 1
        self.num_fragment_mixtures = self.config['num_fragment_mixtures']
        self.num_site_mixtures = self.config['num_site_mixtures']
        self.num_rate_mults = self.config['k_rate_mults']
        self.num_transit_mixtures = ( self.config['num_fragment_mixtures'] *
                                      self.config['num_domain_mixtures'] )# C_tr

        self.subst_model_type = self.config['subst_model_type']
        self.times_from = self.config['times_from']
        
        # optional
        self.indp_rate_mults = self.config.get('indp_rate_mults', False)
        self.gap_idx = self.config.get('gap_idx', 43)
        self.norm_reported_loss_by = 'desc_len'
        self.exponential_dist_param = 1
        
        
        ########################################################
        ### module for transition probabilities, and the       #
        ### fragment-level mixture weights P(c_frag | c_dom)   #
        ########################################################
        self.transitions_module = TKF92TransitionLogprobsFromFile(config = self.config,
                                                 name = f'tkf92 indel model')
        
        
        ###############################################################
        ### probability of having a particular subsitution rate       #
        ### rate multiplier, and the rate multipliers themselves      #
        ###############################################################
        if not self.indp_rate_mults:
            self.rate_mult_module = RateMultipliersPerClassFromFile(config = self.config,
                                                      name = f'get rate multipliers')
        
        elif self.indp_rate_mults:
            self.rate_mult_module = IndpRateMultipliersFromFile(config = self.config,
                                                      name = f'get rate multipliers')
        
        
        ###############################################################
        ### module for equilibrium distribution, and the site-level   # 
        ### mixture weights P(c_sites | c_frag)                       #
        ###############################################################
        self.equl_dist_module = EqulDistLogprobsFromFile(config = self.config,
                                                       name = f'get equilibrium')
        
        
        ###########################################
        ### module for substitution rate matrix   #
        ###########################################
        if self.subst_model_type == 'gtr':
            self.logprob_subst_module = GTRLogprobsFromFile( config = self.config,
                                                  name = f'gtr subst. model' )
            
        elif self.subst_model_type == 'f81':
            self.logprob_subst_module = F81LogprobsFromFile( config = self.config,
                                                     name = f'f81 subst. model' )


