#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 04:33:00 2025


"""
import numpy as np
import pickle

# jumping jax and leaping flax
from flax import linen as nn
import jax
from jax._src.typing import Array, ArrayLike
import jax.numpy as jnp
from jax.scipy.linalg import expm
from jax.scipy.special import logsumexp

from utils.BaseClasses import ModuleBase
from latent_class_mixtures.emission_models import (EqulDistLogprobsFromCounts,
                                                    EqulDistLogprobsPerClass,
                                                    EqulDistLogprobsFromFile,
                                                    GTRLogprobs,
                                                    GTRLogprobsFromFile,
                                                    LG08Logprobs,
                                                    RateMultipliersPerClass,
                                                    IndpRateMultipliers,
                                                    RateMultipliersPerClassFromFile,
                                                    IndpRateMultipliersFromFile,
                                                    F81Logprobs,
                                                    F81LogprobsFromFile)
from latent_class_mixtures.transition_models import (TKF91TransitionLogprobs,
                                                    TKF92TransitionLogprobs,
                                                    TKF91TransitionLogprobsFromFile,
                                                    TKF92TransitionLogprobsFromFile,
                                                    GeomLenTransitionLogprobs,
                                                    GeomLenTransitionLogprobsFromFile)
from latent_class_mixtures.model_functions import (bound_sigmoid,
                                                    safe_log,
                                                    joint_logprob_emit_at_match_per_mixture,
                                                    lse_over_match_logprobs_per_mixture,
                                                    lse_over_equl_logprobs_per_mixture,
                                                    joint_prob_from_counts,
                                                    anc_marginal_probs_from_counts,
                                                    desc_marginal_probs_from_counts,
                                                    write_matrix_to_npy,
                                                    maybe_write_matrix_to_ascii)


class IndpSites(ModuleBase):
    """
    pairHMM that finds joint loglikelihood of alignments, P(Anc, Desc, Align)
    
    
    C_dom: number of mixtures associated with nested TKF92 models (domain-level)
        > here, C_dom = 1
    C_frag: number of mixtures associated with TKF92 fragments (fragment-level)
        > here, C_frag = 1
    C_trans: C_dom * C_frag 1
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
            (one fragment type here)
        
        config['num_site_mixtures'] : int
            number of mixtures associated with the EMISSIONS
        
        config['indp_rate_mults'] :  bool
            if true, then rate multipliers are independent from latent
            site classes; P(k|c_tr, c_sites) = P(k) and \rho_{c_tr, c_sites, k} = \rho{k}
        
        config['subst_model_type'] : {gtr, f81}
            which substitution model
        
        config['indel_model_type'] : {tkf91, tkf92, None}
            which indel model, if any
            
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
    
    
    Methods inherited from neural_models.model_utils.BaseClasses.ModuleBase
    -----------------------------------------------------------------
    sow_flax_intermeds
        for tensorboard logging
    """
    config: dict
    name: str
    
    def setup(self):
        # not set up for domain or fragment level mixtures
        assert self.config['num_fragment_mixtures'] == 1
        assert self.config['num_domain_mixtures'] == 1
        
        ###################
        ### read config   #
        ###################
        # required
        self.num_site_mixtures = self.config['num_site_mixtures']
        self.num_rate_mults = self.config['k_rate_mults']
        self.num_domain_mixtures = 1
        self.num_fragment_mixtures = 1
        self.num_transit_mixtures = 1 # C_tr
        
        self.subst_model_type = self.config['subst_model_type'].lower()
        indel_model_type = self.config['indel_model_type']
        self.indel_model_type = indel_model_type.lower() if indel_model_type is not None else None
        self.times_from = self.config['times_from'].lower()
        
        # hard coded or optional
        self.indp_rate_mults = self.config.get('indp_rate_mults', False)
        self.norm_reported_loss_by = 'desc_len'
        self.exponential_dist_param = 1
        
        
        ###################################################################
        ### module for transition probabilities, and the fragment-level   #
        ### mixture weights P(c_frag | c_dom)                             #
        ###################################################################
        if self.indel_model_type is None:
            self.transitions_module = GeomLenTransitionLogprobs(config = self.config,
                                                     name = f'geom seq lengths model')
            
        elif self.indel_model_type == 'tkf91':
            self.transitions_module = TKF91TransitionLogprobs(config = self.config,
                                                     name = f'tkf91 indel model')
        
        elif self.indel_model_type == 'tkf92':
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
        ### mixture weights P(c_sites | c_frag*c_dom)                 #
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

        elif self.subst_model_type == 'lg08_gtr':
            self.logprob_subst_module = LG08Logprobs( config = self.config,
                                                      name = 'lg08-gtr susbt. model')

    def __call__(self,
                 batch,
                 t_array,
                 sow_flax_intermeds: bool):
        """
        Use this during active model training


        C_dom: number of mixtures associated with nested TKF92 models (domain-level)
            > here, C_dom = 1
        C_frag: number of mixtures associated with TKF92 fragments (fragment-level)
            > here, C_frag = 1
        C_trans: C_dom * C_frag 1
        B = batch size; number of samples
        T = number of branch lengths; this could be: 
            > an array of times for all samples (T; marginalize over these later)
            > an array of time per sample (T=B)
            > a quantized array of times per sample (T = T', where T' <= T)
        S: number of transition states (4 here: M, I, D, start/end)
        A: emission alphebet size (20 for proteins)
        
        
        Returns
        -------
        loss: average across the batch, based on joint log-likelihood
                
        aux_dict: has the following keys and values
          1.) 'joint_neg_logP': sum down the length
          2.) 'joint_neg_logP_length_normed': sum down the length,  
              normalized by descendant length
        """
        # which times to use for scoring matrices
        if self.times_from =='t_per_sample':
            times_for_matrices = batch[4] #(B,)
        
        elif self.times_from in ['geometric','t_array_from_file']:
            times_for_matrices = t_array #(T,)

        # get the scoring matrices needed
        # 
        # scoring_matrices_dict has the following keys (when return_intermeds 
        # is False and return_all_matrices is False):
        #   logprob_emit_at_indel, (A, )
        #   joint_logprob_emit_at_match, (T, A, A)
        #   all_transit_matrices, dict, with joint transit matrix being (T, S, S)
        scoring_matrices_dict = self._get_scoring_matrices(t_array=times_for_matrices,
                                        sow_flax_intermeds=sow_flax_intermeds,
                                        return_all_matrices = False,
                                        return_intermeds = False)
        
        # calculate loglikelihoods; provide both batch and t_array, just in case
        # time marginalization hidden in joint_prob_from_counts function
        #
        # aux_dict has the following keys (when return_intermeds is False)
        #   joint_neg_logP (B)
        #   joint_neg_logP_length_normed (B)
        #   align_length_for_normalization (B,)
        aux_dict = joint_prob_from_counts( batch = batch,
                                           times_from = self.times_from,
                                           score_indels = False if self.indel_model_type is None else True,
                                           scoring_matrices_dict = scoring_matrices_dict,
                                           t_array = t_array,
                                           exponential_dist_param = self.exponential_dist_param,
                                           norm_reported_loss_by = self.norm_reported_loss_by,
                                           return_intermeds = False )

        loss = jnp.mean( aux_dict['joint_neg_logP'] )
        
        return loss, aux_dict
    
    
    def calculate_all_loglikes(self,
                               batch: tuple,
                               t_array: jnp.array,
                               return_intermeds: bool=False):
        """
        Use this during final eval
        
        C_dom: number of mixtures associated with nested TKF92 models (domain-level)
            > here, C_dom = 1
        C_frag: number of mixtures associated with TKF92 fragments (fragment-level)
            > here, C_frag = 1
        C_trans: C_dom * C_frag 1
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
    
        if returning intermediates, include extra things
        """
        # which times to use for scoring matrices
        if self.times_from =='t_per_sample':
            times_for_matrices = batch[4] #(B,)
        
        elif self.times_from in ['geometric','t_array_from_file']:
            times_for_matrices = t_array #(T,)

        # get the scoring matrices needed
        # 
        # scoring_matrices_dict has the following keys
        # (when return_intermeds is False and return_all_matrices is True)
        #   logprob_emit_at_indel, (A, )
        #   joint_logprob_emit_at_match, (T, A, A)
        #   cond_logprob_emit_at_match, (T, A, A)
        #   all_transit_matrices, dict
        scoring_matrices_dict = self._get_scoring_matrices(t_array=times_for_matrices,
                                        sow_flax_intermeds=False,
                                        return_all_matrices = True,
                                        return_intermeds=True)
        
        #########################
        ### joint probability   #
        #########################
        # time marginalization hidden in joint_prob_from_counts function
        #
        # aux_dict has the following keys 
        #   joint_neg_logP (B)
        #   joint_neg_logP_length_normed (B)
        #   align_length_for_normalization (B,)
        #
        # (extra keys when return_intermeds is True)
        #   joint_transit_score, (B,)
        #   joint_emission_score, (B,)
        aux_dict = joint_prob_from_counts( batch = batch,
                                           times_from = self.times_from,
                                           score_indels = False if self.indel_model_type is None else True,
                                           scoring_matrices_dict = scoring_matrices_dict,
                                           t_array = t_array,
                                           exponential_dist_param = self.exponential_dist_param,
                                           norm_reported_loss_by = self.norm_reported_loss_by,
                                           return_intermeds=True )
        
        
        #####################################
        ### ancestor marginal probability   #
        #####################################
        to_add = anc_marginal_probs_from_counts( batch = batch,
                                            score_indels = False if self.indel_model_type is None else True,
                                            scoring_matrices_dict = scoring_matrices_dict,
                                            return_intermeds=True  )
        
        aux_dict = {**aux_dict, **to_add}
        del to_add
        
        
        #######################################
        ### descendant marginal probability   #
        #######################################
        to_add = desc_marginal_probs_from_counts( batch = batch,
                                            score_indels = False if self.indel_model_type is None else True,
                                            scoring_matrices_dict = scoring_matrices_dict )
        
        aux_dict = {**aux_dict, **to_add}
        del to_add
        
        
        #############################
        ### calculate conditional   #
        #############################
        ### divide joint by anc marginal
        cond_neg_logP = -( -aux_dict['joint_neg_logP'] - -aux_dict['anc_neg_logP'] )
        length_for_normalization = aux_dict['align_length_for_normalization']
        cond_neg_logP_length_normed = cond_neg_logP / length_for_normalization
            
        aux_dict['cond_neg_logP'] = cond_neg_logP
        aux_dict['cond_neg_logP_length_normed'] = cond_neg_logP_length_normed
        
        
        return aux_dict
        
    
    def _get_scoring_matrices(self,
                             t_array,
                             sow_flax_intermeds: bool,
                             return_all_matrices: bool,
                             return_intermeds: bool): 
        """
        C_dom: number of mixtures associated with nested TKF92 models (domain-level)
            > here, this is one
        C_frag: number of mixtures associated with TKF92 fragments (fragment-level)
            > here, this is one
        C_trans: C_dom * C_frag = 1
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
                out_dict['logprob_emit_at_indel'] : (A,)
                out_dict['joint_logprob_emit_at_match'] : (T, A, A)
                out_dict['all_transit_matrices'] : dict
            
            if return_all_matrices:
                out_dict['cond_logprob_emit_at_match'] : (T, A, A)
            
            if return_intermeds:
                out_dict['log_equl_dist_per_mixture'] : (C_sites, A)
                out_dict['rate_multipliers'] : (C_sites, C_subrate)
                out_dict['rate_matrix'] : (C_sites, A, A)
                out_dict['exchangeabilities'] : (A, A)
                out_dict['cond_subst_logprobs_per_mixture'] : (T, C_sites, C_subrate A, A)
                out_dict['joint_subst_logprobs_per_mixture'] :  (T, C_sites, C_subrate A, A)
                out_dict['log_site_class_probs'] : (C_sites)
                out_dict['log_rate_mult_probs'] : (C_sites, C_subrate)
        """
        ######################################################
        ### build transition log-probability matrix, get     #
        ### fragment-level mixture probs P(c_frag | c_dom)   #
        ######################################################
        # if TKF91 model:
        #   all_transit_matrices['joint']: (T, S_from, S_to) 
        # 
        #   if return_all_matrices is True, also include:
        #   all_transit_matrices['conditional']: (T, S_from, S_to) 
        #   all_transit_matrices['marginal']: (2, 2)
        #   all_transit_matrices['log_corr']: (1, 1)
        
        # if TKF92 model:
        #   all_transit_matrices['joint']: (T, C_dom=1, C_{frag_from}=1, C_{frag_to}=1, S_from, S_to)
        # 
        #   if return_all_matrices is True, also include:
        #   all_transit_matrices['conditional']: (T, C_dom=1, C_{frag_from}=1, C_{frag_to}=1, S_from, S_to)
        #   all_transit_matrices['marginal']: (C_dom=1, C_{frag_from}=1, C_{frag_to}=1, 2, 2)
        #   all_transit_matrices['log_corr']: (C_dom=1, C_frag=1)
        
        # if geometric indel model
        #   all_transit_matrices['joint']: (2,)
        # 
        #   if return_all_matrices is True, also include:
        #   all_transit_matrices['conditional']: (2,)
        #   all_transit_matrices['marginal']: (2,)
        #   all_transit_matrices['log_corr']: float
        out = self.transitions_module( t_array = t_array,
                                       return_all_matrices = return_all_matrices,
                                       sow_flax_intermeds = sow_flax_intermeds ) 
        all_transit_matrices = out[1]
        del out
        
        
        ###########################################################
        ### build log-transformed equilibrium distribution; get   #
        ### site-level mixture probability P(c_site | c_tr)       #
        ###########################################################
        # log_site_class_probs is (C_tr=1, C_sites)
        # log_equl_dist_per_mixture is (C_tr=1, C_sites, A)
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
        # both are (C_tr=1, C_sites, K)
        log_rate_mult_probs, rate_multipliers = self.rate_mult_module( sow_flax_intermeds = sow_flax_intermeds,
                                                                       log_site_class_probs = log_site_class_probs,
                                                                       log_transit_class_probs = jnp.zeros(1,) ) 
        
        
        ####################################################
        ### build substitution log-probability matrix      #
        ### use this to score emissions from match sites   #
        ####################################################
        # cond_logprobs_per_mixture is (T, C_tr=1, C_sites, C_subrate A, A) 
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
        # joint_subst_logprobs_per_mixture is (T, C_tr=1, C_sites, C_subrate A, A)
        joint_subst_logprobs_per_mixture = joint_logprob_emit_at_match_per_mixture( cond_logprob_emit_at_match_per_mixture = cond_subst_logprobs_per_mixture,
                                                                              log_equl_dist_per_mixture = log_equl_dist_per_mixture ) 
        
        # marginalize over classes and possible rate multipliers
        joint_logprob_emit_at_match = lse_over_match_logprobs_per_mixture(log_site_class_probs = log_site_class_probs,
                                                            log_rate_mult_probs = log_rate_mult_probs,
                                                            logprob_emit_at_match_per_mixture = joint_subst_logprobs_per_mixture) # (T, C_tr=1, A, A)
        
        if return_all_matrices:
            cond_logprob_emit_at_match = lse_over_match_logprobs_per_mixture(log_site_class_probs = log_site_class_probs,
                                                            log_rate_mult_probs = log_rate_mult_probs,
                                                            logprob_emit_at_match_per_mixture = cond_subst_logprobs_per_mixture) # (T, C_tr=1, A, A)

        
        ################################
        ### suppress all unused dims   #
        ################################
        # indel model outputs, if TKF91
        if self.indel_model_type == 'tkf91':
            all_transit_matrices['joint'] = all_transit_matrices['joint'][:,0,...] # (T, S, S)
    
            if return_all_matrices:
                all_transit_matrices['conditional'] = all_transit_matrices['conditional'][:,0,...] # (T, S, S)
                all_transit_matrices['marginal'] = all_transit_matrices['marginal'][0,...] # (S, S)
                
        # indel model outputs, if TKF92
        elif self.indel_model_type == 'tkf92':
            all_transit_matrices['joint'] = all_transit_matrices['joint'][:,0,0,0,...] # (T, S, S)
    
            if return_all_matrices:
                all_transit_matrices['conditional'] = all_transit_matrices['conditional'][:,0,0,0,...] # (T, S, S)
                all_transit_matrices['marginal'] = all_transit_matrices['marginal'][0,0,0,...] # (S, S)
                all_transit_matrices['log_corr'] = all_transit_matrices['log_corr'][0,0] # float
        
        # equilibrium distribution module outputs
        log_site_class_probs = log_site_class_probs[0,...] #(C_sites)
        log_equl_dist_per_mixture = log_equl_dist_per_mixture[0,...] #(C_sites, A)
        logprob_emit_at_indel = logprob_emit_at_indel[0,...] #(A)
        
        # rate multipliers module outputs
        log_rate_mult_probs = log_rate_mult_probs[0,...] #(C_sites, C_subrate)
        rate_multipliers = rate_multipliers[0,...] #(C_sites, C_subrate)
        
        # substitutuion module outputs
        cond_subst_logprobs_per_mixture = cond_subst_logprobs_per_mixture[:, 0, ...] #(T, C_sites, C_subrate A, A)
        joint_subst_logprobs_per_mixture = joint_subst_logprobs_per_mixture[:, 0, ...] #(T, C_sites, C_subrate A, A)
        joint_logprob_emit_at_match = joint_logprob_emit_at_match[:, 0, ...] #(T, A, A)
        
        if return_all_matrices:
            cond_logprob_emit_at_match = cond_logprob_emit_at_match[:,0,...] #(T, A, A)
        
        
        #####################
        ### decide output   #
        #####################
        # always returned (at training, at final eval, etc.)
        out_dict = {'logprob_emit_at_indel': logprob_emit_at_indel, #(A,)
                    'joint_logprob_emit_at_match': joint_logprob_emit_at_match, #(T,A,A)
                    'all_transit_matrices': all_transit_matrices} #dict
        
        # returned if you need conditional and marginal logprob matrices
        if return_all_matrices:
            out_dict['cond_logprob_emit_at_match'] = cond_logprob_emit_at_match #(T, A, A)
        
        # all intermediates
        if return_intermeds:
            to_add = {'log_equl_dist_per_mixture': log_equl_dist_per_mixture, #(C_sites, A)
                      'rate_multipliers': rate_multipliers, #(C_sites, C_subrate)
                      'rate_matrix': subst_module_intermeds.get('rate_matrix',None), #(C_sites,A,A) or None
                      'exchangeabilities': subst_module_intermeds.get('exchangeabilities',None), #(A,A) or None
                      'cond_subst_logprobs_per_mixture': cond_subst_logprobs_per_mixture, # (T, C_sites, C_subrate A, A)
                      'joint_subst_logprobs_per_mixture': joint_subst_logprobs_per_mixture, # (T, C_sites, C_subrate A, A) 
                      'log_site_class_probs': log_site_class_probs, #(C_sites,)
                      'log_rate_mult_probs': log_rate_mult_probs } #(C_sites, C_subrate)
            out_dict = {**out_dict, **to_add}
        
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
        
        # final conditional and joint prob of match (before and after LSE over classes)
        for loss_type in ['joint', 'cond']:
            for suffix in ['logprob_emit_at_match', 'subst_logprobs_per_mixture']:
                mat = np.exp (out[f'{loss_type}_{suffix}'] ) 
                new_key = f'{prefix}_{loss_type}_{suffix}'.replace('log','')
                write_matrix_to_npy( out_folder, mat, new_key )
                maybe_write_matrix_to_ascii( out_folder, mat, new_key )
                del mat, new_key
                
        # transition matrix: joint transition matrix
        mat = np.exp(out['all_transit_matrices']['joint']) #(T, A, A)
        key = f'{prefix}_joint_prob_transit_matrix'
        write_matrix_to_npy( out_folder, mat, key )
        maybe_write_matrix_to_ascii( out_folder, mat, key )
        del mat, key
        
        # transition matrix: conditional and marginal transition matrices
        if self.indel_model_type is not None:
            for loss_type in ['conditional','marginal']:
                mat = np.exp(out['all_transit_matrices'][loss_type]) 
                key = f'{prefix}_{loss_type}_prob_transit_matrix'
                write_matrix_to_npy( out_folder, mat, key )
                maybe_write_matrix_to_ascii( out_folder, mat, key )
                del mat, key
                
        
        #####################################################################
        ### only write once: parameters, things that don't depend on time   #
        #####################################################################
        if write_time_static_objs:
            ### class probs
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
            rate_matrix = out['rate_matrix'] #(C_sites, A, A) or None
            if rate_matrix is not None:
                key = f'{prefix}_rate_matrix'
                write_matrix_to_npy( out_folder, rate_matrix, key )
                del key
                
                for c in range(rate_matrix.shape[0]):
                    mat_to_save = rate_matrix[c,...]
                    key = f'{prefix}_site-class-{c}_rate_matrix'
                    maybe_write_matrix_to_ascii( out_folder, mat_to_save, key )
                    del mat_to_save, key
                    
                            
            ### equilibrium distribution
            # BEFORE marginalizing over site clases
            equl_dist = np.exp(out['log_equl_dist_per_mixture']) #(C_sites, A)
            key = f'{prefix}_equilibriums-per-site-class'
            write_matrix_to_npy( out_folder, equl_dist, key )
            maybe_write_matrix_to_ascii( out_folder, equl_dist, key )
            del key
            
            # AFTER marginalizing over classes
            mat = np.exp( out['logprob_emit_at_indel'] ) #(A,)
            new_key = f'{prefix}_logprob_emit_at_indel'.replace('log','')
            write_matrix_to_npy( out_folder, mat, new_key )
            maybe_write_matrix_to_ascii( out_folder, mat, new_key )
            del mat, new_key
            
            
            ### rate multipliers: \rho_{c,k} or \rho_k
            if not self.rate_mult_module.use_unit_rate_mult:
                rate_multipliers = out['rate_multipliers'] #(C_sites, C_subrate)
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
                
            
            ####################################################
            ### extract transition paramaters, intermediates   # 
            ### needed for final scoring matrices              #
            ### (also does not depend on time)                 #
            ####################################################
            ### under geometric length (only scoring subs)
            if self.indel_model_type is None:
                if self.config['load_all']:
                    geom_p_emit = np.exp(self.transitions_module.out_vec)[0].item() #float
                    
                elif not self.config['load_all']:
                    geom_p_emit = nn.sigmoid(self.transitions_module.p_emit_logit).item() #float
                
                arr = np.array( [geom_p_emit, 1 - geom_p_emit] )
                key = f'{prefix}_geom_seq_len'
                write_matrix_to_npy( out_folder, arr, key )
                del key, arr
                
                with open(f'{out_folder}/ASCII_{prefix}_geom_seq_len.txt','w') as g:
                    g.write(f'P(emit): {geom_p_emit}\n')
                    g.write(f'1-P(emit): {1 - geom_p_emit}\n')
                    
                    
            ### for TKF models (not saved to intermediates)
            elif self.indel_model_type in ['tkf91', 'tkf92']:
                # always write lambda and mu
                if self.config['load_all']:
                    lam = self.transitions_module.param_dict['lambda']
                    mu = self.transitions_module.param_dict['mu']
                    offset = 1 - (lam/mu)
                    
                elif not self.config['load_all']:
                    mu_min_val = self.transitions_module.mu_min_val #float
                    mu_max_val = self.transitions_module.mu_max_val #float
                    offs_min_val = self.transitions_module.offs_min_val #float
                    offs_max_val = self.transitions_module.offs_max_val #float
                    mu_offset_logits = self.transitions_module.tkf_mu_offset_logits #(2,)
                
                    mu = bound_sigmoid(x = mu_offset_logits[0,0],
                                       min_val = mu_min_val,
                                       max_val = mu_max_val).item() #float
                    
                    offset = bound_sigmoid(x = mu_offset_logits[0,1],
                                             min_val = offs_min_val,
                                             max_val = offs_max_val).item() #float
                    lam = mu * (1 - offset)  #float
                    
                with open(f'{out_folder}/ASCII_{prefix}_{self.indel_model_type}_indel_params.txt','w') as g:
                    g.write(f'insert rate, lambda: {lam}\n')
                    g.write(f'deletion rate, mu: {mu}\n')
                    g.write(f'offset: {offset}\n\n')
                
                out_dict = {'lambda': np.array(lam), # shape=()
                            'mu': np.array(mu), # shape=()
                            'offset': np.array(offset)} # shape=()
                                
                # if tkf92, have extra r_ext param
                if self.indel_model_type == 'tkf92':
                    if self.config['load_all']:
                        r_extend = self.transitions_module.param_dict['r_extend']
                        
                    elif not self.config['load_all']:
                        r_extend_min_val = self.transitions_module.r_extend_min_val
                        r_extend_max_val = self.transitions_module.r_extend_max_val
                        r_extend_logits = self.transitions_module.r_extend_logits #(C_dom=1, C_frag=1)
                        
                        r_extend = bound_sigmoid(x = r_extend_logits,
                                                 min_val = r_extend_min_val,
                                                 max_val = r_extend_max_val) #(C_dom=1, C_frag=1)
                    
                    mean_indel_lengths = 1 / (1 - r_extend) #(C_dom=1, C_frag=1)
                    
                    with open(f'{out_folder}/ASCII_{prefix}_{self.indel_model_type}_indel_params.txt','a') as g:
                        g.write(f'extension prob, r: ')
                        [g.write(f'{elem}\t') for elem in r_extend.flatten()]
                        g.write('\n')
                        g.write(f'mean indel length: ')
                        [g.write(f'{elem}\t') for elem in mean_indel_lengths]
                        g.write('\n')
                    
                    out_dict['r_extend'] = r_extend #(C_dom=1, C_frag=1)
                
                with open(f'{out_folder}/PARAMS-DICT_{prefix}_{self.indel_model_type}_indel_params.pkl','wb') as g:
                    pickle.dump(out_dict, g)
                del out_dict


class IndpSitesLoadAll(IndpSites):
    """
    like IndpSites, but load all parameters to use (excluding time, 
        exponential distribution parameter)
    
    
    Initialize with
    ----------------
    config : dict
        config['num_site_mixtures'] :  int
            number of emission site classes
            
        config['subst_model_type'] : {gtr, hky85}
            which substitution model
        
        config['indel_model_type'] : {tkf91, tkf92, None}
            which indel model, if any
            
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
        
    Inherited from IndpSites
    -------------------------
    __call__
        unpack batch and calculate logP(anc, desc, align)
    
    write_params
        write parameters to files

    calculate_all_loglikes
        calculate logP(anc, desc, align), logP(anc), logP(desc), and
        logP(desc, align | anc)
    
    Methods inherited from neural_models.model_utils.BaseClasses.ModuleBase
    -----------------------------------------------------------------
    sow_flax_intermeds
        for tensorboard logging
    """
    config: dict
    name: str
    
    def setup(self):
        # not set up for domain or fragment level mixtures
        assert self.config['num_fragment_mixtures'] == 1
        assert self.config['num_domain_mixtures'] == 1
        
        ###################
        ### read config   #
        ###################
        # required
        self.num_site_mixtures = self.config['num_site_mixtures']
        self.num_rate_mults = self.config['k_rate_mults']
        self.num_domain_mixtures = 1
        self.num_fragment_mixtures = 1
        self.num_transit_mixtures = 1 # C_tr
        
        self.subst_model_type = self.config['subst_model_type'].lower()
        indel_model_type = self.config['indel_model_type']
        self.indel_model_type = indel_model_type.lower() if indel_model_type is not None else None
        self.times_from = self.config['times_from'].lower()
        
        # hard coded or optional
        self.indp_rate_mults = self.config.get('indp_rate_mults', False)
        self.norm_reported_loss_by = 'desc_len'
        self.exponential_dist_param = 1
        
        
        ###########################################
        ### module for transition probabilities   #
        ###########################################        
        if self.indel_model_type is None:
            self.transitions_module = GeomLenTransitionLogprobsFromFile(config = self.config,
                                                     name = f'geom seq lengths model')
            
        elif self.indel_model_type == 'tkf91':
            self.transitions_module = TKF91TransitionLogprobsFromFile(config = self.config,
                                                     name = f'tkf91 indel model')
        
        elif self.indel_model_type == 'tkf92':
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
        ### mixture weights P(c_sites | c_frag*c_dom)                 #
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




