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
from older_indel_models.lg08_gtr_emission_model import (LG08Logprobs,
                                                        equl_dist_logprobs_from_counts)

from older_indel_models.transition_models import ( OtherTransitionLogprobs,
                                                   TKF91TransitionLogprobsOldStyle,
                                                   TKF92TransitionLogprobsOldStyle,
                                                   OtherTransitionLogprobsFromFile,
                                                   TKF91TransitionLogprobsOldStyleFromFile,
                                                   TKF92TransitionLogprobsOldStyleFromFile )
from latent_class_mixtures.model_functions import ( bound_sigmoid,
                                                    safe_log,
                                                    cond_prob_from_counts,
                                                    write_matrix_to_npy,
                                                    maybe_write_matrix_to_ascii)



class IndpSitesOldModels_GTR(ModuleBase):
    """
    pairHMM that finds conditional loglikelihood of alignments, P(Desc, Align | Anc)
    
    
    differences from current version: latent_class_mixtures.IndpSites:
    ------------------------------------------------------------------
    - TKF91 and TKF92 calculates same conditional transition matrix as current 
      version of the code, but it does so directly (without having to divide 
      joint by marginal)
    - Other indel models treat start and end states like match
    
    B = batch size; number of samples
    T = number of branch lengths; this could be: 
        > an array of times for all samples (T; marginalize over these later)
        > an array of time per sample (T=B)
        > a quantized array of times per sample (T = T', where T' <= T)
    S: number of transition states (4 here: M, I, D, start/end)
    
    
    Initialize with
    ----------------
    config : dict
        config['indel_model_type'] : {tkf91, tkf92, None}
            which indel model, if any
            
        config['times_from'] : {geometric, t_array_from_file, t_per_sample}

        config['t_grid_step'] : int, optional
            There is an exponential prior over time; this provides the
            parameter for this during marginalization over times
        
    name : str
        class name, for flax
    
    
    Main methods here
    -----------------
    setup
    
    __call__
        unpack batch and calculate logP(desc, align | anc)
    
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
        # not applicable here
        self.num_fragment_mixtures = 1
        self.num_domain_mixtures = 1
        self.num_transit_mixtures = 1
        self.num_site_mixtures = 1
        self.indp_rate_mults = None
        self.subst_model_type = 'lg08_gtr'
        self.norm_reported_loss_by = 'desc_len'
        
    
        ### read config
        indel_model_type = self.config['indel_model_type']
        self.indel_model_type = indel_model_type.lower() if indel_model_type is not None else None
        self.exponential_dist_param = self.config.get('exponential_dist_param', 1)
        self.times_from = self.config['times_from'].lower()
        
        
        ### init emission models
        # equilibrium distribution
        training_dset_emit_counts = self.config['training_dset_emit_counts']
        self.log_equl_dist = equl_dist_logprobs_from_counts( training_dset_emit_counts = training_dset_emit_counts ) #(A,)
        
        # init gtr substitution model with LG08 exchangeabilities
        norm_rate_matrix = self.config.get('norm_rate_matrix', True)
        self.lg08_gtr = LG08Logprobs( log_equl_dist = self.log_equl_dist,
                                      norm = norm_rate_matrix )
        
        
        ### init indel model
        if self.indel_model_type == 'tkf91':
            self.transitions_module = TKF91TransitionLogprobsOldStyle(config = self.config,
                                                     name = f'tkf91 indel model')
        
        elif self.indel_model_type == 'tkf92':
            self.transitions_module = TKF92TransitionLogprobsOldStyle(config = self.config,
                                                     name = f'tkf92 indel model')
        
        else:
            self.transitions_module = OtherTransitionLogprobs(config = self.config,
                                                     name = f'{indel_model_type} indel model')
            
            
    def __call__(self,
                 batch: list[ArrayLike],
                 t_array: ArrayLike,
                 sow_flax_intermeds: bool):
        """
        Use this during active model training
        
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
        # scoring_matrices_dict has the following keys:
        #   logprob_emit_at_indel: ArrayLike, (A, )
        #   cond_logprob_emit_at_match: ArrayLike, (T, A, A)
        #   cond_transit_matrix: ArrayLike, (T, S, S)
        scoring_matrices_dict = self._get_scoring_matrices(t_array=times_for_matrices,
                                        sow_flax_intermeds=sow_flax_intermeds )
        
        # calculate loglikelihoods
        # 
        # out_dict has the following keys:
        # cond_neg_logP: ArrayLike, (B,)
        # cond_neg_logP_length_normed: ArrayLike, (B,)
        out_dict = cond_prob_from_counts( batch = batch,
                                          times_from = self.times_from,
                                          score_indels = True,
                                          scoring_matrices_dict = scoring_matrices_dict,
                                          t_array = t_array,
                                          exponential_dist_param = self.exponential_dist_param,
                                          norm_reported_loss_by = self.norm_reported_loss_by,
                                          return_intermeds = False )
        
        # add scoring matrices to out_dict
        out_dict = {**out_dict, **scoring_matrices_dict} 
        
        # calculate loss
        loss = jnp.mean( out_dict['cond_neg_logP'] ) #float
        
        return loss, out_dict
    
    
    def _get_scoring_matrices(self,
                             t_array,
                             sow_flax_intermeds: bool,
                             *args,
                             **kwargs): 
        """
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
        
        sow_flax_intermeds : bool
            switch for tensorboard logging
          
        Returns
        -------
        out_dict : dict
            out_dict['logprob_emit_at_indel'] : (A,)
            out_dict['cond_logprob_emit_at_match'] : (T, A, A) or (B, A, A)
            out_dict['all_transit_matrices'] : dictionary of two arrays
                > 'conditional' :  (T, S, S) or (B, S, S)
                > 'log_corr' :  (B,)
        """
        # log equilibrium distribution
        logprob_emit_at_indel = self.log_equl_dist #(A,)
        
        # substitution log probability
        cond_logprob_emit_at_match = self.lg08_gtr(t_array) #(T, A, A) or (B, A, A)
        
        # indel log probability
        out = self.transitions_module( t_array = t_array,
                                       sow_flax_intermeds = sow_flax_intermeds ) 
        cond_transit_matrix, log_corr, maybe_tkf_params = out
        del out
        
        # output still needs to be a dictionary...
        all_transit_matrices = {}
        all_transit_matrices['conditional'] = cond_transit_matrix
        all_transit_matrices['log_corr'] = log_corr
        del cond_transit_matrix, log_corr
        
        
        ### output
        out_dict = {'logprob_emit_at_indel': logprob_emit_at_indel, #(A,)
                    'cond_logprob_emit_at_match': cond_logprob_emit_at_match, #(T,A,A) or (B, A, A)
                    'all_transit_matrices': all_transit_matrices, #dictionary
                    'maybe_tkf_params': maybe_tkf_params} # correction factor for tkf92; otherwise a placeholder
        return out_dict
    
    
    def write_params(self,
                     t_array,
                     out_folder: str,
                     prefix: str,
                     write_time_static_objs: bool):
        ###################################
        ### always write: Full matrices   #
        ###################################
        out = self._get_scoring_matrices(t_array=t_array,
                                         sow_flax_intermeds=False)
        
        # final conditional prob of match 
        mat = np.exp( out[f'cond_logprob_emit_at_match'] ) #(T, A, A) or (B, A, A)
        new_key = f'{prefix}_cond_logprob_emit_at_match'.replace('log','')
        write_matrix_to_npy( out_folder, mat, new_key )
        maybe_write_matrix_to_ascii( out_folder, mat, new_key )
        del mat, new_key
                
        # transition matrix: conditional
        mat = np.exp(out['all_transit_matrices']['conditional'])  #(T, S, S) or (B, S, S)
        key = f'{prefix}_cond_prob_transit_matrix'
        write_matrix_to_npy( out_folder, mat, key )
        maybe_write_matrix_to_ascii( out_folder, mat, key )
        del mat, key
                
        
        #####################################################################
        ### only write once: parameters, things that don't depend on time   #
        #####################################################################
        if write_time_static_objs:
            ### equilibrium distribution (AFTER marginalizing over classes)
            mat = np.exp( out['logprob_emit_at_indel'] ) #(A,)
            new_key = f'{prefix}_logprob_emit_at_indel'.replace('log','')
            write_matrix_to_npy( out_folder, mat, new_key )
            maybe_write_matrix_to_ascii( out_folder, mat, new_key )
            del mat, new_key
                
            
            ### write indel params for TKF models
            if self.indel_model_type in ['tkf91', 'tkf92']:
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
                    
                    if self.transitions_module.tie_params:
                        offset = jnp.array( 1e-4 ) #float
                    
                    elif not self.transitions_module.tie_params:
                        offset = offset = bound_sigmoid(x = mu_offset_logits[0,1],
                                                 min_val = offs_min_val,
                                                 max_val = offs_max_val).item() #float
                        
                    lam = mu * (1 - offset)  #(1,)
                    
                with open(f'{out_folder}/ASCII_{prefix}_{self.indel_model_type}_indel_params.txt','w') as g:
                    g.write(f'insert rate, lambda: {lam}\n')
                    g.write(f'deletion rate, mu: {mu}\n')
                    g.write(f'offset: {offset}\n\n')
                
                out_dict = {'lambda': np.array(lam), 
                            'mu': np.array(mu), 
                            'offset': np.array(offset)} 
                                
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
            
            
            ### write indel params for other models
            elif self.indel_model_type in ['h20', 'rs07', 'lg05', 'km03']:
                if self.config['load_all']:
                    lam = self.transitions_module.param_dict['lam']
                    mu = self.transitions_module.param_dict['mu']
                    x = self.transitions_module.param_dict['x']
                    y = self.transitions_module.param_dict['y']
                
                elif not self.config['load_all']:
                    indel_logits = self.transitions_module.indel_logits #(4,)
                    lam_logits = indel_logits[0]
                    mu_logits = indel_logits[1]
                    x_logits = indel_logits[2]
                    y_logits = indel_logits[3]
                    
                    lambda_min_val = self.transitions_module.lambda_min_val #float
                    lambda_max_val = self.transitions_module.lambda_max_val #float
                    
                    mu_min_val = self.transitions_module.mu_min_val #float
                    mu_max_val = self.transitions_module.mu_max_val #float
                    
                    x_min_val = self.transitions_module.x_min_val #float
                    x_max_val = self.transitions_module.x_max_val #float
                    
                    y_min_val = self.transitions_module.y_min_val #float
                    y_max_val = self.transitions_module.y_max_val #float
                    
                    mu = bound_sigmoid(x = mu_logits,
                                       min_val = mu_min_val,
                                       max_val = mu_max_val).item() #float
                    
                    x = bound_sigmoid(x = x_logits,
                                       min_val = x_min_val,
                                       max_val = x_max_val).item() #float
                    
                    if self.transitions_module.tie_params:
                        lam = mu #float
                        y = x #float
                    
                    elif not self.transitions_module.tie_params: 
                        lam = bound_sigmoid(x = lam_logits,
                                           min_val = lambda_min_val,
                                           max_val = lambda_max_val).item() #float
                        
                        y = bound_sigmoid(x = y_logits,
                                           min_val = y_min_val,
                                           max_val = y_max_val).item() #float
                    
                with open(f'{out_folder}/ASCII_{prefix}_{self.indel_model_type}_indel_params.txt','a') as g:
                    g.write(f'insert rate, lambda: {lam}\n')
                    g.write(f'deletion rate, mu: {mu}\n')
                    g.write(f'probability of extending insertion, x: {x}\n')
                    g.write(f'probability of extending deletion, y: {y}\n')
                    
                    
class IndpSitesOldModelsLoadAll(IndpSitesOldModels):
    """
    like IndpSitesOldModels, but load all parameters to use (excluding time, 
        exponential distribution parameter)
    
    B = batch size; number of samples
    T = number of branch lengths; this could be: 
        > an array of times for all samples (T; marginalize over these later)
        > an array of time per sample (T=B)
        > a quantized array of times per sample (T = T', where T' <= T)
    S: number of transition states (4 here: M, I, D, start/end)
    
    
    Initialize with
    ----------------
    config : dict
        config['filenames'] : files of parameters to load
        
        config['indel_model_type'] : {tkf91, tkf92, None}
            which indel model, if any
            
        config['times_from'] : {geometric, t_array_from_file, t_per_sample}

        config['t_grid_step'] : int, optional
            There is an exponential prior over time; this provides the
            parameter for this during marginalization over times
        
    name : str
        class name, for flax
    
    
    Main methods here
    -----------------
    setup
    
    
    Methods inherited from IndpSitesOldModels
    ------------------------------------------
    __call__
        unpack batch and calculate logP(desc, align | anc)
    
    write_params
        write parameters to files
    
    
    Methods inherited from ModuleBase
    ---------------------------------
    sow_flax_intermeds
        for tensorboard logging
    """
    config: dict
    name: str
    
    def setup(self):
        # not applicable here
        self.num_fragment_mixtures = 1
        self.num_domain_mixtures = 1
        self.num_transit_mixtures = 1
        self.num_site_mixtures = 1
        self.indp_rate_mults = None
        self.subst_model_type = 'lg08_gtr'
        self.norm_reported_loss_by = 'desc_len'
        
    
        ### read config
        indel_model_type = self.config['indel_model_type']
        self.indel_model_type = indel_model_type.lower() if indel_model_type is not None else None
        self.t_grid_step = self.config.get('t_grid_step', jnp.nan)
        self.times_from = self.config['times_from'].lower()
        
        
        ### init emission models
        # equilibrium distribution
        training_dset_emit_counts = self.config['training_dset_emit_counts']
        self.log_equl_dist = equl_dist_logprobs_from_counts( training_dset_emit_counts = training_dset_emit_counts ) #(A,)
        
        # init gtr substitution model with LG08 exchangeabilities
        norm_rate_matrix = self.config.get('norm_rate_matrix', True)
        self.lg08_gtr = LG08Logprobs( log_equl_dist = self.log_equl_dist,
                                      norm = norm_rate_matrix )
        
        
        ####################
        ### CHANGE HERE    #
        ####################
        ### init indel model
        if self.indel_model_type == 'tkf91':
            self.transitions_module = TKF91TransitionLogprobsOldStyleFromFile(config = self.config,
                                                     name = f'tkf91 indel model')
        
        elif self.indel_model_type == 'tkf92':
            self.transitions_module = TKF92TransitionLogprobsOldStyleFromFile(config = self.config,
                                                     name = f'tkf92 indel model')
        
        else:
            self.transitions_module = OtherTransitionLogprobsFromFile(config = self.config,
                                                     name = f'{indel_model_type} indel model')
            
        
        
                   