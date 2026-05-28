#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 13:43:50 2025


 'LocalTKF91' (base class, not used directly),

 'TKF92LocalRateLocalFragSize',
- (OPTIONAL) pred_config['mu_range']
- (OPTIONAL) pred_config['offset_range']
- (OPTIONAL) pred_config['r_range']
- (OPTIONAL) pred_config['tkf_function']

"""
from flax import linen as nn
import jax
import jax.numpy as jnp

from utils.BaseClasses import (neuralTKFModuleBase, 
                                ModuleBase)
from neural_models.neural_hmm_predict.model_functions import (bound_sigmoid,
                                                       safe_log,
                                                       logsumexp_with_arr_lst,
                                                       log_one_minus_x,
                                                       switch_tkf,
                                                       regular_tkf,
                                                       approx_tkf,
                                                       logprob_tkf91,
                                                       logprob_tkf92)


# add another tensorboard recording utililty function
class transitionModuleBase(neuralTKFModuleBase):
    def maybe_sow_indel_interms(self,
                                mu,
                                offset,
                                r_extend,
                                tkf_params_dict,
                                sow_flax_intermeds):
        # lambda, mu, extension prob (r)
        lam = mu * (1-offset)
        lam = jnp.squeeze(lam)
        mu = jnp.squeeze(mu)
        r_extend = jnp.squeeze(r_extend)
        
        self.maybe_sow(sow_flax_intermeds = sow_flax_intermeds,
               vals = lam,
               label = f'{self.name}/neuralTKF_lambda',
               include_min_max = True,
               include_perc_zeros = False)
        
        self.maybe_sow(sow_flax_intermeds = sow_flax_intermeds,
               vals = mu,
               label = f'{self.name}/neuralTKF_mu',
               include_min_max = True,
               include_perc_zeros = False)
        
        if r_extend is not None:
            self.maybe_sow(sow_flax_intermeds = sow_flax_intermeds,
                   vals = r_extend,
                   label = f'{self.name}/neuralTKF92_r_extend',
                   include_min_max = True,
                   include_perc_zeros = True)
        
        # alpha, beta, gamma
        self.maybe_sow(sow_flax_intermeds = sow_flax_intermeds,
               vals = jnp.exp(tkf_params_dict['log_alpha']),
               label = f'{self.name}/neuralTKF_alpha',
               include_min_max = True,
               include_perc_zeros = False)
        
        self.maybe_sow(sow_flax_intermeds = sow_flax_intermeds,
               vals = jnp.exp(tkf_params_dict['log_beta']),
               label = f'{self.name}/neuralTKF_beta',
               include_min_max = True,
               include_perc_zeros = False)
        
        self.maybe_sow(sow_flax_intermeds = sow_flax_intermeds,
               vals = jnp.exp(tkf_params_dict['log_gamma']),
               label = f'{self.name}/neuralTKF_gamma',
               include_min_max = True,
               include_perc_zeros = False)


###############################################################################
### LOCAL indel rates   #######################################################
###############################################################################
class LocalTKF91(transitionModuleBase):
    """
    (idk if I'll use this, but it's useful to have this before defining
       the next TKF92 function: local rates and local fragment sites)
    
    CONDITIONAL LOG-PROBABILITY!!! calculating logP(align_i|align_{i-1},Anc,Desc_{...i-1},t)
    
    B = batch size; number of samples
    T = number of branch lengths; this could be: 
        > an array of times for all samples (T; marginalize over these later)
        > an array of time per sample (T=B)
        
    Initialize with
    ----------------
    config : dict 
        config["mu_range"] : Tuple, (2,)
            range for bound sigmoid activation that determines mu
            DEFAULT: -1e-4, 2
        
        config["offset_range"] : Tuple, (2,)
            range for bound sigmoid activation that determines offset 
            (which determines lambda)
            DEFAULT: -1e-4, 0.333
            
    name : str
        class name, for flax
    
    
    Methods here
    ------------
    setup
    __call__
    get_r_extend
        placeholder; returns None
    """
    config: dict
    name: str
    
    def setup(self):
        """
        Flax Module Parameters
        -----------------------
        tkf_mu_offset: 
        """
        name = f'{self.name}/Project to mu, offset'
        self.use_bias = self.config.get('use_bias', True)
        
        self.mu_offset_final_project = nn.Dense(features = 2,
                                                use_bias = self.use_bias,
                                                name = name)
        self.mu_min_val, self.mu_max_val = self.config.get( 'mu_range', 
                                                            [1e-4, 2] )
        self.offs_min_val, self.offs_max_val = self.config.get( 'offset_range', 
                                                                [1e-4, 0.333] )
        
        # decide tkf function
        tkf_function_name = self.config.get('tkf_function', 'switch_tkf')
        tkf_fn_registry = {'regular_tkf': regular_tkf,
                           'approx_tkf': approx_tkf,
                           'switch_tkf': switch_tkf}
        self.tkf_function = tkf_fn_registry[tkf_function_name]
        
        # declare the logprob function
        self.cond_logprob_fn = logprob_tkf91
        
        
    def __call__(self,
                 datamat,
                 t_array,
                 unique_time_per_sample: bool,
                 sow_flax_intermeds: bool):
        """
        T: number of times
        B: batch size
        H: input hidden dim
        L_align: length of alignment
        
        
        Arguments
        ----------
        datamat : ArrayLike, (B, L_align, H)
        
        t_array : ArrayLike, (T,)
            branch lengths, times for marginalizing over
        
        unique_time_per_sample : Bool
            whether there's one time per sample, or a grid of times you'll 
            marginalize over
            
        sow_flax_intermeds : bool
            switch for tensorboard logging
          
        Returns
        -------
        cond_logprob : ArrayLike
            > if unique time per sample: (B, L_align, 4, 4) 
            > if not unique time per sample: (T, B, L_align, 4, 4) 
            log-probability matrix for transitions
        
        use_approx : Tuple( ArrayLike, ArrayLike ), ( (T,), (T,) ) or ( (B,), (B,) )
            where tkf approximation formulas were used
            
        """
        ### logits to values
        # get mu and offset; (B, L_align, H) -> (B, L_align, 2)
        tkf_mu_offset_logits = self.mu_offset_final_project(datamat)
        
        mu = self.apply_bound_sigmoid_activation( logits = tkf_mu_offset_logits[...,0],
                                                  min_val = self.mu_min_val,
                                                  max_val = self.mu_max_val,
                                                  param_name = 'tkf mu',
                                                  sow_flax_intermeds = False ) #(B,L_align)
        
        offset = self.apply_bound_sigmoid_activation( logits = tkf_mu_offset_logits[...,1],
                                                      min_val = self.offs_min_val,
                                                      max_val = self.offs_max_val,
                                                      param_name = 'tkf offset',
                                                      sow_flax_intermeds = False ) #(B,L_align)
        
        r_extend = self.get_r_extend(datamat)
        
        
        ### get tkf alpha, beta, gamma
        # contents of out_dict ( all ArrayLike[float32], (T,B,L_align) or (B,L_align) ):
        #   out_dict['log_alpha']
        #   out_dict['log_one_minus_alpha']
        #   out_dict['log_beta']
        #   out_dict['log_one_minus_beta']
        #   out_dict['log_gamma']
        #   out_dict['log_one_minus_gamma']
        tkf_params_dict = self.tkf_function(mu = mu, 
                                            offset = offset,
                                            t_array = t_array,
                                            unique_time_per_sample = unique_time_per_sample)
        
        # record values to tensorboard
        self.maybe_sow_indel_interms(mu = mu,
                                    offset = offset,
                                    r_extend = r_extend,
                                    tkf_params_dict = tkf_params_dict,
                                    sow_flax_intermeds = sow_flax_intermeds)
        
        # cond_logprob is either:
        # (T, B, L_align, 4, 4), or
        # (B, L_align, 4, 4)
        cond_logprob =  self.cond_logprob_fn( tkf_params_dict = tkf_params_dict,
                                              r_extend = r_extend,
                                              offset = offset,
                                              unique_time_per_sample = unique_time_per_sample ) 
        
        intermed_params_dict = {'lambda': mu * (1-offset),
                                'mu': mu,
                                'r_extend': r_extend}
        
        return cond_logprob, intermed_params_dict
        
    
    def get_r_extend(self, 
                       *args, 
                       **kwargs):
        """
        placeholder; r would actually be zero, but this plays poorly with 
          log-transformed functions
        """
        return None


class TKF92LocalRateLocalFragSize(LocalTKF91):
    """
    CONDITIONAL LOG-PROBABILITY!!! calculating logP(align_i|align_{i-1},Anc,Desc_{...i-1},t)
    
    B = batch size; number of samples
    T = number of branch lengths; this could be: 
        > an array of times for all samples (T; marginalize over these later)
        > an array of time per sample (T=B)
        
    Initialize with
    ----------------
    config : dict 
        config["init_mu_offset_logits"] : Tuple, (1, 1, 2)
            initial values for logits that determine mu, offset
            DEFAULT: -2, -5
        
        config["mu_range"] : Tuple, (2,)
            range for bound sigmoid activation that determines mu
            DEFAULT: -1e-4, 2
        
        config["offset_range"] : Tuple, (2,)
            range for bound sigmoid activation that determines offset 
            (which determines lambda)
            DEFAULT: -1e-4, 0.333
            
        config["r_range"]
            range for bound sigmoid activation that determines TKF r
            DEFAULT: -1e-4, 0.999
            
    name : str
        class name, for flax
    
    
    Methods here
    -------------
    setup
    __call__
    
    Inherited from LocalTKF91
    ----------------------------
    setup (run before this setup)
    get_r_extend (not used)
    """
    config: dict
    name: str
    
    def setup(self):
        super().setup()
        
        ### extra stuff for TKF92: R extension probability 
        # overwrite cond_logprob_fn
        self.cond_logprob_fn = logprob_tkf92
        
        # setting limits for bound sigmoid function
        self.r_extend_min_val, self.r_extend_max_val = self.config.get( 'r_range', 
                                                                [1e-4, 0.999] )
        
        # linear projection to R
        name = f'{self.name}/Project to R extension prob'
        self.final_project_to_r = nn.Dense(features = 1,
                                           use_bias = self.use_bias,
                                           name = name)
        
    def get_r_extend(self,
                       datamat):
        # r: (B, L_align, H) -> (B, L_align, 1) -> (B, L_align)
        r_logits = self.final_project_to_r(datamat)[...,0]
        r_extend = self.apply_bound_sigmoid_activation( logits = r_logits,
                                                        min_val = self.r_extend_min_val,
                                                        max_val = self.r_extend_max_val,
                                                        param_name = 'tkf92 r',
                                                        sow_flax_intermeds = False )
        return r_extend
        