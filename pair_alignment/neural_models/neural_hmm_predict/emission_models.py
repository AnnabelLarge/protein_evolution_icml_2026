#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 13:46:56 2025


equilibrium dist, pred_config entries:
----------------------------------------
'GlobalEqul',
- pred_config['training_dset_emit_counts']

'LocalEqul',
- pred_config['emission_alphabet_size']

'EqulFromFile',
- pred_config['filenames']['equl_dist']


F81, pred_config entries:
--------------------------
'LocalF81',
- (OPTIONAL) pred_config['rate_mult_range']

"""
from flax import linen as nn
import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm 

from utils.BaseClasses import (neuralTKFModuleBase, 
                                ModuleBase)
from neural_models.neural_hmm_predict.model_functions import (safe_log,
                                                       bound_sigmoid,
                                                       logprob_f81)



###############################################################################
### Equilibrium distribution models   #########################################
###############################################################################
class GlobalEqul(neuralTKFModuleBase):
    """
    Use the observed amino acid frequencies as the equilibrium distribution
    
    Doesn't have to be a module, but make it one for consistency, I guess
    """
    config: dict
    name: str
    
    def setup(self):
        """
        Flax Module Parameters
        -----------------------
        None
        """
        training_dset_emit_counts = self.config['training_dset_emit_counts'] #(A,)
        prob_equilibr = training_dset_emit_counts/training_dset_emit_counts.sum() #(A,)
        logprob_equilibr = safe_log( prob_equilibr ) #(A,)
        self.logprob_equilibr = logprob_equilibr[None,None,:] #(1,1,A)
        
    def __call__(self,
                 *args,
                 **kwargs): 
        """
        Arguments
        ----------
        None
        
        Returns
        --------
        ArrayLike, (1, 1, A) 
            scoring matrix for emissions from indels, from observed frequencies
        """
        return self.logprob_equilibr  #(1,1,A)

    
class LocalEqul(GlobalEqul):
    """
    Use a set of logits to find equilibrium distribution for each position, 
      each sample
    """
    config: dict
    name: str
    
    def setup(self):
        """
        H = number of features of input matrix
        A = alphabet size
        
        Flax Module Parameters
        -----------------------
        self.final_project : Flax module
          > kernel : ArrayLike, (H, A)
          > bias : ArrayLike, (A)  
        """
        emission_alphabet_size = self.config['emission_alphabet_size']
        self.use_bias = self.config.get('use_bias', True)
        
        name = f'{self.name}/Project to equilibriums'
        self.final_project = nn.Dense(features = emission_alphabet_size,
                                      use_bias = self.use_bias,
                                      name = name)
    
    def __call__(self,
                datamat: jnp.array,
                sow_flax_intermeds: bool):
        """
        apply final linear projection and log_softmax to get final 
          equilibrium distribution
        
        B: batch size
        L_align: length of alignment
        H: input hidden dim
        A: final alphabet size
        
        Arguments
        ----------
        datamat : ArrayLike, (B, L_align, H)
        
        sow_flax_intermeds : bool
            switch for tensorboard logging
          
        Returns
        --------
        ArrayLike, (B, L_align, A) 
            scoring matrix for emissions from indels
        """
        # (B, L_align, H) -> (B, L_align, A)
        logits = self.final_project(datamat)
        
        return self.apply_log_softmax_activation(logits = logits,
                                                 sow_flax_intermeds = sow_flax_intermeds,
                                                 param_name = 'to equilibriums') # (B, L_align, A)


class EqulFromFile(neuralTKFModuleBase):
    """
    read one equilibrium distribution from numpy array file
    """
    config: dict
    name: str
    
    def setup(self):
        """
        Flax Module Parameters
        -----------------------
        None
        """
        equl_file = self.config['filenames']['equl_dist']
        
        with open(equl_file,'rb') as f:
            prob_equilibr = jnp.load(f, allow_pickle=True) #(A,) or (1,A) or (1,1,A)
        
        # if only one equlibrium distribution loaded, and it doesn't have
        #   the correct number of dimensions, fix that to (1,1,A)
        if len(prob_equilibr.shape) < 3:
            prob_equilibr = jnp.reshape(prob_equilibr,
                                        (1, 1, prob_equilibr.shape[-1])) #(1,1,A)
        
        self.logprob_equilibr = safe_log(prob_equilibr)
        
    def __call__(self,
                  *args,
                  **kwargs): 
        """
        Returns
        --------
        ArrayLike, (1, 1, A) 
            log-probability matrix for emissions from indels, which includes 
            placeholder dimensions for B and L_align
        """
        return self.logprob_equilibr #(1, 1, A) 
    

###############################################################################
### Substitution models: F81   ################################################
###############################################################################
class LocalF81(neuralTKFModuleBase):
    """
    Decide F81 model for each sample, each position; normalize then scale by
      rate multiplier for the site
     
    """
    config: dict
    name: str
    
    def setup(self):
        """
        H = number of features of input matrix
        
        Flax Module Parameters
        -----------------------
        self.final_project : Flax module
          > kernel : ArrayLike, (H, 1)
          > bias : ArrayLike, (1)  
        """
        self.rate_mult_min_val, self.rate_mult_max_val  = self.config.get( 'rate_mult_range', 
                                                                           (0.01, 10) )
        
        # only change these when debugging
        self.use_bias = self.config.get('use_bias', True)
        self.force_unit_rate_multiplier = self.config.get( 'force_unit_rate_multiplier',
                                                            False )
        
        if not self.force_unit_rate_multiplier:
            name = f'{self.name}/Project to rate multipliers'
            self.final_project = nn.Dense(features = 1,
                                          use_bias = self.use_bias,
                                          name = name)
    
    def __call__(self,
                 datamat: jnp.array,
                 padding_mask: jnp.array,
                 log_equl: jnp.array,
                 t_array: jnp.array,
                 unique_time_per_sample: bool,
                 sow_flax_intermeds: bool):
        """
        apply final linear projection and bound_sigmoid activation to get final 
          log-probability at match sites
        
        T: number of times in the grid
        B: batch size
        L_align: length of alignment
        H: input hidden dim
        A: alphabet size
        
        Arguments
        ----------
        datamat : ArrayLike, (B, L_align, H)
        
        pading_mask : ArrayLike, (B, L_align)
        
        log_equl : ArrayLike
            > if global: (1, 1, A)
            > if per-sample, per-position: (B, L_align, A); this is usually
              what I'll provide to this class
            log-transformed equilibrium distribution
        
        t_array : ArrayLike, (T,) or (B,)
            branch lengths; eventually, marginalize over this
        
        unique_time_per_sample : Bool
            whether there's one time per sample, or a grid of times you'll 
            marginalize over
        
        sow_flax_intermeds : bool
            switch for tensorboard logging
        
        Returns
        --------
        ArrayLike
          > if unique time per sample: (B, L_align, A, 2)
          > if not unique time per sample: (T, B, L_align, A, 2)
            log-probability matrix for emissions from match sites
        """
        ### rate multiplier
        if not self.force_unit_rate_multiplier:
            # (B, L_align, H) -> (B, L_align, 1) -> (B, L_align)
            rate_mult_logits = self.final_project(datamat)[...,0] # (B, L_align)
            rate_multiplier = self.apply_bound_sigmoid_activation(logits = rate_mult_logits,
                                               min_val = self.rate_mult_min_val,
                                               max_val = self.rate_mult_max_val,
                                               param_name = 'rate mult.',
                                               sow_flax_intermeds = sow_flax_intermeds) # (B, L_align)
                
        elif self.force_unit_rate_multiplier:
            final_shape = ( datamat.shape[0], datamat.shape[1] )
            rate_multiplier = jnp.ones( final_shape ) # (B, L_align)
        
        
        ### equilibrium distribution
        # return equilibrium probabilities directly
        equl = jnp.exp(log_equl) # (B, L_align, A)
        
        
        ### output
        #   if unique time per sample: (B, L_align, A, 2)
        #   if not unique time per sample: (T, B, L_align, A, 2)
        cond_logprobs = logprob_f81(equl = equl,
                                    rate_multiplier = rate_multiplier,
                                    t_array = t_array,
                                    unique_time_per_sample = unique_time_per_sample)
    
        intermed_params_dict = {'rate_multiplier': rate_multiplier}
    
        return cond_logprobs, intermed_params_dict


