#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 15:52:59 2024



about:
=======

ModuleBase: gives each model the sow_flax_intermeds and summary_stats 
            helpers, for tensorboard writing

neuralTKFModuleBase: adds functions for automatically applying key 
                     activations: bound_sigmoid and log_softmax

SeqEmbBase: inherits ModuleBase and adds extra helpers for sequence embedding
            applying encoder and decoder in training/eval; the following 
            models will need newer versions (and why):
                - LSTM (uses "datalens" in argument list)
                - Transformer (handle "output attn weights" argument)
                - if you ever want to implement BatchNorm, rage quit and 
                  migrate to flax.NNX (jk)

"""
from flax import linen as nn
import jax
import jax.numpy as jnp
import optax

from neural_models.neural_hmm_predict.model_functions import bound_sigmoid

from typing import Callable, Literal
from numpy.typing import ArrayLike


class ModuleBase(nn.Module):
    def summary_stats(self, 
                      mat: ArrayLike, 
                      key_prefix: str,
                      include_min_max: bool = False,
                      include_perc_zeros: bool = False
                      ):
        """
        extract metrics from matrix
        
        NOTE: metrics could be skewed with many zeros
        
        
        arguments
        ---------
        mat : ArrayLike
            the matrix of interest
            
        key_prefix : str
            the variable name
        
        include_min_max, include per_zeros : bool
            if true, also include max, min, and percent zeros
        
        
        returns
        -------
        out_dict : dict
            dictionary containing summary stats
        """
        if mat.size == 1:
            out_dict = {f'{key_prefix}': jnp.squeeze(mat)}
        
        else:
            # always include: L2 norm, mean, variance
            l2_norm = jnp.linalg.norm(mat.reshape(-1), ord=2)
            mean = mat.mean()
            variance = mat.var()
            
            out_dict = {f'{key_prefix}/L2_NORM': l2_norm,
                        f'{key_prefix}/MEAN': mean,
                        f'{key_prefix}/VAR': variance}
            
            # optional inclusions
            if include_min_max:
                out_dict[f'{key_prefix}/MAX'] = mat.max()
                out_dict[f'{key_prefix}/MIN'] = mat.min()
            
            if include_perc_zeros:
                out_dict[f'{key_prefix}/PERC_ZEROS'] = (mat==0).sum() / mat.size
            
        return out_dict


    def sow_flax_intermeds(self, 
                           mat: ArrayLike, 
                           label: str, 
                           include_min_max: bool = False,
                           include_perc_zeros: bool = False):
        """
        helper to sow intermediate values
        
        
        arguments
        ---------
        mat : ArrayLike
            the matrix of interest
            
        label : str
            the variable name
        """
        # summarize
        out_dict = self.summary_stats(mat=mat, 
                            key_prefix=label,
                            include_min_max=include_min_max,
                            include_perc_zeros=include_perc_zeros)
        
        # sow; only keep the most recent value
        for name, value in out_dict.items():
            self.sow(col = "sowed_intermeds",
                     name = name,
                     value = value,
                     reduce_fn = lambda a, b: b)
    
    
    def maybe_sow(self,
                   sow_flax_intermeds: bool,
                   vals: ArrayLike,
                   label: str,
                   include_min_max: bool = False,
                   include_perc_zeros: bool = False):
        """
        sow_flax_intermeds : bool
            do this function or not
        
        vals : ArrayLike
            values to summarize and record
        
        label : str
            parameter name
        
        include_min_max, include_perc_zeros : bool
            include min, max, and zeros (I don't always need to do this)
        """
        if sow_flax_intermeds:
            self.sow_flax_intermeds(mat=vals, 
                                    label=label, 
                                    include_min_max=include_min_max,
                                    include_perc_zeros=include_perc_zeros)
    
    
class neuralTKFModuleBase(ModuleBase):
    """
    base class for neural TKF / neural HMM models
    
    
    methods:
    ---------
    maybe_sow : a wrapper around sow_flax_intermeds
    
    apply_bound_sigmoid_activation : a wrapper around bound_sigmoid
    
    apply_log_softmax_activation : a wrapper around log_softmax
        
    
    inherited from ModuleBase:
    --------------------------
    maybe_sow
    sow_flax_intermeds
    summary_stats
    """
    def apply_bound_sigmoid_activation(self,
                                       logits: ArrayLike,
                                       min_val: float,
                                       max_val: float,
                                       param_name: str,
                                       sow_flax_intermeds: bool,
                                       include_min_max: bool = False,
                                       include_perc_zeros: bool = False):
        """
        sigmoid(x) = 1 / ( 1 + exp(-x) )
        bound_sigmoid(x, min, max) = min + ( ( max - min ) / ( 1 + exp(-x) ) )
        """
        params = bound_sigmoid(logits, min_val, max_val) 
        self.maybe_sow( vals = params,
                         label = f'{self.name}/{param_name}',
                         sow_flax_intermeds = sow_flax_intermeds,
                         include_min_max=include_min_max,
                         include_perc_zeros=include_perc_zeros )
        
        return params 
    
    def apply_log_softmax_activation(self,
                                     logits: ArrayLike,
                                     param_name: str,
                                     sow_flax_intermeds: bool,
                                     include_min_max: bool = False,
                                     include_perc_zeros: bool = False):
        """
        log_softmax(x) = log( softmax(x) )
        """
        params = nn.log_softmax( logits, axis = -1 )
        self.maybe_sow( vals = params,
                         label = f'{self.name}/{param_name}',
                         sow_flax_intermeds = sow_flax_intermeds,
                         include_min_max=include_min_max,
                         include_perc_zeros=include_perc_zeros  )
        
        return params


class SeqEmbBase(ModuleBase):
    """
    base class for neural sequence embedding models
    
    methods:
    ---------
    apply_seq_embedder_in_training : 
        apply model during training
        
    update_seq_embedder_tstate :
        update parameters based on gradients
    
    apply_seq_embedder_in_eval :
        apply model during training
        
    
    inherited from ModuleBase:
    --------------------------
    maybe_sow
    sow_flax_intermeds
    summary_stats
    """
    def apply_seq_embedder_in_training(self,
                                       seqs: ArrayLike,
                                       tstate,
                                       rng_key,
                                       params_for_apply: dict,
                                       sow_flax_intermeds: bool,
                                       *args,
                                       **kwargs):
        """
        apply model during training
        
        
        arguments
        ----------
        seqs : ArrayLike
            inputs for function
        
        tstate : Flax.Trainstate
            trainstate for sequence embedder
        
        rng_key : Jax rng (whatever type that is)
            rng key if needed (for example: for dropout)
        
        params_for_apply : dict
            parameters for trainstate object
        
        sow_flax_intermeds : bool
            whether or not to record intermediates, broadly
            never record min, max, or percent zeros for these
        
        
        returns
        -------
        out_embeddings : ArrayLike
            per-position sequence embeddings
        
        aux_data : dict
            contains output, weight, and gradient from flax.sow, as well as
            any additional values needed for embedding the ancestor
        """
        # embed the sequence
        out_embeddings, out_aux_dict = tstate.apply_fn(variables = params_for_apply,
                                            datamat = seqs,
                                            training = True,
                                            sow_flax_intermeds = sow_flax_intermeds,
                                            mutable = ["sowed_intermeds"] if sow_flax_intermeds else [],
                                            rngs={'dropout': rng_key})
        
        # pack up all the auxilary data
        metrics_dict_name = f'{self.embedding_which}_layer_intermediates' 
        aux_data = {f'{metrics_dict_name}/sowed_intermeds' : out_aux_dict.get("sowed_intermeds", dict()) }
        
        # if you ever use batch norm in ancestor sequence embedder, need 
        #  to replace this whole method and extract batch_stats from out_aux_dict
        if self.embedding_which == 'anc':
            aux_data['anc_aux'] = None
        
        return (out_embeddings, aux_data)
    
    
    def update_seq_embedder_tstate(self, 
                                   tstate,
                                   new_opt_state,
                                   optim_updates,
                                   *args,
                                   **kwargs):
        """
        If you apply batch norm ever, you'll need a new one of these
        
        
        arguments
        ----------
        tstate : Flax.Trainstate
            trainstate for sequence embedder
        
        new_opt_state : optimizer to overwrite
        
        optim_updates : updates to apply
        
        
        returns
        --------
        new_tstate  : Flax.Trainstate
            updated trainstate
        """
        new_params = optax.apply_updates(tstate.params, 
                                         optim_updates)
        
        new_tstate = tstate.replace(params = new_params,
                                    opt_state = new_opt_state)
        
        return new_tstate
    
    
    def apply_seq_embedder_in_eval(self,
                                   seqs: ArrayLike,
                                   tstate,
                                   sow_flax_intermeds: bool,
                                   *args,
                                   **kwargs):
        """
        apply model during eval steps
        
        
        arguments
        ----------
        seqs : ArrayLike
            inputs for function
        
        tstate : Flax.Trainstate
            trainstate for sequence embedder
        
        sow_flax_intermeds : bool
            whether or not to record intermediates, broadly
            never record min, max, or percent zeros for these
        
        
        returns
        -------
        out_embeddings : ArrayLike
            per-position sequence embeddings
        
        aux_data : dict
            contains output, weight, and gradient from flax.sow, as well as
            any additional values needed for embedding the ancestor
        """
        # embed the sequence
        out_embeddings, out_aux_dict = tstate.apply_fn(variables = tstate.params,
                                        datamat = seqs,
                                        training = False,
                                        sow_flax_intermeds = sow_flax_intermeds,
                                        mutable = ["sowed_intermeds"] if sow_flax_intermeds else [])
        
        # pack up all the auxilary data 
        metrics_dict_name = f'{self.embedding_which}_layer_intermediates' 
        aux_data = {f'{metrics_dict_name}/sowed_intermeds' : out_aux_dict.get("sowed_intermeds", dict()) }
        
        # if you ever use batch norm in ancestor sequence embedder, need 
        #  to replace this whole method and extract batch_stats from out_aux_dict
        if self.embedding_which == 'anc':
            aux_data['anc_aux'] = None
        
        return (out_embeddings, aux_data)
    
