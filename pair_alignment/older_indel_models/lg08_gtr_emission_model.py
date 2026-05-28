#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 17 17:53:01 2025

"""
import jax
from jax import numpy as jnp

from jax.typing import ArrayLike

from latent_class_mixtures.model_functions import ( safe_log,
                                                    rate_matrix_from_exch_equl,
                                                    cond_logprob_emit_at_match_per_mixture )


class LG08Logprobs:
    def __init__(self,
                 log_equl_dist: ArrayLike,
                 norm: bool = True):
        """
        Get the conditional logprobs for a GTR model using the LG08 rate 
        matrix
        
        init with
        ----------
        log_equl_dist : ArrayLike, (A)
            log-transformed equilibrium distribution
        """
        # read LG08 rate matrix
        with open(f'older_indel_models/LG08_exchangeability_r.npy', 'rb') as f:
            exchangeabilities_mat = jnp.load(f) #(20, 20)
        
        # undo log transform on equilibrium
        equl = jnp.exp(log_equl_dist)[None, None, :] #(1, 1, A)
        
        # prepare rate matrix Q_c = \chi * \diag(\pi_c); normalize such that 
        #   t=1 is one substitution
        rate_matrix_Q = rate_matrix_from_exch_equl( exchangeabilities = exchangeabilities_mat,
                                                    equilibrium_distributions = equl,
                                                    norm = True ) #(1, 1, A, A)
        self.rate_matrix_Q = rate_matrix_Q
        
    def __call__(self,
                 t_array: ArrayLike,
                 *args,
                 **kwargs):
        """
        t_array : ArrayLike, (B,) or (T,)
            times to evaluate at
        """
        # cond_logprobs is either (T, 1, 1, 1, A, A) or (B, 1, 1, 1, A, A)
        cond_logprobs = cond_logprob_emit_at_match_per_mixture( t_array = t_array,
                                     scaled_rate_mat_per_mixture = self.rate_matrix_Q )
        
        # remove unused dims
        cond_logprobs = cond_logprobs[:, 0, 0, 0, :, :] #(T, A, A) or (B, A, A)
        
        return cond_logprobs #(T, A, A) or (B, A, A)


def equl_dist_logprobs_from_counts( training_dset_emit_counts,
                                    *args,
                                    **kwargs ):
    """
    Construct an equilibrium distribution from observed frequencies
    """
    equl_dist = training_dset_emit_counts / ( training_dset_emit_counts.sum() ) #(A,)
    log_equl_dist = safe_log( equl_dist ) #(A,)
    
    return log_equl_dist #(A,)
