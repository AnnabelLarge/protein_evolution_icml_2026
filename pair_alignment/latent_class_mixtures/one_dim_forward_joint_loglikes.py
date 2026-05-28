#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 17:18:45 2025

"""
import jax
from jax import numpy as jnp
from jax.scipy.special import logsumexp
from jax.scipy.linalg import expm
from jax._src.typing import Array, ArrayLike

import numpy as np

from latent_class_mixtures.one_dim_fwd_bkwd_helpers import (init_forward_recurs,
                                                            joint_loglike_emission,
                                                            joint_forward_message_passing)


def joint_only_one_dim_forward(aligned_inputs,
                                    joint_logprob_emit_at_match,
                                    logprob_emit_at_indel,
                                    joint_logprob_transit,
                                    unique_time_per_sample: bool,
                                    return_all_intermeds: bool = False):
    """
    unique_time_per_sample = False
    
    forward algo ONLY to find joint loglike
    
    L_align: length of pairwise alignment
    T: number of timepoints
    B: batch size
    C_trans = C: number of latent site clases
      > could be C_frag or C_dom * C_frag
    A: alphabet (20 for proteins, 4 for DNA)
    S: possible states; here, this is 4: M, I, D, start/end
    
    T' = T, B' = 1 if using a grid of times for ALL samples
    T' = 1, B' = B if using one unique time per sample
    
    
    Arguments
    ----------
    aligned_inputs : ArrayLike, (B, L, 3)
        dim2=0: ancestor
        dim2=1: descendant
        dim2=2: alignment state; M=1, I=2, D=3, S=4, E=5
    
    joint_logprob_emit_at_match : ArrayLike, (T, C, A, A) or (B, C, A, A)
        logP(anc, desc | c, t); log-probability of emission at match site
    
    logprob_emit_at_indel : ArrayLike, (C, A)
        logP(anc | c) or logP(desc | c); log-equilibrium distribution
    
    joint_logprob_transit : ArrayLike, (T, C, C, S, S) or (B, C, C, S, S)
        logP(new state, new class | prev state, prev class, t); the joint 
        transition matrix for finding logP(anc, desc, align | c, t)
    
    
    Returns:
    ---------
    loglike : ArrayLike, (T', B)
    
    stacked_outputs : ArrayLike, (L_align-1, T', C, B) 
        the cache from the forward algorithm; this is the total log-probability 
        of ending at a given alignment column (l \in L_align) in class C, given
        the observed alignment
        
        to marginalize over all possible combinations of hidden site classes 
        for a given alignment: extract the final element of the length 
        dimension (i.e. stacked_outputs[-1,...]) and do logsumexp over all 
        classes C. This leaves you with the joint probability of the observed 
        alignment, at all branch lengths in T
    """
    B = aligned_inputs.shape[0]
    L_align = aligned_inputs.shape[1]
    
    # expand matrices
    if not unique_time_per_sample:
        T = joint_logprob_transit.shape[0]
        joint_logprob_emit_at_match = joint_logprob_emit_at_match[:, None, ...] #(T, 1, C, A, A) 
        joint_logprob_transit = joint_logprob_transit[:, None, ...] #(T, 1, C, C, S, S) 
        
    
    elif unique_time_per_sample:
        joint_logprob_emit_at_match = joint_logprob_emit_at_match[None, ...] #(1, B, C, A, A) 
        joint_logprob_transit = joint_logprob_transit[None, ...] #(1, B, C, C, S, S) 
        T = 1
    
    
    ### initialize with <start> -> any 
    # logP( Z_1, \tau_1 | Z_0 = <START>, t )
    init_alpha = init_forward_recurs( aligned_inputs,
                              joint_logprob_emit_at_match,
                              logprob_emit_at_indel,
                              joint_logprob_transit ) #(T', C, B)
    
    
    ######################################################
    ### scan down length dimension to end of alignment   #
    ######################################################
    def scan_fn(prev_alpha, pos):
        ### unpack
        anc_toks =   aligned_inputs[:,   pos, 0]
        desc_toks =  aligned_inputs[:,   pos, 1]

        prev_state = aligned_inputs[:, pos-1, 2]
        curr_state = aligned_inputs[:,   pos, 2]
        
        # remove invalid indexing tokens; this doesn't affect the actual 
        #   calculated loglike
        prev_state = jnp.where( prev_state!=5, prev_state, 4 )
        curr_state = jnp.where( curr_state!=5, curr_state, 4 )
        
        
        ### emissions
        e = joint_loglike_emission( aligned_inputs=aligned_inputs,
                                    pos=pos,
                                    joint_logprob_emit_at_match=joint_logprob_emit_at_match,
                                    logprob_emit_at_indel=logprob_emit_at_indel ) # (T', C, B) 
        
        
        ### message passing
        def main_body(in_carry, ps, cs):
            # logP( Z_{1:k}, c_k | Z_0 = <START>, t )
            # replace padding idx with 1 to prevent NaN gradients; this doesn't
            #   affect the actual calculated loglike
            ps = jnp.maximum(ps, 1) #(B,)
            cs = jnp.maximum(cs, 1) #(B,)
            accum_sum = joint_forward_message_passing( prev_message = in_carry, 
                                                   ps = ps, 
                                                   cs = cs, 
                                                   joint_logprob_transit = joint_logprob_transit ) #(T', C_curr, B)
            return accum_sum + e  #(T', C_curr, B)
        
        def end(in_carry, ps, cs_not_used):
            # logP( Z_{1:L}, c_{L} | Z_0 = <START>, t )
            # replace padding idx with 1 to prevent NaN gradients; this doesn't
            #   affect the actual calculated loglike
            ps = jnp.maximum(ps, 1)
            
            # simple indexing to get end state
            final_tr = joint_logprob_transit[:, jnp.arange(B), :, -1, ps-1, -1] #(B, T', C_prev)  
            final_tr = jnp.transpose( final_tr, (1,2,0) ) #(T', C_prev, B)
            
            return final_tr + in_carry #(T, C, B) 
        
        
        ### alpha update, in log space ONLY if curr_state is not pad
        new_alpha = jnp.where(curr_state != 0, 
                              jnp.where( curr_state != 4,
                                          main_body(prev_alpha, prev_state, curr_state),
                                          end(prev_alpha, prev_state, curr_state) ),
                              prev_alpha) #(T', C, B) 
        
        return (new_alpha, new_alpha)
    
    ### end scan function definition, use scan
    # stacked_outputs is cumulative sum PER POSITION, PER TIME
    idx_arr = jnp.array( [ i for i in range(2, L_align) ] ) #(L_align - 2)
    
    if not return_all_intermeds:
        # last_alpha = logP( Z_{1:L}, \tau_{L} | Z_0 = <START>, t )
        last_alpha, _ = jax.lax.scan( f = scan_fn,
                                      init = init_alpha,
                                      xs = idx_arr,
                                      length = idx_arr.shape[0] )  #(T', C, B) 
        
        # loglike = \sum_{s \in {M,I,D}} logP( Z_{1:L}, \tau_{L} = s | Z_0 = <START>, t )
        loglike = logsumexp(last_alpha, axis = 1) #(T', B) 
        if unique_time_per_sample:
            loglike = loglike[0, :] #(B,)
        
        # returns logP( Z_{1:L} | Z_0 = <START>, t )
        return loglike #(T, B) or (B,)

        
    elif return_all_intermeds:
        # stacked_outputs = logP( Z_{1:k}, \tau_k | Z_0 = <START>, t ) for all k in L
        _, stacked_outputs = jax.lax.scan( f = scan_fn,
                                            init = init_alpha,
                                            xs = idx_arr,
                                            length = idx_arr.shape[0] )  #(L_align-2, T', C, B) 
        
        # append the first return value (from sentinel -> first alignment column)
        stacked_outputs = jnp.concatenate( [ init_alpha[None,...], #(1, T', C, B)
                                             stacked_outputs ], #(L_align-2, T', C, B)
                                          axis=0) #(L_align-1, T', C, B) 
        if unique_time_per_sample: 
            stacked_outputs = stacked_outputs[:, 0, ...] #(L_align-1, C, B) 
            
        # returns logP( Z_{1:k}, \tau_k | Z_0 = <START>, t ) for all k in L
        return stacked_outputs #(L_align-1, T, C, B) or (L_align-1, C, B) 
