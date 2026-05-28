#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 17:18:45 2025


functions:
===========
'_joint_loglike_emission_per_state',
'flip_alignments',
'init_forward_recurs',
'init_marginals',
'joint_forward_message_passing',
'joint_loglike_emission',
'marginal_message_passing',
"""
import jax
from jax import numpy as jnp
from jax.scipy.special import logsumexp
from jax.scipy.linalg import expm
from jax._src.typing import Array, ArrayLike

from functools import partial


###############################################################################
### SCORING EMISSIONS   #######################################################
###############################################################################
def _joint_loglike_emission_per_state( aligned_inputs,
                                       pos,
                                       joint_logprob_emit_at_match,
                                       logprob_emit_at_indel ):
    """
    Exctact emission probabilities at column pos=k for possible match, ins, 
      or del column
    
    L: length of pairwise alignment
    T: number of timepoints
    B: batch size
    C: number of transition-level latent classes (frag mix * domain mix)
    A: alphabet size (20 for proteins, 4 for amino acids)
    
    s_k: state (M/I/D/Start/End) of column k
    Z_k = (anc_k, desc_k, s_k): observed alignment column at position k, 
      which includes emissions and state!
    c_k: transition-level latent class of column k (frag mix, domain mix)
    
    
    Arguments
    ----------
    aligned_inputs : ArrayLike, (B, L, 3)
        dim2=0: ancestor
        dim2=1: descendant
        dim2=2: alignment state; M=1, I=2, D=3, S=4, E=5
    
    pos : int
        which alignment column you're at
    
    joint_logprob_emit_at_match : ArrayLike, (T', B', C, A, A)
        if using a grid of times: T' = T, B' = 1
        if using unique time per sample: T' = 1, B' = B
        logP(anc_k, desc_k | c_k, s_k = M, t); log-probability of emission at  
          match site per transition class; emission class label has already  
          been marginalized out of this!!!
    
    logprob_emit_at_indel : ArrayLike, (C, A)
        logP(desc_k | c_k, s_k = I) or logP(anc_k | c_k, s_k = D); 
          log-equilibrium distribution per transition class; emission 
          class label has already been marginalized out of this!!!
        
    Returns
    -------
    joint_emit_if_match : ArrayLike, (T', C, B')
        emission at match column
        logP( anc_k, desc_k | c_k, s_k = Match, t )
    
    emit_if_indel_desc : ArrayLike, (C, B)
        emission at insert column
        logP( desc_k | c_k, s_k = Ins )
    
    emit_if_indel_anc : ArrayLike, (C, B)
        emission at delete column
        logP( anc_k | c_k, s_k = Del )
    """
    B = aligned_inputs.shape[0]
    
    ### unpack
    # state_at_pos = 0: Match
    # state_at_pos = 1: Ins
    # state_at_pos = 2: Del
    anc_toks = aligned_inputs[:,pos,0]-3 #(B,)
    desc_toks = aligned_inputs[:,pos,1]-3 #(B,)
    state_at_pos = aligned_inputs[:,pos,2]-1 #(B,)
    
    
    ### get all possible scores
    # logP( anc_k, desc_k | c_k, s_k = Match, t )
    joint_emit_if_match = joint_logprob_emit_at_match[:, 
                                                      jnp.arange(B),
                                                      :, 
                                                      anc_toks, 
                                                      desc_toks]  #(B, T', C)
    joint_emit_if_match = jnp.transpose(joint_emit_if_match, (1,2,0)) #(T', C, B)
    
    # logP( desc_k | c_k, s_k = Ins )
    emit_if_indel_desc = logprob_emit_at_indel[:, desc_toks] #(C, B)
    
    # logP( anc_k | c_k, s_k = Del )
    emit_if_indel_anc = logprob_emit_at_indel[:, anc_toks] #(C, B)
    
    out_lst =  [ joint_emit_if_match, 
                 emit_if_indel_desc, 
                 emit_if_indel_anc ]
    return ( joint_emit_if_match, emit_if_indel_desc, emit_if_indel_anc )
    

def joint_loglike_emission(aligned_inputs,
                           pos,
                           joint_logprob_emit_at_match,
                           logprob_emit_at_indel):
    """
    Exctact emission probabilities at column pos=k, given column type state_at_pos
    
    L: length of pairwise alignment
    T: number of timepoints
    B: batch size
    C: number of transition-level latent classes (frag mix * domain mix)
    A: alphabet size (20 for proteins, 4 for amino acids)
    
    s_k: state (M/I/D/Start/End) of column k
    Z_k = (anc_k, desc_k, s_k): observed alignment column at position k, 
      which includes emissions and state!
    c_k: transition-level latent class of column k (frag mix, domain mix)
    
    
    Arguments
    ----------
    aligned_inputs : ArrayLike, (B, L, 3)
        dim2=0: ancestor
        dim2=1: descendant
        dim2=2: alignment state; M=1, I=2, D=3, S=4, E=5
    
    pos : int
        which alignment column you're at
    
    joint_logprob_emit_at_match : ArrayLike, (T', B', C, A, A)
        if using a grid of times: T' = T, B' = 1
        if using unique time per sample: T' = 1, B' = B
        logP(anc_k, desc_k | c_k, s_k = M, t); log-probability of emission at  
          match site per transition class; emission class label has already  
          been marginalized out of this!!!
    
    logprob_emit_at_indel : ArrayLike, (C, A)
        logP(desc_k | c_k, s_k = I) or logP(anc_k | c_k, s_k = D); 
          log-equilibrium distribution per transition class; emission 
          class label has already been marginalized out of this!!!
        
    Returns
    -------
    joint_e : ArrayLike, (T', C, B)
        log-probability of emission at given column, per transition class label
    """
    T = joint_logprob_emit_at_match.shape[0]
    C = logprob_emit_at_indel.shape[0]
    B = aligned_inputs.shape[0]
    state_at_pos = aligned_inputs[:,pos,2]-1 #(B,)
    
    
    ### get all possible scores
    # joint_emit_if_match: (T', C, B)
    # emit_if_indel_desc:  (C, B)
    # emit_if_indel_anc:   (C, B)
    out = _joint_loglike_emission_per_state( aligned_inputs = aligned_inputs,
                                             pos = pos,
                                             joint_logprob_emit_at_match = joint_logprob_emit_at_match,
                                             logprob_emit_at_indel = logprob_emit_at_indel )
    joint_emit_if_match, emit_if_indel_desc, emit_if_indel_anc = out
    del out
                 
    
    ### stack all
    emit_if_indel_desc = jnp.broadcast_to( emit_if_indel_desc[None, :, :], 
                                           (T, C, B) ) #(T', C, B)
    emit_if_indel_anc = jnp.broadcast_to( emit_if_indel_anc[None, :, :], 
                                          (T, C, B) ) #(T', C, B)
    
    # joint_emissions is the following list:
    # [ logP( anc_k, desc_k | c_k, s_k = Match, t ),
    #   logP( desc_k | c_k, s_k = Ins ),
    #   logP( anc_k | c_k, s_k = Del ) ]
    joint_emissions = jnp.stack([joint_emit_if_match, 
                                 emit_if_indel_desc, 
                                 emit_if_indel_anc], axis=0) #(3, T', C, B)
    
    
    ### select the correct emission, according to state_at_pos
    # logP( Z_k | c_k, t )
    joint_e = joint_emissions[state_at_pos, :, :, jnp.arange(B)] #(B, T', C)
    joint_e = jnp.transpose( joint_e, (1,2,0) ) #(T', C, B)

    return joint_e

###############################################################################
### INIT FUNCTIONS   ##########################################################
###############################################################################
def init_forward_recurs( aligned_inputs,
                         joint_logprob_emit_at_match,
                         logprob_emit_at_indel,
                         joint_logprob_transit ):
    """
    T: number of timepoints
    B: batch size
    L: length of pairwise alignment
    C: number of latent site clases
    S: number of transitions, 4 (M, I, D, S/E)
    A: alphabet size (20 for proteins, 4 for amino acids)
    
    s_k: state (M/I/D/Start/End) of column k
    Z_k = (anc_k, desc_k, s_k): observed alignment column at position k, 
      which includes emissions and state!
    c_k: transition-level latent class of column k (frag mix, domain mix)
    
    
    Arguments
    ----------
    aligned_inputs : ArrayLike, (B, L, 3)
        dim2=0: ancestor
        dim2=1: descendant
        dim2=2: alignment state; M=1, I=2, D=3, S=4, E=5
        already reversed, if doing backward algo
    
    joint_logprob_emit_at_match : ArrayLike, (T', B', C, A, A)
        if using a grid of times: T' = T, B' = 1
        if using unique time per sample: T' = 1, B' = B
        logP(anc_k, desc_k | c_k, s_k = M, t); log-probability of emission at  
          match site per transition class; emission class label has already  
          been marginalized out of this!!!
    
    logprob_emit_at_indel : ArrayLike, (C, A)
        logP(desc_k | c_k, s_k = I) or logP(anc_k | c_k, s_k = D); 
          log-equilibrium distribution per transition class; emission 
          class label has already been marginalized out of this!!!
    
    joint_logprob_transit : ArrayLike, (T', B', C, C, S, S)
        if using a grid of times: T' = T, B' = 1
        if using unique time per sample: T' = 1, B' = B
        logP(s_k, c_k | s_{k-1}, c_{k-1}, t); the transition matrix for 
        next-column joint probability logP(X, Z | t)
    
    Returns
    -------
    ArrayLike, (T', C, B')
        initial value for forward algo
    """
    # transitions
    state_idx = aligned_inputs[:, 1, 2]-1 #(B,)
    B = state_idx.shape[0]
  
    ### transition: logP( c_1 | Z_0 = <start>, t )
    # initial state is 4 (<start>); take the last row i.e. S_prev = -1
    # use c_0=0 for start class i.e. C_prev = 0 
    start_any = joint_logprob_transit[:, :, 0, :, -1, :] #(T', B', C_curr, S_curr)
    tr = start_any[:, jnp.arange(B), :, state_idx] #(B, T', C_curr)
    tr = jnp.transpose(tr, (1, 2, 0)) #(T', C_curr, B)
    
    
    ### emission: logP( Z_1 | c_1, t )
    e = joint_loglike_emission( aligned_inputs=aligned_inputs,
                                pos=1,
                                joint_logprob_emit_at_match=joint_logprob_emit_at_match,
                                logprob_emit_at_indel=logprob_emit_at_indel ) # (T', C, B)
    
    
    ### carry value: 
    # logP( c_1 | Z_0 = <start>, t ) + logP( Z_1 | c_1, t )
    #   = logP( Z_1, c_1 | Z_0 = <start> )
    init_alpha = e + tr #(T', C, B)
    
    return init_alpha


def init_marginals(aligned_inputs,
                   logprob_emit_at_indel,
                   first_tr):
    """
    T: number of timepoints
    B: batch size
    L: length of pairwise alignment
    C: number of latent site clases
    S: number of transitions, 4 (M, I, D, S/E)
    A: alphabet size (20 for proteins, 4 for amino acids)
    
    s_k: state (emit/Start/End) of column k
    Z_k = (anc_k, desc_k, s_k): observed alignment column at position k, 
      which includes emissions and state!
    c_k: transition-level latent class of column k (frag mix, domain mix)
    
    
    Arguments
    ----------
    aligned_inputs : ArrayLike, (B, L, 3)
        dim2=0: ancestor
        dim2=1: descendant
        dim2=2: alignment state; M=1, I=2, D=3, S=4, E=5
        already reversed, if doing backward algo
    
    logprob_emit_at_indel : ArrayLike, (C, A)
        logP(desc_k | c_k, s_k = I) or logP(anc_k | c_k, s_k = D); 
          log-equilibrium distribution per transition class; emission 
          class label has already been marginalized out of this!!!
    
    first_tr : ArrayLike,
        Start -> emit transition, pre-indexed and provided as-is
    
    Returns
    -------
    ArrayLike, (T', C, B')
        initial value for forward or backward algo
    """
    # start at pos=1
    anc_toks =   aligned_inputs[:, 1, 0] #(B,)
    desc_toks =  aligned_inputs[:, 1, 1] #(B,)
    curr_state = aligned_inputs[:, 1, 2] #(B,)
    
    ### logP(anc)
    # emissions; only valid if current position is match or delete
    anc_mask = (curr_state == 1) | (curr_state == 3)  # (B,)
    init_anc_e = logprob_emit_at_indel[:, anc_toks - 3] * anc_mask  # (C, B)
    
    # transitions (if anc emitted yet)
    first_anc_emission_flag = anc_mask  # (B,)
    init_anc_tr = first_tr * first_anc_emission_flag  # (C, B)
    init_anc_alpha = init_anc_e + init_anc_tr # (C, B)
    del init_anc_e, init_anc_tr, anc_mask
    
    
    ### logP(desc); (C, B)
    # emissions; only valid if current position is match or ins
    desc_mask = (curr_state == 1) | (curr_state == 2) #(B,)
    init_desc_e = logprob_emit_at_indel[:, desc_toks - 3] * desc_mask # (C, B)
    
    # transitions (if desc emitted yet)
    first_desc_emission_flag = desc_mask # (B,)
    init_desc_tr = first_tr * first_desc_emission_flag # (C, B)
    init_desc_alpha = init_desc_e + init_desc_tr  # (C, B)
    del init_desc_e, init_desc_tr, desc_mask, curr_state
    
    return {'first_anc_emission_flag': first_anc_emission_flag,
            'first_desc_emission_flag': first_desc_emission_flag,
            'init_anc_alpha': init_anc_alpha,
            'init_desc_alpha': init_desc_alpha }


###############################################################################
### MESSAGE PASSING   #########################################################
###############################################################################
def joint_forward_message_passing( prev_message,
                                   ps,
                                   cs,
                                   joint_logprob_transit ):
    """
    T: number of timepoints
    B: batch size
    L: length of pairwise alignment
    C: number of latent site clases
    S: number of transitions, 4 (M, I, D, S/E)
    A: alphabet size (20 for proteins, 4 for amino acids)
    
    s_k: state (emit/Start/End) of column k
    Z_k = (anc_k, desc_k, s_k): observed alignment column at position k, 
      which includes emissions and state!
    c_k: transition-level latent class of column k (frag mix, domain mix)
    
    
    Arguments
    ----------
    prev_message: ArrayLike, (T', C, B)
        \alpha(k-1) = logP( Z_{1:k-1}, c_{k-1} | START, t )
    
    ps, cs: int
        previous and current alignment states (M/I/D/Start/End)
                               
    joint_logprob_transit : ArrayLike, (T', B', C, C, S, S)
        if using a grid of times: T' = T, B' = 1
        if using unique time per sample: T' = 1, B' = B
        logP(s_k, c_k | s_{k-1}, c_{k-1}, t); the transition matrix for 
          next-column joint probability logP(X, Z | t)
    
    Returns
    -------
    partial_new_message: ArrayLike, (T', C, B)
        logP( Z_{1:k-1}, c_k | START, t ); need to add emission probability
        at Z_k in order to complete message
    """
    B = joint_logprob_transit.shape[1]
    ps = ps-1 #(B,)
    cs = cs-1 #(B,)
    
    # tr_per_class = logP( c_k, s_k = cs | c_{k-1}, s_{k-1} = ps, t )
    # abbreviate this to logP( c_k | c_{k-1}, t ), since ps, cs are given
    tr_per_class = joint_logprob_transit[:,
                                          jnp.arange(B), 
                                          :,
                                          :,
                                          ps, 
                                          cs] #(B', T', C_prev, C_curr)
    tr_per_class = jnp.transpose(tr_per_class, (1,2,3,0))  #(T', C_prev, C_curr, B)
    
    
    ### prev (k-1) -> curr (k)
    # prev_message_expanded = logP( Z_{1:k-1}, c_{k-1} | START, t )
    prev_message_expanded = prev_message[:, :, None, :] #(T', C_prev, 1, B)
    
    # to_lse = logP( Z_{1:k-1}, c_{k-1} | START, t ) + logP( c_k | c_{k-1}, t ) for all possible transitions
    to_lse = prev_message_expanded + tr_per_class #(T', C_prev, C_curr, B')
    
    # partial_new_message = \LSE_h [ logP( Z_{1:k-1}, c_{k-1} = h | START, t ) + 
    #                                logP( c_k | c_{k-1} = h, t ) ]
    #                                = logP( Z_{1:k-1}, c_k | START, t )
    partial_new_message = logsumexp( to_lse, axis=1 ) #(T', C_curr, B)
    
    # emission at Z_k will be added outside this function
    return partial_new_message

    
def marginal_message_passing(prev_message, 
                             marginal_logprob_transit):
    """
    only used in 1D forward
    
    prev_message is (C_prev, B)
    marginal_logprob_transit is (C_prev, C_curr, 2, 2)
    """
    prev_message_reshaped = prev_message[:,None,:] #(C_prev, 1, B)
    marginal_logprob_transit_reshaped = marginal_logprob_transit[...,0,0][...,None] #(C_prev, C_curr, 1)
    to_logsumexp = prev_message_reshaped + marginal_logprob_transit_reshaped #(C_prev, C_curr, B)
    return logsumexp(to_logsumexp, axis=0) # (C_curr, B)



###############################################################################
### BACKWARD HELPERS   ########################################################
###############################################################################
def flip_alignments(inputs):
    """
    adapted from flax.linen.recurrent.flip_sequences
    https://github.com/google/flax/blob/ \
        c0ea12d3ecae1b87982131dbb637547b9f4eb43a/flax/linen/recurrent.py#L1180
    
    flips along axis 1, but keeps padding at the end!
    
    example:
        
        [[1, 1, 4],
         [3, 4, 1],
         [2, 2, 5],
         [0, 0, 0],
         [0, 0, 0]]
        
             |
             v
             
        [[2, 2, 5],
         [3, 4, 1],
         [1, 1, 4],
         [0, 0, 0],
         [0, 0, 0]]
        
    
    Arguments:
    ------------
    inputs : ArrayLike, (B, L, 3)
        aligned inputs
        dim0 = aligned ancestor
        dim1 = aligned descendant
        dim2 = state
       
    Returns:
    ---------
    outputs : ArrayLike, (B, L, 3)
        inputs, flipped along length axis
        
    """
    B = inputs.shape[0]
    L = inputs.shape[1]
    
    seq_lengths = (inputs[...,0] != 0).sum(axis=1) #(B,)
    max_steps = inputs.shape[1]
    seq_lengths = seq_lengths[:,None,None] #(B, 1, 1)
    
    idxs = jnp.arange(max_steps - 1, -1, -1)  # (L,)
    idxs = jnp.reshape( idxs, (1, max_steps, 1) ) #(1, L, 1)
    idxs = (idxs + seq_lengths) % max_steps  # (B, L, 1)
    idxs = jnp.broadcast_to( idxs, (B, L, 3) ) #(B, L, 3)
    
    # inputs is (63, 1, 4); why does it have extra dimension?
    # ValueError: Incompatible shapes for broadcasting: shapes=[(63, 2326, 3), (63, 1, 4)]
    outputs = jnp.take_along_axis( inputs, idxs, axis=1 )
    
    return outputs
