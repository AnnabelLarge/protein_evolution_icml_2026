#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 15:18:44 2024


functions:
-----------
'score_f81_substitutions_marg_over_times',
'score_f81_substitutions_t_per_samp',
'score_indels',
'score_transitions'

"""
import jax
from jax import numpy as jnp

###############################################################################
### helpers   #################################################################
###############################################################################
def index_square_scoring_matrix_marg_over_time(scoring_matrix,
                                               samples):
    """
    helper to index a 2D scoring matrix (used for transitions and substitution 
        logprobs)
    
    T: number of times
    B: batch size
    L_align: length of alignment
    N: last two dimensions of scoring_matrix; this is the square part
    
    Arguments
    ---------
    scoring_matrix : ArrayLike, (T, B, L_align-1, N, N) OR (T, 1, 1, N, N)
        scoring matrix of log-probabilities
    
    samples : ArrayLike, (B, L-align-1, 2)
        dim2=0 are the row indices
        dim2=1 are the col indices
    
    Returns
    -------
    raw_logprobs : ArrayLike, (T, B, L_align)
        elements of scoring_matrix, extracted according to samples 
        (still needs to be masked, selected, whatever)
    """
    T = scoring_matrix.shape[0]
    B = samples.shape[0]
    L = samples.shape[1]
    N = scoring_matrix.shape[-1] 
    
    # global: one matrix for all samples, all positions
    if scoring_matrix.shape == (T, 1, 1, N, N):
        scoring_matrix = scoring_matrix[:,0,0,...] #(T, N, N)

        row_idx = samples[..., 0][None,...] #(1, B, L_align-1)
        row_idx = jnp.broadcast_to(row_idx, (T, B, L)) #(T, B, L_align-1)

        col_idx = samples[..., 1][None,...] #(1, B, L_align-1)        
        col_idx = jnp.broadcast_to(col_idx, (T, B, L)) #(T, B, L_align-1)
        
        raw_logprobs = scoring_matrix[jnp.arange(T)[:, None, None], row_idx, col_idx]  # (T, B, L_align-1)
                
    
    # local: unique transition matrix for each sample, each position
    elif scoring_matrix.shape == (T, B, L, N, N):
        row_idx = samples[..., 0][None, ..., None,None] #(1, B, L_align-1, 1, 1)
        interm = jnp.take_along_axis(scoring_matrix, 
                                     row_idx,
                                     axis=-2)  #(T, B, L_align-1, 1, N)
        
        col_idx = samples[..., 1][None, ..., None,None] #(1, B, L_align-1, 1, 1)
        raw_logprobs = jnp.take_along_axis(interm, 
                                           col_idx, 
                                           axis=-1)  #(T, B, L_align-1, 1, 1)
        
        # squash to (T, B, L_align-1)
        raw_logprobs = raw_logprobs[...,0,0] 
    
    return raw_logprobs


def index_square_scoring_matrix_t_per_samp(scoring_matrix,
                                           samples):
    """
    helper to index a 2D scoring matrix (used for transitions and substitution 
        logprobs)
    
    B: batch size
    L_align: length of alignment
    N: last two dimensions of scoring_matrix; this is the square part
    
    Arguments
    ---------
    scoring_matrix : ArrayLike, (B, L_align-1, N, N) OR (1, 1, N, N)
        scoring matrix of log-probabilities
    
    samples : ArrayLike, (B, L-align-1, 2)
        dim2=0 are the row indices
        dim2=1 are the col indices
    
    Returns
    -------
    raw_logprobs : ArrayLike, (B, L_align)
        elements of scoring_matrix, extracted according to samples 
        (still needs to be masked, selected, whatever)
    """
    B = samples.shape[0]
    L = samples.shape[1]
    N = scoring_matrix.shape[-1] 
    
    # global: one transition matrix for all samples, all positions
    if scoring_matrix.shape == (B, 1, N, N):
        scoring_matrix = scoring_matrix[:,0,...] # (B, N, N)

        row_idx = samples[..., 0] # (B, L_align-1)
        col_idx = samples[..., 1] # (B, L_align-1) 
        batch_idx = jnp.arange(B)[:, None]  # (B, 1)
        
        raw_logprobs = scoring_matrix[batch_idx, row_idx, col_idx]  # (B, L_align-1)
                
    
    # local: unique transition matrix for each sample, each position
    elif scoring_matrix.shape == (B, L, N, N):
        # get rows: B, L_align-1, 1, S
        row_idx = samples[..., 0][..., None,None] # (B, L_align-1, 1, 1)
        interm = jnp.take_along_axis(scoring_matrix, 
                                     row_idx,
                                     axis=-2)  # (B, L_align-1, 1, S)
        
        # get columns: B, L_align-1, 1, 1
        col_idx = samples[..., 1][..., None,None] # (B, L_align-1, 1, 1)
        raw_logprobs = jnp.take_along_axis(interm, 
                                           col_idx, 
                                           axis=-1)  # (B, L_align-1, 1, 1) 
        
        # squash to (B, L_align-1)
        raw_logprobs = raw_logprobs[...,0,0] 
    
    return raw_logprobs


def preproc_emissions( samples: jnp.array,
                       alphabet_size: int,
                       gap_idx: int=43,
                       padding_idx: int=0,
                       start_idx: int=1,
                       end_idx: int=2 ):
    """
    B: batch size
    L_align: length of alignment
    
    Arguments
    ----------
    samples : ArrayLike, (B, L_align-1, ...)
        inputs to remap
    
    alphabet_size : int
        replace all the special tokens with this value
    
    gap_idx, padding_idx, start_idx, end_idx : int
        special tokens, encoded as:
        <pad>: 0
        <start>: 1
        <end>: 2
        <gap>: 43
    
    
    Returns:
    ----------
    samples_adj : ArrayLike, (B, L_align-1, ...)
        after replacing special tokens, and shifting values down by three
    """
    # map <pad>, <bos>, <eos>, and <gap> to last token, so that jax doesn't
    # have invalid indexing and NaN gradients
    specials = ( (samples == padding_idx) | 
                 (samples == start_idx) | 
                 (samples == end_idx) | 
                 (samples == gap_idx) )
    samples_adj = jnp.where(specials, alphabet_size, samples)
    
    # remap tokens, to account for the <pad>, <bos>, <eos> tokens in the 
    # beginning of the alphabet; for example, for proteins:
    # A: 0
    # C: 1
    # D: 2
    # (etc.)
    samples_adj = samples_adj - 3 
    return samples_adj
    

###############################################################################
### score transitions   #######################################################
###############################################################################
def score_transitions(staggered_alignment_state: jnp.array, 
                      logprob_trans_mat: jnp.array,
                      unique_time_per_sample: bool,
                      padding_idx: int = 0):
    """
    T: number of times (only a valid dimension if unique_time_per_sample)
    B: batch size
    L_align: length of alignment
    S: number of transition states, here, it's 4
    
    
    Arguments
    ----------
    staggered_alignment_state : ArrayLike, (B, L_align-1, 2)
      > dim2=0: prev position's state
      > dim2=1: curr position's state (the position you're trying to predict)
      > <pad>: 0
      > Match, M: 1
      > Insert, I: 2
      > Delete, D: 3
      > Start, S: 4
      > End, E: 5
     
    unique_time_per_sample : bool
      > True if using a unique branch length per sample
      
    logprob_trans_mat : ArrayLike
      > if unique_time_per_sample: (B, L_align-1, 4, 4) OR (B, 1, 4, 4)
      > if not unique_time_per_sample: (T, B, L_align-1, 4, 4) OR (T, 1, 1, 4, 4)
      > order of rows and columns: M, I, D, S/E
    
    padding_idx: int
      > default = 0
    
    
    Returns:
    ----------
    final_logprobs : ArrayLike
      > if unique_time_per_sample: (B, L_align-1)
      > if not unique_time_per_sample: (T, B, L_align-1)
    """
    ### preprocess
    # get padding mask, do this BEFORE adjusting the encoding!!!
    padding_mask =  ( (staggered_alignment_state[...,0] != padding_idx) &
                      (staggered_alignment_state[...,1] != padding_idx) ) #(B, L_align-1)
    
    # adjust encoding
    # end: write as combined start/end token (5 -> 4)
    # pad: write as delete, so that indexing invalid positions doesn't cause 
    #      jax gradients to be NaN (0 -> 3)
    alignment_state_adj = jnp.where(staggered_alignment_state != 5, staggered_alignment_state, 4) # (B, L_align-1, 2)
    alignment_state_adj = jnp.where(alignment_state_adj != 0, alignment_state_adj, 3) # (B, L_align-1, 2)
    
    # by how much do you offset tokens for indexing the transition matrix? 
    # default offset is 1, which remaps tokens to:
    # > Match, M: 0
    # > Insert, I: 1
    # > Delete, D, and pad: 2
    # > Start, S, and End, E: 3
    token_offset = 1
    
    # move all positions down
    alignment_state_adj = alignment_state_adj - token_offset
    
    ### Score, mask, return
    if unique_time_per_sample:
        indexing_fn = index_square_scoring_matrix_t_per_samp
    
    elif not unique_time_per_sample:
        indexing_fn = index_square_scoring_matrix_marg_over_time
        padding_mask = padding_mask[None, ...] #(1, B, L_align-1)
        
    raw_logprobs = indexing_fn( scoring_matrix = logprob_trans_mat,
                                samples = alignment_state_adj ) #(T, B, L_align-1) or (B, L_align-1)
    final_logprobs = raw_logprobs * padding_mask #(T, B, L_align-1) or (B, L_align-1)
    return final_logprobs  #(T, B, L_align-1) or (B, L_align-1)


###############################################################################
### score emissions from indel sites   ########################################
###############################################################################
def score_indels(true_alignment_without_start: jnp.array, 
                 logprob_scoring_vec: jnp.array, 
                 which_seq: str,
                 gap_idx: int=43,
                 padding_idx: int=0,
                 start_idx: int=1,
                 end_idx: int=2):
    """
    T: number of times
    B: batch size
    L_align: length of alignment
    A: alphabet size
    
    
    Arguments
    ----------
    true_alignment_without_start : ArrayLike, (B, L_align - 1, 3)
        given alignment, not including start
        > dim2=0: gapped ancestor
        > dim2=1: gapped descendant
      
    logprob_scoring_vec : ArrayLike (B, L_align - 1, A) OR (1, 1, A)
        equilibrium distribution
    
    gap_idx, padding_idx, start_idx, end_idx : int
        special tokens, encoded as:
        <pad>: 0
        <start>: 1
        <end>: 2
        <gap>: 43

    which_seq : ['anc','desc'] 
        'anc' to score ancestor, 'desc' to score descendant
    
    Returns:
    ---------
    final_logprobs: (B, L_align - 1)
      
    """
    ### preprocess
    # dims
    B = true_alignment_without_start.shape[0]
    L = true_alignment_without_start.shape[1] #this is L_align-1
    A = logprob_scoring_vec.shape[-1]
    
    # determine which to index
    if which_seq == 'anc':
        residue_tokens = true_alignment_without_start[...,0] #(B, L-align-1)
    
    elif which_seq == 'desc':
        residue_tokens = true_alignment_without_start[...,1] #(B, L-align-1)
    
    # create mask BEFORE remapping tokens
    padding_mask = ~( (residue_tokens == padding_idx) | 
                      (residue_tokens == gap_idx) |
                      (residue_tokens == start_idx) |
                      (residue_tokens == end_idx) )  #(B, L-align-1)
    
    # remap
    residue_tokens_adj = preproc_emissions( samples = residue_tokens,
                                            alphabet_size = A,
                                            gap_idx = gap_idx,
                                            padding_idx = padding_idx,
                                            start_idx = start_idx,
                                            end_idx = end_idx ) #(B, L_align-1)
    
    
    ### score
    logprob_scoring_vec = jnp.broadcast_to( logprob_scoring_vec,
                                            (B,L,A) ) #(B,L_align-1,A)
    
    raw_logprobs = jnp.take_along_axis(arr=logprob_scoring_vec, 
                                         indices=residue_tokens_adj[...,None], 
                                         axis=-1)[...,0] #(B, L_align-1)
        
    
    ### mask and return
    final_logprobs = raw_logprobs * padding_mask #(B, L_align-1)
    
    return final_logprobs #(B, L_align-1)



###############################################################################
### score emissions from match sites   ########################################
###############################################################################
### F81: two separate implementations
def score_f81_substitutions_marg_over_times(true_alignment_without_start: jnp.array, 
                                            logprob_scoring_mat: jnp.array, 
                                            gap_idx: int=43,
                                            padding_idx: int=0,
                                            start_idx: int=1,
                                            end_idx: int=2,
                                            *args,
                                            **kwargs):
    """
    T: number of times
    B: batch size
    L_align: length of alignment
    A: alphabet size
    
    
    Arguments
    ----------
    true_alignment_without_start : ArrayLike, (B, L_align - 1, 3)
        given alignment, not including start
        > dim2=0: gapped ancestor
        > dim2=1: gapped descendant
      
    logprob_scoring_mat : ArrayLike (T, B, L_align, A, 2) OR (T, 1, 1, A, 2)
        logprob of match/mismatch
        > dim3 corresponds with EMITTED residue (i.e. identity of descendant token)
        > dim4=0: logprob of descendant token if site is a MATCH (anc==desc)
        > dim4=1: logprob of descendant token if site is a SUBSTITUTION (anc!=desc)
    
    gap_idx, padding_idx, start_idx, end_idx : int
        special tokens, encoded as:
        <pad>: 0
        <start>: 1
        <end>: 2
        <gap>: 43
        
    Returns:
    ---------
    final_logprobs : ArrayLike, (T, B, L_align-1)
        log-probability of match or mismatch at each site
    """
    ### preprocess
    # dims
    T = logprob_scoring_mat.shape[0]
    B = true_alignment_without_start.shape[0]
    L = true_alignment_without_start.shape[1] #this is L_align-1
    A = logprob_scoring_mat.shape[-2]
    
    # create masks BEFORE remapping tokens
    anc_toks = true_alignment_without_start[...,0] #(B, L_align-1)
    desc_toks = true_alignment_without_start[...,1] #(B, L_align-1)

    anc_valid_tok = ~( (anc_toks == padding_idx) | 
                       (anc_toks == gap_idx) |
                       (anc_toks == start_idx) |
                       (anc_toks == end_idx) ) #(B, L-align-1)
    
    desc_valid_tok = ~( (desc_toks == padding_idx) | 
                        (desc_toks == gap_idx) |
                        (desc_toks == start_idx) |
                        (desc_toks == end_idx) )  #(B, L-align-1)
    
    match_pos = anc_valid_tok & desc_valid_tok & (anc_toks==desc_toks) #(B, L-align-1)
    sub_pos =   anc_valid_tok & desc_valid_tok & (anc_toks!=desc_toks) #(B, L-align-1)
    
    padding_mask = ~( (desc_toks == padding_idx) | 
                      (anc_toks == gap_idx) |
                      (desc_toks == gap_idx) |
                      (desc_toks == end_idx) )  #(B, L-align-1)
    
    # remap
    desc_toks_adj = preproc_emissions( samples = desc_toks,
                                       alphabet_size = A,
                                       gap_idx = gap_idx,
                                       padding_idx = padding_idx,
                                       start_idx = start_idx,
                                       end_idx = end_idx ) #(B, L_align-1)
    
    
    ### score
    # split scoring matrix
    logprob_matrix_match = logprob_scoring_mat[...,0] #(T,B,L_align-1,A) or (T, 1, 1, A)
    logprob_matrix_subs = logprob_scoring_mat[...,1] #(T,B,L_align-1,A) or (T, 1, 1, A)
    del logprob_scoring_mat
    
    if logprob_matrix_match.shape == (T,1,1,A):
        logprob_matrix_match = jnp.broadcast_to(logprob_matrix_match, (T,B,L,A)) #(T,B,L_align-1,A)
        logprob_matrix_subs = jnp.broadcast_to(logprob_matrix_subs, (T,B,L,A)) #(T,B,L_align-1,A)
    
    
    # score both
    desc_toks_adj = desc_toks_adj[None,...,None] # (1, B, L_align-1, 1)
    score_if_match = jnp.take_along_axis(logprob_matrix_match, desc_toks_adj, axis=3)[...,0]  # (T, B, L_align-1)
    score_if_subs = jnp.take_along_axis(logprob_matrix_subs, desc_toks_adj, axis=3)[...,0]  # (T, B, L_align-1)
    
    # use previous masking to select
    raw_logprob_match = jnp.where( match_pos,
                                   score_if_match,
                                   0 )  # (T, B, L_align-1)
    
    raw_logprob_subs = jnp.where( sub_pos,
                                  score_if_subs,
                                  0 )  # (T, B, L_align-1)
    
    raw_logprob = raw_logprob_match + raw_logprob_subs
    final_logprob = raw_logprob * padding_mask[None,...]
    return final_logprob # (T, B, L_align-1)
        

def score_f81_substitutions_t_per_samp(true_alignment_without_start: jnp.array, 
                                       logprob_scoring_mat: jnp.array, 
                                       gap_idx: int=43,
                                       padding_idx: int=0,
                                       start_idx: int=1,
                                       end_idx: int=2,
                                       *args,
                                       **kwargs):
    """
    B: batch size
    L_align: length of alignment
    A: alphabet size
    
    
    Arguments
    ----------
    true_alignment_without_start : ArrayLike, (B, L_align - 1, 3)
        given alignment, not including start
        > dim2=0: gapped ancestor
        > dim2=1: gapped descendant
      
    logprob_scoring_mat : ArrayLike (B, L_align, A, 2) OR (B, 1, A, 2)
        logprob of match/mismatch
        > dim2 corresponds with EMITTED residue (i.e. identity of descendant token)
        > dim3=0: logprob of descendant token if site is a MATCH (anc==desc)
        > dim3=1: logprob of descendant token if site is a SUBSTITUTION (anc!=desc)
    
    gap_idx, padding_idx, start_idx, end_idx : int
        special tokens, encoded as:
        <pad>: 0
        <start>: 1
        <end>: 2
        <gap>: 43
        
    Returns:
    ---------
    final_logprobs : ArrayLike, (B, L_align-1)
        log-probability of match or mismatch at each site
    """
    ### preprocess
    # dims
    B = true_alignment_without_start.shape[0]
    L = true_alignment_without_start.shape[1] #this is L_align-1
    A = logprob_scoring_mat.shape[-2]
    
    # create masks BEFORE remapping tokens
    anc_toks = true_alignment_without_start[...,0] #(B, L_align-1)
    desc_toks = true_alignment_without_start[...,1] #(B, L_align-1)

    anc_valid_tok = ~( (anc_toks == padding_idx) | 
                       (anc_toks == gap_idx) |
                       (anc_toks == start_idx) |
                       (anc_toks == end_idx) ) #(B, L-align-1)
    
    desc_valid_tok = ~( (desc_toks == padding_idx) | 
                        (desc_toks == gap_idx) |
                        (desc_toks == start_idx) |
                        (desc_toks == end_idx) )  #(B, L-align-1)
    
    match_pos = anc_valid_tok & desc_valid_tok & (anc_toks==desc_toks) #(B, L-align-1)
    sub_pos =   anc_valid_tok & desc_valid_tok & (anc_toks!=desc_toks) #(B, L-align-1)
    
    padding_mask = ~( (desc_toks == padding_idx) | 
                      (anc_toks == gap_idx) |
                      (desc_toks == gap_idx) |
                      (desc_toks == end_idx) )  #(B, L-align-1)
    
    # remap
    desc_toks_adj = preproc_emissions( samples = desc_toks,
                                       alphabet_size = A,
                                       gap_idx = gap_idx,
                                       padding_idx = padding_idx,
                                       start_idx = start_idx,
                                       end_idx = end_idx ) #(B, L_align-1)
    
    
    ### score
    # split scoring matrix
    logprob_matrix_match = logprob_scoring_mat[...,0] #(B,L_align-1,A) or (B,1,A)
    logprob_matrix_subs = logprob_scoring_mat[...,1] #(B,L_align-1,A) or (B,1,A)
    del logprob_scoring_mat
    
    if logprob_matrix_match.shape == (B,1,A):
        logprob_matrix_match = jnp.broadcast_to(logprob_matrix_match, (B,L,A)) #(B,L_align-1,A)
        logprob_matrix_subs = jnp.broadcast_to(logprob_matrix_subs, (B,L,A)) #(B,L_align-1,A)
    
    
    batch_idx = jnp.arange(B)[:, None]  # (B, 1)
    pos_idx = jnp.arange(L)[None, :]  # (1, L)
    
    score_if_match = logprob_matrix_match[batch_idx, pos_idx, desc_toks_adj] #(B,L_align-1)
    score_if_subs = logprob_matrix_subs[batch_idx, pos_idx, desc_toks_adj] #(B,L_align-1)
    
    # use previous masking to select
    raw_logprob_match = jnp.where( match_pos,
                                   score_if_match,
                                   0 )  # (T, B, L_align-1)
    
    raw_logprob_subs = jnp.where( sub_pos,
                                  score_if_subs,
                                  0 )  # (T, B, L_align-1)
    
    raw_logprob = raw_logprob_match + raw_logprob_subs
    final_logprob = raw_logprob * padding_mask  # (B, L_align-1)
    return final_logprob # (B, L_align-1)
