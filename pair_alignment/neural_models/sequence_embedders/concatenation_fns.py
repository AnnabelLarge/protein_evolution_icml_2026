#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 13:32:50 2023


ABOUT:
=======
These contain the concatenation functions used combine position-specific
  embeddings. 

"""
from flax import linen as nn
import jax
import jax.numpy as jnp


def extract_embs(anc_encoded, 
                 desc_encoded, 
                 idx_lst, 
                 align_idx_padding: int = -9,
                 **kwargs):
    """
    B = batch size
    L_align = length of alignment
    H = hidden dim
    
    
    extract embeddings, according to coordinates given by idx_lst
    need this as a class in order to initialize the special indexing function
      once
    
    inputs:
        - anc_encoded: full ancestor sequence embeddings, from 
          full-context encoding modules
          > (batch, seq_len, hid_dim)
          
        - desc_encoded: full descendant sequence embeddings, from 
          causal-context encoding modules
          > (batch, seq_len, hid_dim)
        
        - idx_lst: indices to concatenate; ancestor indices are in first 
          column, descendant indices are in the second
          > (batch, seq_len, 2)
        
    outputs:
        - tuple of (anc_embs, desc_embs)
          > both of size (B, L_align, H)
        - mask for alignment positions (B, L_align)
    """
    # get indexes needed; each are (B, L_align, 1)
    anc_idxes = idx_lst[:,:,0][...,None]
    desc_idxes = idx_lst[:,:,1][...,None] 
    masking_vec = ( anc_idxes != align_idx_padding )
    
    # -9 is actually causes NaNs; replace with 0 and mask this result out later
    anc_idxes = jnp.where( masking_vec,
                           anc_idxes,
                           0 )
    
    desc_idxes = jnp.where( masking_vec,
                            desc_idxes,
                            0 )
    
    # index with jnp take
    anc_selected = jnp.take_along_axis(anc_encoded, anc_idxes, axis=1)
    anc_selected = anc_selected * masking_vec
    
    desc_selected = jnp.take_along_axis(desc_encoded, desc_idxes, axis=1)
    desc_selected = desc_selected * masking_vec
    
    return [anc_selected, desc_selected], masking_vec[...,0]
    
