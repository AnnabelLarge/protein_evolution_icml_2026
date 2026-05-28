#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 15:22:44 2024

    

modules to project (B, L) -> (B, L, H), before sending to main architecture

"""
from typing import Optional, Any, Dict

from flax import linen as nn
import jax.numpy as jnp

# custom
from utils.BaseClasses import ModuleBase


class EmbeddingWithPadding(ModuleBase):
    """
    replicated torch's embedding function, with padding_idx option 
    
    doesn't really matter if it's causal or not; keeping here to preserve trace
    
    configs have (at minimum):
    --------------------------
    hidden_dim (int): length of the embedded vector
    padding_idx (int = 0): padding token
    args.in_alph_size (int): <pad>, <bos>, <eos>, then all alphabet 
                                  (20 for amino acids, 4 for DNA)
                              
    """
    embedding_which: str
    config: Dict
    name: str
    causal: Optional[Any] = None
    
    def setup(self):
        # unpack config
        self.features = self.config['hidden_dim'] #H
        self.vocab_size = self.config['in_alph_size']
        self.seq_padding_idx = self.config.get('seq_padding_idx', 0)
        
        # layers to use
        self.initial_embedding = nn.Embed(num_embeddings = self.vocab_size, 
                                          features = self.features)
        
        
    def __call__(self, 
                 datamat: jnp.array,  #(B, L)
                 training: Optional[Any] = None):
        B = datamat.shape[0]
        L = datamat.shape[1]
        final_shape = (B, L, self.features)
        
        # padding mask
        padding_mask = (datamat != self.seq_padding_idx) #(B, L)
        padding_mask_expanded = jnp.broadcast_to(padding_mask[..., None], final_shape) # (B, L, H)
        
        # embed: (B,L) -> (B, L, H)
        datamat = self.initial_embedding(datamat) # (B, L, H)
        datamat = jnp.multiply(datamat, padding_mask_expanded) # (B, L, H)
        
        # datamat is (B, L, H)
        # padding_mask is (B, L)
        return (datamat, padding_mask)


    