#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 17:12:00 2023


Custom layers to throw into larger CNN sequence embedders


configs will have:
-------------------
- hidden_dim (int): length of the embedded vector

- kern_size_lst (list): list of kernel sizes 
  >> these are 1D convolutions, so each elem will be a one-element 
     list of integers: [int]

- in_alph_size (int = 23): <pad>, <bos>, <eos>, then all alphabet 
                              (20 for amino acids, 4 for DNA)

- dropout (float = 0.0): dropout rate


"""
from typing import Callable

from flax import linen as nn
import jax
import jax.numpy as jnp

from utils.BaseClasses import ModuleBase


class ConvnetBlock(ModuleBase):
    """
    one Conv Block:
        
       |
       v
      in --------- 
       |         |
       v         |
      norm       |
       |         |
       v         |
      conv       |
       |         |
       v         |
      silu       |
       |         | 
       v         |
    dropout      |
       |         |
       v         |
       ---> + <---
            |
            v
           out
    
    then, padding positions in "out" are reset to zeros
       
    
    (B, L, H) -> (B, L, H)
    """
    config: dict
    kern_size: int
    causal: bool
    name: str
    
    def setup(self):
        ### unpack from config
        self.hidden_dim = self.config['hidden_dim']
        self.dropout = self.config.get('dropout', 0.0)
        
        
        ### set up layers of the CNN block
        # normalization
        self.norm = nn.LayerNorm(reduction_axes=-1, feature_axes=-1)
        self.norm_type = 'Instance'
        
        # convolution
        self.conv = nn.Conv(features = self.hidden_dim,
                            kernel_size = self.kern_size,
                            strides = 1,
                            padding =  'CAUSAL' if self.causal else 'SAME')
        
        # activation
        self.act_type = 'silu'
        self.act = nn.silu
        
        # dropout
        self.dropout_layer = nn.Dropout(rate=self.dropout)
        
        
    def __call__(self, 
                 datamat: jnp.array, #(B, L, H)
                 padding_mask: jnp.array, #(B, L) 
                 sow_flax_intermeds:bool, 
                 training:bool):
        # mask for padding tokens; broadcast to the datamat input (which will
        # not change in shape through this whole operation)
        mask = jnp.broadcast_to( padding_mask[...,None], datamat.shape ) #(B, L, H)
        datamat = jnp.multiply(datamat, mask) #(B, L, H)
        
        # skip connection
        skip = datamat #(B, L, H)

        ### start block
        # 1.) norm, mask padding tokens
        datamat = self.norm(datamat)  #(B, L, H)
        datamat = jnp.multiply(datamat, mask) #(B, L, H)
        
        # 2.) convolution, mask padding tokens
        datamat = self.conv(datamat) #(B, L, H)
        datamat = jnp.multiply(datamat, mask) #(B, L, H)
        
        # 3.) activation (silu)
        datamat = self.act(datamat) #(B, L, H)
        
        # 4.) dropout
        datamat = self.dropout_layer(datamat, 
                                     deterministic = not training) #(B, L, H)
        
        
        ### residual add to the block input; again, mask padding tokens
        datamat = datamat + skip
        datamat = jnp.multiply(datamat, mask) #(B, L, H)
        
        return datamat

