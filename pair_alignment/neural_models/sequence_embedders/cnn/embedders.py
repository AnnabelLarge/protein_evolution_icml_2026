#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 17:05:47 2023


ABOUT:
======
The embedding trunk for both ancestor and descendant sequence, using:
 CONV RESNET

"""
# flax n jax
from flax import linen as nn
import jax
import jax.numpy as jnp

# custom imports
from utils.BaseClasses import SeqEmbBase


class CNNSeqEmb(SeqEmbBase):
    """
    Residual CNN that does: (1) -> norm -> conv -> act -> dropout -> += (1)
    
    
    init with:
    ==========
    initial_embed_module (callable): module for initial projection to hidden dim
    first_block_module (callable): first CNN block
    subsequent_block_module (callable): subsequent CNN blocks, if desired
    embedding_which (str): ancestor or descendant
    causal (bool): true if working with the descendant sequence; false otherwise
    config (dict): config to pass to each subsequent module
    name (str): "ANCESTOR EMBEDDER" or "DESCENDANT EMBEDDER"
    
    
    config will have:
    =================
    hidden_dim (int): length of the embedded vector

    kern_size_lst (list): list of kernel sizes 
      >> these are 1D convolutions, so each elem will be a one-element 
         list of integers: [int]

    dropout (float = 0.0): dropout rate
    
    automatically added:
    --------------------
    in_alph_size (int): <pad>, <bos>, <eos>, then all alphabet 
                              (20 for amino acids, 4 for DNA)
    seq_padding_idx (int = 0): padding token
    
    
    
    call arguments are:
    ===================
    datamat: matrix of sequences (B, L)
    training: controls behavior of intermediate dropout layers
    sow_flax_intermeds: if you want to capture intermediates for debugging
    
    
    outputs:
    ========
    datamat (altered matrix): position-specific encodings for all 
                             sequences (B, L, H)
    
    """
    initial_embed_module: callable
    first_block_module: callable
    subsequent_block_module: callable
    embedding_which: str
    causal: bool
    config: dict
    name: str
    
    def setup(self):
        # first module projects (B,L) -> (B,L,H)
        self.initial_embed = self.initial_embed_module(embedding_which = self.embedding_which,
                                                       config = self.config,
                                                       causal = self.causal,
                                                       name = f'{self.name} 0/initial embed')
        
        # second module does the first sequence embedding: (B,L,H) -> (B,L,H)
        self.first_block = self.first_block_module(config = self.config,
                                              causal = self.causal,
                                              kern_size = self.config["kern_size_lst"][0],
                                              name = f'{self.name} 1/CNN Block 0')
        
        # may have additional blocks: (B,L,H) -> (B,L,H)
        subsequent_blocks = []
        for i, kern_size in enumerate(self.config["kern_size_lst"][1:]):
            layer_idx = i + 2
            block_idx = i + 1
            l = self.subsequent_block_module(config = self.config,
                                         causal = self.causal,
                                         kern_size = kern_size,
                                         name = f'{self.name} {layer_idx}/CNN Block {block_idx}')
            subsequent_blocks.append(l)
        self.subsequent_blocks = subsequent_blocks
    
    
    def __call__(self, 
                 datamat: jnp.array, 
                 sow_flax_intermeds: bool, 
                 training: bool):
        ### 1.) initial embedding: (B,L) -> (B,L,H)
        # datamat is (B, L, H)
        # padding_mask is (B, L)
        datamat, padding_mask = self.initial_embed(datamat, training)
        
        # possibly sow values
        self.maybe_sow(sow_flax_intermeds = sow_flax_intermeds,
                       vals = datamat,
                       label = f'{self.name} 0/after initial embed',
                       include_min_max = True,
                       include_perc_zeros = False)
        
        
        ### 2.) first convolution: (B, L, H) -> (B, L, H)
        block_idx = 0
        datamat = self.first_block(datamat = datamat,
                                   padding_mask = padding_mask,
                                   sow_flax_intermeds = sow_flax_intermeds, 
                                   training = training) # (B, L, H)
        
        # possibly sow values
        self.maybe_sow(sow_flax_intermeds = sow_flax_intermeds,
                       vals = datamat,
                       label = f'{self.name} 1/after CNN Block 0',
                       include_min_max = True,
                       include_perc_zeros = True)
        
        
        ### 3.) apply successive blocks; these start at layernum=2, CNN Block 1
        # (B, L, H) -> (B, L, H)
        for i,block in enumerate(self.subsequent_blocks):
            layer_idx = i+2
            block_idx = i+1
            datamat = block(datamat = datamat,
                            padding_mask = padding_mask,
                            sow_flax_intermeds = sow_flax_intermeds, 
                            training = training) # (B, L, H)
            
            # possibly sow values
            self.maybe_sow(sow_flax_intermeds = sow_flax_intermeds,
                           vals = datamat,
                           label = f'{self.name} {layer_idx}/after CNN Block {block_idx}',
                           include_min_max = True,
                           include_perc_zeros = True)
        
        
        return datamat # (B, L, H)
    
    