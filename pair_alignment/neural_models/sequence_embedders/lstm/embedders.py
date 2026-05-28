#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 17:05:47 2023


ABOUT:
======
The embedding trunk for both ancestor and descendant sequence, using:
    LSTM
    
"""
from typing import Callable

from flax import linen as nn
import jax
import jax.numpy as jnp

from utils.BaseClasses import SeqEmbBase


class LSTMSeqEmb(SeqEmbBase):
    """
    init with:
    ==========
    initial_embed_module (callable): module for initial projection to hidden dim
    first_block_module (callable): first LSTM block
    subsequent_block_module (callable): subsequent LSTM blocks, if desired
    embedding_which (str): ancestor or descendant
    config (dict): config to pass to each subsequent module
    name (str): "ANCESTOR EMBEDDER" or "DESCENDANT EMBEDDER"
    
    
    config will have:
    =================
    n_layers (int): number of LSTM layers
    
    hidden_dim (int): length of the embedded vector
    
    padding_idx (int = 0): padding token
    
    in_alph_size (int = 23): <pad>, <bos>, <eos>, then all alphabet 
                                  (20 for amino acids, 4 for DNA)
                                  
    dropout (float = 0.0): dropout rate
    
    
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
        # !!! hard-code this
        self.return_final_carry = False
        
        
        ### unpack config
        n_layers = self.config["n_layers"]
        self.padding_idx = self.config.get("seq_padding_idx", 0)
        
        
        ### setup layers
        # first module projects (B,L) -> (B,L,H)
        name = f'{self.name} 0/initial embed'
        self.initial_embed = self.initial_embed_module(embedding_which = self.embedding_which,
                                                       config = self.config,
                                                       causal = self.causal,
                                                       name = name)
        del name
        
        # second module does the first sequence embedding: (B,L,H) -> (B,L,H)
        # note: LSTM modules don't take "causal" argument
        name = f'{self.name} 1/LSTM Block 0'
        self.first_block = self.first_block_module(config = self.config,
                                              name = name)
        del name
        
        # may have additional blocks: (B,L,H) -> (B,L,H)
        subsequent_blocks = []
        for i in range(n_layers-1):
            layer_idx = i + 2
            block_idx = i + 1
            name = f'{self.name} {layer_idx}/LSTM Block {block_idx}'
            l = self.subsequent_block_module(config = self.config,
                                         name = name)
            subsequent_blocks.append(l)
        self.subsequent_blocks = subsequent_blocks
    
    
    def __call__(self, 
                 datamat, 
                 sow_flax_intermeds: bool, 
                 training: bool):
        ### get the sequence lengths in this batch
        # (B,L) -> (B,)
        datalens = jnp.where( datamat != self.padding_idx, 1, 0 ).sum( axis=1 ) #(B,)
        
        
        ### 1.) initial embedding: (B,L) -> (B,L,H)
        # datamat is (B, L, H)
        # padding_mask is (B, L)
        datamat, padding_mask = self.initial_embed(datamat)
        
        self.maybe_sow(sow_flax_intermeds = sow_flax_intermeds,
               vals = datamat,
               label = f'{self.name} 0/after initial embed',
               include_min_max = True,
               include_perc_zeros = False)
        
        
        ### 2.) first LSTM: (B, L, H) -> (B, L, H)
        # out_carry is a tuple of two matrices, each (B, L, H)
        # datamat is (B, L, H)
        out_carry, datamat = self.first_block(datamat = datamat,
                                              datalens = datalens,
                                              training = training,
                                              carry = None)
        
        # possibly sow values
        self.maybe_sow(sow_flax_intermeds = sow_flax_intermeds,
                       vals = datamat,
                       label = f'{self.name} 1/after LSTM Block 0',
                       include_min_max = True,
                       include_perc_zeros = False)
        
        # if desired, write summary statistics of carry with: self.write_carry_wrapper
        
        
        ### 3.) apply successive blocks; these start at layernum=2, LSTM Block 1
        # (B, L, H) -> (B, L, H)
        for i,block in enumerate(self.subsequent_blocks):
            layer_idx = i+2
            block_idx = i+1
            
            # out_carry is a tuple of two matrices, each (B, L, H)
            # datamat is (B, L, H)
            out_carry, datamat = block(datamat = datamat,
                                       datalens = datalens,
                                       training = training,
                                       carry = None)
            
            # possibly sow values
            label = (f'{self.name} {layer_idx}/'+
                     f'after LSTM Block {block_idx}/')

            self.maybe_sow(sow_flax_intermeds = sow_flax_intermeds,
                           vals = datamat,
                           label = label,
                           include_min_max = True,
                           include_perc_zeros = False)
            
            del label
                
                    
        ### return the carry from the final LSTM layer, if you want
        if self.return_final_carry:
            return (out_carry, datamat)
        
        else:
            return (None, datamat)
    
    
    def write_carry(self, 
                    carry_tuple, 
                    layer_name,
                    sow_flax_intermeds):
        # cell_state and hidden_state are (B, L, H)
        cell_state, hidden_state = carry_tuple
        
        self.maybe_sow(sow_flax_intermeds = sow_flax_intermeds,
               vals = cell_state,
               label = f'{self.name}/{layer_name} cell state',
               include_min_max = True,
               include_perc_zeros = True)
        
        self.maybe_sow(sow_flax_intermeds = sow_flax_intermeds,
               vals = hidden_state,
               label = f'{self.name}/{layer_name} hidden state',
               include_min_max = True,
               include_perc_zeros = True)
        
        
    def write_carry_wrapper(self, 
                            carry_tuple, 
                            prefix,
                            sow_flax_intermeds):
        """
        helper to sow the carry for LSTMs
        
        carry is ( c, f ) if uni-directional
        carry is ( (c_fw, h_fw),  (c_rv, h_rv) ) if bidirectional
        """
        if self.causal:
            # all carry components are (B, L, H)
            fw_out_carry, rv_out_carry = out_carry
            
            self.write_carry(carry_tuple = fw_out_carry, 
                             layer_name = f'FORW_{prefix}',
                             sow_flax_intermeds = sow_flax_intermeds)
            
            self.write_carry(carry_tuple = rv_out_carry, 
                             layer_name = f'REV_{prefix}',
                             sow_flax_intermeds = sow_flax_intermeds)
        
        elif not self.causal:
            self.write_carry(carry_tuple = out_carry,
                             layer_name = prefix,
                             sow_flax_intermeds = sow_flax_intermeds)

    def apply_seq_embedder_in_training(self, **kwargs):
        # unpack kwargs
        seqs = kwargs['seqs']
        rng_key = kwargs['rng_key']
        params_for_apply = kwargs['params_for_apply']
        tstate = kwargs['tstate']
        sow_flax_intermeds = kwargs['sow_flax_intermeds']
        
        # embed the sequence
        mutable = ["sowed_intermeds"] if sow_flax_intermeds else []

        # out_carry is a tuple of two matrices, each of size (B, L, H)
        # out_embeddings is (B, L, H)
        (out_carry, out_embeddings), out_aux_dict = tstate.apply_fn(variables = params_for_apply,
                                                                   datamat = seqs,
                                                                   training = True,
                                                                   sow_flax_intermeds = sow_flax_intermeds,
                                                                   mutable = mutable,
                                                                   rngs={'dropout': rng_key})
        
        # pack up all the auxilary data 
        metrics_dict_name = f'{self.embedding_which}_layer_intermediates'
        aux_data = {f'{metrics_dict_name}/sowed_intermeds' : out_aux_dict.get("sowed_intermeds", dict()),
                    f'{metrics_dict_name}/out_carry' : out_carry}
        
        # if you ever use batch norm in ancestor sequence embedder, need 
        #  to replace this whole method and extract batch_stats from out_aux_dict
        if self.embedding_which == 'anc':
            aux_data['anc_aux'] = None
        
        return (out_embeddings, aux_data)
    
    def apply_seq_embedder_in_eval(self,
                                   seqs,
                                   tstate,
                                   sow_flax_intermeds,
                                   **kwargs):
        # embed the ancestor seq
        mutable = ["sowed_intermeds"] if sow_flax_intermeds else []
        
        # out_carry is a tuple of two matrices, each of size (B, L, H)
        # out_embeddings is (B, L, H)
        (out_carry, out_embeddings), out_aux_dict = tstate.apply_fn(variables = tstate.params,
                                                                 datamat = seqs,
                                                                 training = False,
                                                                 sow_flax_intermeds = sow_flax_intermeds,
                                                                 mutable = mutable)
        
        # pack up all the auxilary data 
        metrics_dict_name = f'{self.embedding_which}_layer_intermediates'
        aux_data = {f'{metrics_dict_name}/sowed_intermeds' : out_aux_dict.get("sowed_intermeds", dict()),
                    f'{metrics_dict_name}/out_carry' : out_carry}
        
        # if you ever use batch norm in ancestor sequence embedder, need 
        #  to replace this whole method and extract batch_stats from out_aux_dict
        if self.embedding_which == 'anc':
            aux_data['anc_aux'] = None
        
        return (out_embeddings, aux_data)
    