#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 02:31:21 2023


ABOUT:
======
Use these projection layers when generating possible descendant + alignment,
  conditioned on an ancestor sequence

(note: REMOVE <bos> token from output alphabet)

full amino acid alphabet is 44 tokens:
  - <pad>: 0
  - <bos>: 1
  - <eos>: 2
  - 20 AAs + match: 3-22
  - 20 AAs + insert: 23 - 42
  - gap: 43

full DNA alphabet would be 12 tokens:
  - <pad>: 0
  - <bos>: 1
  - <eos>: 2
  - 4 nucls + match: 3-6
  - 4 nucls + insert: 7-10
  - gap: 11

  
"""
import logging
from typing import Any, Callable, Sequence, Union, Tuple
from dataclasses import field

from flax import linen as nn
import jax
import jax.numpy as jnp
import optax

from utils.BaseClasses import ModuleBase
from neural_models.neural_shared.postprocessing_models import (FeedforwardPostproc,
                                                       SelectMask)

class FeedforwardPredict(ModuleBase):
    """
    process concatenated feature vectors into output alphabet;
    
    repeat [norm -> dense -> activation -> dropout] blocks
      > norm in first block can be different from norm in other blocks
      > norm can be None (in which case, no normalization applied)
      > could have no blocks at all
    
    then, use final dense layer to project from last block's hidden dimension
      to out_alph_size
    """
    config: dict
    name: str
    
    def setup(self):
        # read config
        self.postproc_model_type = self.config['postproc_model_type']
        self.output_size = self.config['out_alph_size']
        self.use_bias = self.config.get('use_bias', True)
        
        # setup up postprocessing layers
        postproc_module_registry = {'selectmask': SelectMask,
                                    'feedforward': FeedforwardPostproc,
                                    None: lambda datamat, *args, **kwargs: None}
        
        postproc_module = postproc_module_registry[self.postproc_model_type]
        self.postproc = postproc_module( config = self.config,
                                         name = f'{self.name}/{self.postproc_model_type}_postproc' )
        
        # final projection to alignment-augmented alphabet
        self.final_proj = nn.Dense(features = self.output_size, 
                                  use_bias = self.use_bias, 
                                  kernel_init = nn.initializers.lecun_normal(),
                                  name=f'{self.name}/final projection')
        
    @nn.compact
    def __call__(self, 
                 datamat_lst: list, 
                 padding_mask: jnp.array, # (B, L)
                 t_array: jnp.array, #(B,) or None
                 training: bool, 
                 sow_flax_intermeds: bool=False,
                 *args,
                 **kwargs):
        # elements of datamat_lst are:
        # anc_embeddings: (B, L, H)
        # desc_embeddings: (B, L, H)
        # prev_align_one_hot_vec: (B, L, 5)
        # if times are provided, append those too
        datamat = self.postproc( anc_emb = datamat_lst[0],
                                 desc_causal_emb = datamat_lst[1],
                                 prev_align_one_hot_vec = datamat_lst[2],
                                 padding_mask = padding_mask,
                                 training = training,
                                 sow_flax_intermeds = sow_flax_intermeds,
                                 t_array = t_array )  #(B, L, H_out)
        
        # get final logits, mask, and return: (B, L, H_out) -> (B, L, A_aug - 1)
        final_logits = self.final_proj(datamat) #(B, L, A_aug - 1)
        expanded_mask = jnp.broadcast_to( padding_mask[...,None], final_logits.shape ) #(B, L, A_aug - 1)
        final_logits = jnp.multiply( final_logits, expanded_mask ) #(B, L, A_aug - 1)
        
        return final_logits
    

    def neg_loglike_in_scan_fn( self, 
                                final_logits: jnp.array, # (B, L, A_aug - 1)
                                padding_mask: jnp.array, # (B, L)
                                true_out: jnp.array, # (B, L)
                                return_result_before_sum: bool = False,
                                *args,
                                **kwargs ):
        """
        Cross-entropy loss per position, along with collecting accuracy metrics
        """
        
        ### loss
        # logP(desc, align | anc) comes from -cross_ent()
        logprob_perSamp_perPos = -optax.softmax_cross_entropy_with_integer_labels(logits = final_logits, 
                                                                                  labels = true_out) # (B, L)
        logprob_perSamp_perPos = jnp.multiply( logprob_perSamp_perPos, padding_mask ) #(B, L)
        logprob_perSamp = logprob_perSamp_perPos.sum(axis=-1) #(B, L)
        
        
        ### accuracy
        # get predicted outputs
        pred_outputs = jnp.argmax(final_logits, axis=-1) #(B, L)
        
        # accumulate matches
        correct_predictions_perSamp = (pred_outputs == true_out) & (padding_mask) #(B, L)
        correct_predictions_perSamp = correct_predictions_perSamp.sum(axis=-1) #(B)
        valid_positions_perSamp = padding_mask.sum(axis=-1) #(B)
        
        out_dict = {'logprob_perSamp': logprob_perSamp, #(B)
                    'correct_predictions_perSamp': correct_predictions_perSamp,  #(B)
                    'valid_positions_perSamp': valid_positions_perSamp} #(B)
        
        if return_result_before_sum:
            out_dict['logprob_perSamp_perPos'] = logprob_perSamp_perPos #(B, L)
        
        return out_dict
    
    def evaluate_loss_after_scan(self, 
                                 scan_dict, # dictionary
                                 length_for_normalization_for_reporting, #(B, )
                                 *args,
                                 **kwargs):
        # loss
        logprob_perSamp = scan_dict['logprob_perSamp'] #(B,)
        loss = -jnp.mean(logprob_perSamp) # one float value
        
        # accuracy
        correct_predictions_perSamp = scan_dict['correct_predictions_perSamp'] #(B,)
        valid_positions_perSamp = scan_dict['valid_positions_perSamp'] #(B,)
        acc_perSamp = correct_predictions_perSamp / valid_positions_perSamp #(B,)
        
        # outputs
        # don't keep correct_predictions_perSamp or valid_positions_perSamp
        intermediate_vals = { 'sum_neg_logP': -logprob_perSamp,
                              'neg_logP_length_normed': -logprob_perSamp/length_for_normalization_for_reporting,
                              'acc_perSamp': acc_perSamp }
        
        return loss, intermediate_vals
    
    def get_perplexity_per_sample(self,
                                  loss_fn_dict):
        neg_logP_length_normed = loss_fn_dict['neg_logP_length_normed']
        perplexity_perSamp = jnp.exp(neg_logP_length_normed) #(B,)
        return perplexity_perSamp
