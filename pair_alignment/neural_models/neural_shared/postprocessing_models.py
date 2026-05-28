#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 12:44:26 2025


About:
=======
Concatenated outputs from both sequence embedders and postprocess for 
downstream blocks that create logits, features, etc.


classes available:
==================
1.) SelectMask: concatenate outputs from sequence embedders, possibly norm
2.) FeedforwardPostproc: repeat blocks of norm -> dense -> act -> dropout

"""
from flax import linen as nn
import jax
import jax.numpy as jnp

from typing import Optional

from utils.BaseClasses import ModuleBase


class SelectMask(ModuleBase):
    config: dict
    name: str
    
    def setup(self):
        self.use_anc_emb = self.config['use_anc_emb']
        self.use_desc_emb = self.config['use_desc_emb']
        self.use_prev_align_info = self.config['use_prev_align_info']
        self.use_t_per_sample = self.config.get('t_per_sample', False)
        self.normalize_seq_embeddings_before_block = self.config.get('normalize_seq_embeddings_before_block', False)
        
        if self.normalize_seq_embeddings_before_block:
            self.norm = nn.LayerNorm( reduction_axes= -1, 
                                      feature_axes=-1,
                                      name = f'{self.name}/InstanceNorm')
    
    def _mask_padding_tokens( self,
                             x: jnp.array, 
                             mask: jnp.array ):
        expanded_mask = jnp.broadcast_to( mask[...,None], x.shape ) 
        return jnp.multiply(expanded_mask, x)
    
    
    def _concatenate_and_norm( self,
                               padding_mask: jnp.array,
                               anc_emb: Optional[jnp.array] = None,
                               desc_causal_emb: Optional[jnp.array] = None,
                               prev_align_one_hot_vec: Optional[jnp.array] = None,
                               t_array: Optional[jnp.array] = None,
                               norm_fn = None ):
        ### combine sequence embeddings
        to_concat = []
        if self.use_anc_emb and (anc_emb is not None):
            to_concat.append( anc_emb )
            
        if self.use_desc_emb and (desc_causal_emb is not None):
            to_concat.append( desc_causal_emb )
        
        # concatenate, mask; embeddings_datamat could be:
        #   (B, L_align, H): (use_anc_emb | use_desc_emb) 
        #   (B, L_align, 2H): (use_anc_emb & use_desc_emb) 
        embeddings_datamat = jnp.concatenate( to_concat, axis = -1 ) # (B, L, n*H)
        del to_concat
        
        # possibly normalize and mask
        embeddings_datamat = norm_fn(embeddings_datamat) # (B, L, n*H)
        embeddings_datamat = self._mask_padding_tokens( x = embeddings_datamat,
                                                        mask = padding_mask ) # (B, L, n*H)
            
        
        ### possibly concat other things (outside of normalization)
        to_concat = [embeddings_datamat]
        
        if self.use_prev_align_info and (prev_align_one_hot_vec is not None):
            to_concat.append(prev_align_one_hot_vec) #(B, L, 5)
        
        if self.use_t_per_sample and (t_array is not None):
            B = t_array.shape[0]
            L = embeddings_datamat.shape[1]
            out_shape = ( (B, L, 1) )
            t_array_exp = jnp.broadcast_to( t_array[..., None, None], out_shape ) #(B, L, 1)
            to_concat.append( t_array_exp )
            
        datamat = jnp.concatenate( [] + to_concat, axis=-1 ) #(B, L, H_out)
        
        # mask again
        datamat = self._mask_padding_tokens( x = datamat,
                                             mask = padding_mask )
        
        return datamat
    
        
    def __call__(self,
                 sow_flax_intermeds: bool,
                 anc_emb: Optional[jnp.array] = None,
                 desc_causal_emb: Optional[jnp.array] = None,
                 prev_align_one_hot_vec: Optional[jnp.array] = None,
                 t_array: Optional[jnp.array] = None,
                 padding_mask: Optional[jnp.array] = None,
                 *args,
                 **kwargs):
        """
        B: batch size
        L_align: length of alignment
        
        Arguments
        ----------
        sow_flax_intermeds : bool

        anc_emb : ArrayLike, (B, L, H) or None
        
        desc_causal_emb : ArrayLike, (B, L, H) or None
        
        prev_align_one_hot_vec : ArrayLike, (B, L, 5) or None
        
        t_array : ArrayLike, (B,) or None
            > note: this is ONE unique time per branch length!

        padding_mask : ArrayLike, (B, L) or None
        
        Returns
        --------
        datamat : ArrayLike, (B, L_align, H_out)
            concatenated/masked data
            
        """
        datamat = self._concatenate_and_norm( padding_mask = padding_mask,
                                              anc_emb = anc_emb, 
                                              desc_causal_emb = desc_causal_emb,
                                              prev_align_one_hot_vec = prev_align_one_hot_vec,
                                              t_array = t_array,
                                              norm_fn = lambda x: x if not self.normalize_seq_embeddings_before_block else self.norm )
        
        # possibly sow values
        self.maybe_sow(sow_flax_intermeds = sow_flax_intermeds,
                       vals = datamat,
                       label = f'{self.name}/concatenated_embeddings_features',
                       include_min_max = True,
                       include_perc_zeros = True)
        
        return datamat
        

class FeedforwardPostproc(SelectMask):
    """
    apply this blocks as many times as specified by layer_sizes: 
        [norm -> dense -> activation -> dropout]
    """
    config: dict
    name: str
    
    def setup(self):
        """
        B = batch size
        L_align = alignment length
        A = alphabet size
        
        """
        ### read config
        # required
        self.layer_sizes = self.config['layer_sizes']
        self.use_anc_emb = self.config['use_anc_emb']
        self.use_desc_emb = self.config['use_desc_emb']
        self.use_prev_align_info = self.config['use_prev_align_info']
        
        # optional
        self.use_t_per_sample = self.config.get('use_t_per_sample', False)
        self.normalize_seq_embeddings_before_block = self.config.get("normalize_seq_embeddings_before_block", True)
        self.dropout = self.config.get("dropout", 0.0)
        use_bias = self.config.get("use_bias", True)
        
        
        ### set up parameterized layers of the MLP
        dense_layers = []
        norm_layers = []
        
        # first layer: possible instance norm and dense
        if self.normalize_seq_embeddings_before_block:
            norm_layers.append( nn.LayerNorm( reduction_axes= -1, 
                                              feature_axes=-1,
                                              name = f'{self.name}/instance norm 0') )
        
        elif not self.normalize_seq_embeddings_before_block:
            norm_layers.append( lambda x: x )
        
        dense_layers.append( nn.Dense(features = self.layer_sizes[0], 
                                      use_bias = use_bias, 
                                      kernel_init = nn.initializers.lecun_normal(),
                                      name=f'{self.name}/feedforward layer 0') )
        
        # subsequent normalization and dense layers
        for i, hid_dim in enumerate(self.layer_sizes[1:]):
            layer_idx = i + 1
            norm_layers.append( nn.LayerNorm( reduction_axes= -1, 
                                              feature_axes=-1,
                                              name = f'{self.name}/instance norm {layer_idx}') )
            dense_layers.append( nn.Dense(features = hid_dim, 
                                          use_bias = use_bias, 
                                          kernel_init = nn.initializers.lecun_normal(),
                                          name=f'{self.name}/feedforward layer {layer_idx}') )
            
        self.dense_layers = dense_layers
        self.norm_layers = norm_layers
        self.act= nn.silu 
    
    @nn.compact
    def __call__(self, 
                 sow_flax_intermeds: bool,
                 training: bool,
                 anc_emb: Optional[jnp.array] = None,
                 desc_causal_emb: Optional[jnp.array] = None,
                 prev_align_one_hot_vec: Optional[jnp.array] = None,
                 t_array: Optional[jnp.array] = None,
                 padding_mask: Optional[jnp.array] = None,
                 *args,
                 **kwargs):
        """
        B: batch size
        L_align: length of alignment
        H_in, H_out: size of embedding dimension in/out of this block
        
        Arguments
        ----------
        sow_flax_intermeds : bool

        anc_emb : ArrayLike, (B, L, H) or None
        
        desc_causal_emb : ArrayLike, (B, L, H) or None
        
        prev_align_one_hot_vec : ArrayLike, (B, L, 5) or None
        
        t_array : ArrayLike, (B,) or None
            > note: this is ONE unique time per branch length!

        padding_mask : ArrayLike, (B, L) or None
        
        Returns
        --------
        datamat : ArrayLike, (B, L_align, H_out)
            concatenated and post-processed data
            > n=1, if only using ancestor embedding OR descendant embedding
            > n=2, if using both embeddings
        """
        ### First block
        # 1.) select (potentially concat+norm) the ancestor and descendant embeddings
        # padding tokens are masked in this function
        datamat = self._concatenate_and_norm( padding_mask = padding_mask,
                                              anc_emb = anc_emb, 
                                              desc_causal_emb = desc_causal_emb,
                                              prev_align_one_hot_vec = prev_align_one_hot_vec,
                                              t_array = t_array,
                                              norm_fn = None if not self.normalize_seq_embeddings_before_block else self.norm_layers[0] ) #(B, L, H_out)
        
        # 2.) dense layer, mask
        datamat = self.dense_layers[0](datamat) #(B, L, hid_dim[0] )
        datamat = self._mask_padding_tokens( x=datamat,
                                             mask=padding_mask ) #(B, L, hid_dim[0] )
        
        # 3.) activation
        datamat = self.act(datamat) #(B, L, hid_dim[0] )
        
        # 4.) dropout
        datamat = nn.Dropout(rate = self.dropout)(datamat,
                                            deterministic = not training)  #(B, L, layer_sizes[0] )
         
        # record results from first block
        self.maybe_sow(sow_flax_intermeds = sow_flax_intermeds,
                       vals = datamat,
                       label = f'{self.name}/final_feedforward_layer_0/after_dropout',
                       include_min_max = True,
                       include_perc_zeros = False)
        
        # mask tokens after block
        datamat = self._mask_padding_tokens( x=datamat,
                                             mask=padding_mask ) #(B, L, layer_sizes[0] )
        
        
        ### Subsequent blocks: norm -> dense -> activation -> dropout
        for i in range( len(self.layer_sizes[1:]) ):
            layer_idx = i + 1
             
            # 1.) instance norm, mask
            datamat = self.norm_layers[layer_idx](datamat) #(B, L, layer_sizes[layer_idx-1])
            datamat = self._mask_padding_tokens( x=datamat,
                                                 mask=padding_mask ) #(B, L, layer_sizes[layer_idx-1])
            
            # 2.) dense, mask
            datamat = self.dense_layers[layer_idx](datamat) #(B, L, layer_sizes[layer_idx-1])
            datamat = self._mask_padding_tokens( x=datamat,
                                                 mask=padding_mask ) #(B, L, layer_sizes[layer_idx-1])
            
            # 3.) activation
            datamat = self.act(datamat) #(B, L, layer_sizes[layer_idx-1])
            
            # 4.) dropout
            datamat = nn.Dropout(rate = self.dropout)(datamat,
                                                deterministic = not training) #(B, L, layer_sizes[layer_idx-1])
            
            # record results from first block
            self.maybe_sow(sow_flax_intermeds = sow_flax_intermeds,
                           vals = datamat,
                           label = (f'{self.name}/'+
                                    f'final_feedforward_layer_{layer_idx}/'+
                                    f'after_dropout'),
                           include_min_max = True,
                           include_perc_zeros = False)
        
            # mask out padding tokens after block
            datamat = self._mask_padding_tokens( x=datamat,
                                                 mask=padding_mask ) #(B, L, layer_sizes[-1])
            
        return datamat  #(B, L, layer_sizes[-1])

    