#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 17:12:00 2023


ABOUT:
=======
extra functions and classes to include in the transformer blocks

"""
# general python
import numpy as np
from functools import partial

# flax n jax
import jax
from jax import lax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen import combine_masks, make_causal_mask
from flax.linen.attention import dot_product_attention_weights

from utils.BaseClasses import ModuleBase


###############################################################################
### FUNCTIONS   ###############################################################
###############################################################################
def expand_padding_mask(padding_mask):
    """
    repeat padding mask to make compatible with attention heads
    
    padding_mask is (B, L)

    shape expansion is: (B,L,1) -> (B,L,L) -> (B,1,L,L)
    
    """
    q_mask = padding_mask[:, :, None]  # (B, L, 1)
    k_mask = padding_mask[:, None, :]  # (B, 1, L)
    combined_mask = jnp.logical_and(q_mask, k_mask)  # (B, L, L)
    out = combined_mask[:,None,...] #(B, 1, L, L)
    return out


### helpers for attention with rotary positional embedding
def create_sinusoidal_positions(num_pos, dim):
    """
    for rotary positional embedding; taken from HuggingFace
    
    https://github.com/huggingface/transformers/blob/ \
        868d36d29ec132deeaaf8571b25b6a1b911d0145/src/ \
            transformers/models/llama/modeling_flax_llama.py#L131
    """
    inv_freq = 1.0 / (10000 ** (np.arange(0, dim, 2) / dim))
    freqs = np.einsum("i , j -> i j", np.arange(num_pos), inv_freq).astype("float32")

    emb = np.concatenate((freqs, freqs), axis=-1)
    out = np.concatenate((np.sin(emb)[:, None, :], np.cos(emb)[:, None, :]), axis=-1)
    return jnp.array(out[:, :, :num_pos])


def rotate_half(tensor):
    """
    for rotary positional embedding; taken from HuggingFace
    
    https://github.com/huggingface/transformers/blob/ \
        868d36d29ec132deeaaf8571b25b6a1b911d0145/src/ \
            transformers/models/llama/modeling_flax_llama.py
            
    Rotates half the hidden dims of the input.
    """
    rotate_half_tensor = jnp.concatenate(
        (-tensor[..., tensor.shape[-1] // 2 :], 
         tensor[..., : tensor.shape[-1] // 2]), axis=-1
    )
    return rotate_half_tensor


def apply_rotary_pos_emb(tensor, sin_pos, cos_pos):
    """
    for rotary positional embedding; taken from HuggingFace
    
    https://github.com/huggingface/transformers/blob/ \
        868d36d29ec132deeaaf8571b25b6a1b911d0145/src/ \
            transformers/models/llama/modeling_flax_llama.py
    """
    return (tensor * cos_pos) + (rotate_half(tensor) * sin_pos)



###############################################################################
### CLASSES   #################################################################
###############################################################################
class PositionalEncoding(ModuleBase):
    """
    implementation of positional encoding from-
        https://github.com/google/flax/blob/main/examples/lm1b/models.py
    
    """
    hidden_dim : int         # Hidden dimensionality of the input.
    max_len : int = 5000  # Maximum length of a sequence to expect.

    def setup(self):
        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = np.zeros((self.max_len, self.hidden_dim))
        position = np.arange(0, self.max_len)[:,None]
        div_term = np.exp(np.arange(0, self.hidden_dim, 2) * (-np.log(10000.0) / self.hidden_dim))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        self.pe = pe[None, :, :]
        
        # output this matrix to load/visualize later
        # with open(f'positional_encoding_mat.npy','wb') as g:
        #     np.save(g, self.pe)
        
    def __call__(self, x):
        # (B, L, H) -> (B, L, H)
        x = x + self.pe[:, :x.shape[1], :]
        return x
    
    
class RotaryEmbedding(ModuleBase):
    """
    for RotaryEmbeddingSelfAttention; taken from HuggingFace
    
    adapted from huggingface, with as few modifications as possible-
        https://github.com/huggingface/transformers/blob/v4.44.0/src/\
            transformers/models/llama/modeling_flax_llama.py
    """
    hidden_dim: int
    num_heads: int
    max_position_embeddings: int=5000
    
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        head_dim = self.hidden_dim // self.num_heads
        self.sincos = create_sinusoidal_positions(self.max_position_embeddings, head_dim) #(max_position_embeddings, 1, H)

    def __call__(self, 
                 key, #(B, L, num_heads, H/num_heads)
                 query, #(B, L, num_heads, H/num_heads) 
                 position_ids):
        sincos = self.sincos[position_ids] #(B, L, 1, H)
        sin_pos, cos_pos = jnp.split(sincos, 2, axis=-1) #(B, L, 1, H/num_heads)

        key = apply_rotary_pos_emb(key, sin_pos, cos_pos) #(B, L, num_heads, H/num_heads)
        query = apply_rotary_pos_emb(query, sin_pos, cos_pos) #(B, L, num_heads, H/num_heads)

        key = jnp.asarray(key, dtype=self.dtype) #(B, L, num_heads, H/num_heads)
        query = jnp.asarray(query, dtype=self.dtype) #(B, L, num_heads, H/num_heads)

        return key, query


class RotaryEmbeddingSelfAttention(ModuleBase):
    """
    attention with rotary positional embedding
    
    adapted from huggingface, with as few modifications as possible-
        https://github.com/huggingface/transformers/blob/v4.44.0/src/\
            transformers/models/llama/modeling_flax_llama.py
    """
    num_heads: int
    hidden_dim: int
    causal: bool
    max_position_embeddings: int = 5000
    use_bias: bool = True
    dropout: float = 0.0
    
    # these aren't in AbsPosEmbeddingSelfAttention, so for now, don't change them
    initializer_range: float=0.02
    dtype: jnp.dtype = jnp.float32
    
    def setup(self):
        self.embed_dim = self.hidden_dim
        self.head_dim = self.embed_dim // self.num_heads
        self.num_key_value_heads = self.num_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads # this is always 1
        self.attention_softmax_in_fp32 = self.dtype is not jnp.float32

        ### original llama model used normal init
        # dense = partial(
        #     nn.Dense,
        #     use_bias=self.use_bias,
        #     dtype=self.dtype,
        #     kernel_init=jax.nn.initializers.normal(self.initializer_range)
        # )
        
        ### more standard initialization is lecun; this is the default for 
        ###   other linear layers
        dense = partial(
            nn.Dense,
            use_bias=self.use_bias,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.lecun_normal()  
            )

        self.q_proj = dense(self.num_heads * self.head_dim)
        self.k_proj = dense(self.num_key_value_heads * self.head_dim)
        self.v_proj = dense(self.num_key_value_heads * self.head_dim)
        self.o_proj = dense(self.embed_dim)
        if (self.head_dim * self.num_heads) != self.embed_dim:
            raise ValueError(
                f"hidden_dim must be divisible by num_heads (got `hidden_dim`: {self.embed_dim}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.rotary_emb = RotaryEmbedding(hidden_dim = self.hidden_dim,
                                          num_heads = self.num_heads,
                                          max_position_embeddings = self.max_position_embeddings,
                                          dtype=self.dtype)
        
    def _sow_metrics_per_head(self,
                              mat, #(B, L, num_heads, H/num_heads)
                              key_prefix: str,
                              include_perc_zeros: bool = False):
        ### calculate metrics per head; note: this will be a VECTOR, not a value
        # always include: L2 norm, mean, variance
        # l2_norm = jnp.linalg.norm(mat, axis=(0, 1, 3), ord=2) #(num_heads, )
        rms = jnp.sqrt(jnp.mean(mat**2, axis=(0,1,3))) #(num_heads, )
        mean = mat.mean(axis=(0, 1, 3)) #(num_heads, )
        variance = mat.var(axis=(0, 1, 3)) #(num_heads, )
        max_vals = mat.max( axis=(0,1,3) ) #(num_heads, )
        min_vals = mat.min( axis=(0,1,3) ) #(num_heads, )
        
        if include_perc_zeros:
            size_per_head = mat.shape[0] * mat.shape[1] * mat.shape[3] #float
            perc_zeros = (mat==0).sum( axis=(0,1,3) ) / size_per_head #(num_heads, )
            
            
        ### start sowing
        self.sow(col = "sowed_intermeds",
                 name = f'{key_prefix}/per_head_RMS',
                 value = rms,
                 reduce_fn = lambda a, b: b)
        
        self.sow(col = "sowed_intermeds",
                 name = f'{key_prefix}/per_head_MEAN',
                 value = mean,
                 reduce_fn = lambda a, b: b)
        
        self.sow(col = "sowed_intermeds",
                 name = f'{key_prefix}/per_head_VAR',
                 value = variance,
                 reduce_fn = lambda a, b: b)
        
        self.sow(col = "sowed_intermeds",
                 name = f'{key_prefix}/per_head_MAX',
                 value = max_vals,
                 reduce_fn = lambda a, b: b)
        
        self.sow(col = "sowed_intermeds",
                 name = f'{key_prefix}/per_head_MIN',
                 value = min_vals,
                 reduce_fn = lambda a, b: b)
        
        if include_perc_zeros:
            self.sow(col = "sowed_intermeds",
                     name = f'{key_prefix}/per_head_PERC_ZEROS',
                     value = perc_zeros,
                     reduce_fn = lambda a, b: b)
    
    def _sow_entropy_per_head(self,
                              attn_weights, #(B, num_heads, L_q, L_k)
                              key_prefix: str):
        # weights are in prob-space
        entropy = -(attn_weights * jnp.log(attn_weights + 1e-9)).sum(axis=3) #(B, num_heads, L_q)
        ave_entropy_per_head = jnp.mean(entropy, axis = (0,2)) #(num_heads,)
        self.sow(col = "sowed_intermeds",
                 name = f'{key_prefix}/per_head_attn_weight_ave_entropy',
                 value = ave_entropy_per_head,
                 reduce_fn = lambda a, b: b)
        
    def _split_heads(self, hidden_states, num_heads):
        return hidden_states.reshape(hidden_states.shape[:2] + (num_heads, self.head_dim))

    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.embed_dim,))

    def __call__(self, 
                 inputs_q, 
                 mask, 
                 deterministic: bool,
                 sow_flax_intermeds: bool,
                 sow_weights: bool):
        """
        sow_flax_intermds is for statistics of interest from intermediate 
          activations
        
        sow_weights is for attention weights only
        """
        #############
        ### setup   #
        #############
        # alias my trace to the arguments from original implementation
        hidden_states = inputs_q #(B, L, H)
        attention_mask = mask  #(B, 1, L, L)
        
        # probably won't use these arguments
        init_cache = False
    
        # need to make position_ids for rotary embeddings
        position_ids = jnp.array( range(0, inputs_q.shape[1]) )[None, :] #(1,L)
        new_shape = (inputs_q.shape[0], 
                     position_ids.shape[1])
        position_ids = jnp.broadcast_to(position_ids, new_shape) #(B,L)
        del new_shape
        
        
        #####################################
        ### project to key, query, values   #
        #####################################
        # project
        query = self.q_proj(hidden_states) #(B, L, H) 
        key = self.k_proj(hidden_states) #(B, L, H)
        value = self.v_proj(hidden_states) #(B, L, H)
        
        # split
        query = self._split_heads(query, self.num_heads) #(B, L, num_heads, H/num_heads)
        key = self._split_heads(key, self.num_key_value_heads) #(B, L, num_heads, H/num_heads)
        value = self._split_heads(value, self.num_key_value_heads) #(B, L, num_heads, H/num_heads)
        
        # possibly sow values
        if sow_flax_intermeds:
            self._sow_metrics_per_head(mat = query,
                           key_prefix = f'{self.name}/query_before_RoPE',
                           include_perc_zeros = False)
            
            self._sow_metrics_per_head(mat = key,
                           key_prefix = f'{self.name}/key_before_RoPE',
                           include_perc_zeros = False)
            

        #############################
        ### add rotary embeddings   #
        #############################
        key, query = self.rotary_emb(key, query, position_ids) #(B, L, num_heads, H/num_heads)
        query_length, key_length = query.shape[1], key.shape[1]
        key = jnp.repeat(key, self.num_key_value_groups, axis=2) #(B, L, num_heads, H/num_heads)
        value = jnp.repeat(value, self.num_key_value_groups, axis=2) #(B, L, num_heads, H/num_heads)
        
        # possibly sow values
        if sow_flax_intermeds:
            self._sow_metrics_per_head(mat = query,
                           key_prefix = f'{self.name}/query_after_RoPE',
                           include_perc_zeros = False)
            
            self._sow_metrics_per_head(mat = key,
                           key_prefix = f'{self.name}/key_after_RoPE',
                           include_perc_zeros = False)
        

        ############################
        ### Handle mask, dropout   #
        ############################
        # (makes masked positions = -inf)
        # specifically, add zero to valid positions, and -inf to invalid positions
        bias_for_attention_layer = lax.select(
            attention_mask > 0,
            jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
            jnp.full(attention_mask.shape, jnp.finfo(self.dtype).min).astype(self.dtype),
        ) #(B, 1, L, L)
        
        # dropout rng
        dropout_rng = None
        if not deterministic and self.dropout > 0.0:
            dropout_rng = self.make_rng("dropout")
            # # for unit testing, explicitly pass an rng key
            # dropout_rng = jax.random.key(0)
            # print('PROVIDING jax.random.key(0) to RotaryEmbeddingSelfAttention')


        ###################################
        ### usual dot product attention   #
        ###################################
        attention_dtype = jnp.float32 if self.attention_softmax_in_fp32 else self.dtype
        attn_weights = dot_product_attention_weights(
            query,
            key,
            bias=bias_for_attention_layer,
            dropout_rng=dropout_rng,
            dropout_rate=self.dropout,
            deterministic=deterministic,
            dtype=attention_dtype,
        ) #(B, num_heads, L, L)

        if self.attention_softmax_in_fp32:
            attn_weights = attn_weights.astype(self.dtype)
        
        # possibly sow average entropy per head
        if sow_flax_intermeds:
            self._sow_entropy_per_head(attn_weights = attn_weights, 
                                       key_prefix = self.name)

        # replicate MultiHeadDotProductAttention sowing behavior
        if sow_weights:
            self.sow(col='intermediates', 
                     name='attention_weights', 
                     value=attn_weights)


        ########################################
        ### apply attention weights to value   #
        ########################################
        attn_output = jnp.einsum("...hqk,...khd->...qhd", 
                                 attn_weights, 
                                 value) #(B, L, num_heads, H/num_heads)
        attn_output = self._merge_heads(attn_output) #(B,L,H)
        attn_output = self.o_proj(attn_output) #(B,L,H)
        
        # possibly sow values
        self.maybe_sow(sow_flax_intermeds = sow_flax_intermeds,
                       vals = attn_output,
                       label = f'{self.name}/attention_output',
                       include_min_max = True,
                       include_perc_zeros = False)

        return attn_output
