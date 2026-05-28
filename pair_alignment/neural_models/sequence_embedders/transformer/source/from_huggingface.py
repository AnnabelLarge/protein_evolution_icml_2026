#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 18:22:59 2024


The original self-attention layer from huggingface; only causal
"""
from functools import partial
from typing import Optional, Tuple, Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, make_causal_mask
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax

def create_sinusoidal_positions(num_pos, dim):
    inv_freq = 1.0 / (10000 ** (np.arange(0, dim, 2) / dim))
    freqs = np.einsum("i , j -> i j", np.arange(num_pos), inv_freq).astype("float32")

    emb = np.concatenate((freqs, freqs), axis=-1)
    out = np.concatenate((np.sin(emb)[:, None, :], np.cos(emb)[:, None, :]), axis=-1)
    return jnp.array(out[:, :, :num_pos])


def rotate_half(tensor):
    """Rotates half the hidden dims of the input."""
    rotate_half_tensor = jnp.concatenate(
        (-tensor[..., tensor.shape[-1] // 2 :], tensor[..., : tensor.shape[-1] // 2]), axis=-1
    )
    return rotate_half_tensor


def apply_rotary_pos_emb(tensor, sin_pos, cos_pos):
    return (tensor * cos_pos) + (rotate_half(tensor) * sin_pos)


class FlaxLlamaRotaryEmbedding(nn.Module):
    # config: LlamaConfig
    config: Any
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        head_dim = self.config.hidden_size // self.config.num_attention_heads
        self.sincos = create_sinusoidal_positions(self.config.max_position_embeddings, head_dim)

    def __call__(self, key, query, position_ids):
        sincos = self.sincos[position_ids]
        sin_pos, cos_pos = jnp.split(sincos, 2, axis=-1)

        key = apply_rotary_pos_emb(key, sin_pos, cos_pos)
        query = apply_rotary_pos_emb(query, sin_pos, cos_pos)

        key = jnp.asarray(key, dtype=self.dtype)
        query = jnp.asarray(query, dtype=self.dtype)

        return key, query


class FlaxLlamaAttention(nn.Module):
    # config: LlamaConfig
    config: Any
    dtype: jnp.dtype = jnp.float32
    causal: bool = True
    is_cross_attention: bool = False

    def setup(self):
        config = self.config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.attention_softmax_in_fp32 = self.dtype is not jnp.float32

        dense = partial(
            nn.Dense,
            use_bias=config.attention_bias,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )

        self.q_proj = dense(self.num_heads * self.head_dim)
        self.k_proj = dense(self.num_key_value_heads * self.head_dim)
        self.v_proj = dense(self.num_key_value_heads * self.head_dim)
        self.o_proj = dense(self.embed_dim)
        if (self.head_dim * self.num_heads) != self.embed_dim:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.embed_dim}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.causal_mask = make_causal_mask(jnp.ones((1, config.max_position_embeddings), dtype="bool"), dtype="bool")
        self.rotary_emb = FlaxLlamaRotaryEmbedding(config, dtype=self.dtype)

    def _split_heads(self, hidden_states, num_heads):
        return hidden_states.reshape(hidden_states.shape[:2] + (num_heads, self.head_dim))

    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.embed_dim,))

    @nn.compact
    # Copied from transformers.models.gpt_neo.modeling_flax_gpt_neo.FlaxGPTNeoSelfAttention._concatenate_to_cache
    def _concatenate_to_cache(self, key, value, query, attention_mask):
        """
        This function takes projected key, value states from a single input token and concatenates the states to cached
        states from previous steps. This function is slighly adapted from the official Flax repository:
        https://github.com/google/flax/blob/491ce18759622506588784b4fca0e4bf05f8c8cd/flax/linen/attention.py#L252
        """
        # detect if we're initializing by absence of existing cache data.
        is_initialized = self.has_variable("cache", "cached_key")
        cached_key = self.variable("cache", "cached_key", jnp.zeros, key.shape, key.dtype)
        cached_value = self.variable("cache", "cached_value", jnp.zeros, value.shape, value.dtype)
        cache_index = self.variable("cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32))

        if is_initialized:
            *batch_dims, max_length, num_heads, depth_per_head = cached_key.value.shape
            # update key, value caches with our new 1d spatial slices
            cur_index = cache_index.value
            indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
            key = lax.dynamic_update_slice(cached_key.value, key, indices)
            value = lax.dynamic_update_slice(cached_value.value, value, indices)
            cached_key.value = key
            cached_value.value = value
            num_updated_cache_vectors = query.shape[1]
            cache_index.value = cache_index.value + num_updated_cache_vectors
            # causal mask for cached decoder self-attention: our single query position should only attend to those key positions that have already been generated and cached, not the remaining zero elements.
            pad_mask = jnp.broadcast_to(
                jnp.arange(max_length) < cur_index + num_updated_cache_vectors,
                tuple(batch_dims) + (1, num_updated_cache_vectors, max_length),
            )
            attention_mask = combine_masks(pad_mask, attention_mask)
        return key, value, attention_mask

    def __call__(
        self,
        hidden_states,
        attention_mask,
        position_ids,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
    ):
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query = self._split_heads(query, self.num_heads)
        key = self._split_heads(key, self.num_key_value_heads)
        value = self._split_heads(value, self.num_key_value_heads)

        key, query = self.rotary_emb(key, query, position_ids)

        query_length, key_length = query.shape[1], key.shape[1]

        if self.has_variable("cache", "cached_key"):
            mask_shift = self.variables["cache"]["cache_index"]
            max_decoder_length = self.variables["cache"]["cached_key"].shape[1]
            causal_mask = lax.dynamic_slice(
                self.causal_mask, (0, 0, mask_shift, 0), (1, 1, query_length, max_decoder_length)
            )
        else:
            causal_mask = self.causal_mask[:, :, :query_length, :key_length]

        batch_size = hidden_states.shape[0]
        causal_mask = jnp.broadcast_to(causal_mask, (batch_size,) + causal_mask.shape[1:])

        attention_mask = jnp.broadcast_to(jnp.expand_dims(attention_mask, axis=(-3, -2)), causal_mask.shape)
        attention_mask = combine_masks(attention_mask, causal_mask)

        dropout_rng = None
        if not deterministic and self.config.attention_dropout > 0.0:
            #dropout_rng = self.make_rng("dropout")
            # for unit testing, explicitly pass an rng key
            dropout_rng = jax.random.key(0)
            print(f'PROVIDING jax.random.key(0) to FlaxLlamaAttention')

        # During fast autoregressive decoding, we feed one position at a time,
        # and cache the keys and values step by step.
        if self.has_variable("cache", "cached_key") or init_cache:
            key, value, attention_mask = self._concatenate_to_cache(key, value, query, attention_mask)

        key = jnp.repeat(key, self.num_key_value_groups, axis=2)
        value = jnp.repeat(value, self.num_key_value_groups, axis=2)

        # transform boolean mask into float mask
        attention_bias = lax.select(
            attention_mask > 0,
            jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
            jnp.full(attention_mask.shape, jnp.finfo(self.dtype).min).astype(self.dtype),
        )

        # usual dot product attention
        attention_dtype = jnp.float32 if self.attention_softmax_in_fp32 else self.dtype
        attn_weights = dot_product_attention_weights(
            query,
            key,
            bias=attention_bias,
            dropout_rng=dropout_rng,
            dropout_rate=self.config.attention_dropout,
            deterministic=deterministic,
            dtype=attention_dtype,
        )

        if self.attention_softmax_in_fp32:
            attn_weights = attn_weights.astype(self.dtype)

        attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, value)
        attn_output = self._merge_heads(attn_output)
        attn_output = self.o_proj(attn_output)

        outputs = (attn_output, attn_weights) if output_attentions else (attn_output,)
        return outputs