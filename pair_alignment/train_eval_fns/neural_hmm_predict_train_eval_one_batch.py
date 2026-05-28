#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 00:50:39 2023


ABOUT:
======
train and eval functions for one batch of data


THIS USES THE WHOLE SEQUENCE LENGTH

"""
# regular python
import numpy as np
from collections.abc import MutableMapping
import pickle
import math
from functools import partial

# flax, jax, and optax
import jax
import jax.numpy as jnp
from jax import Array
from jax import config
from flax import linen as nn
import optax



###############################################################################
### HELPERS    ################################################################
###############################################################################
def _one_hot_pad_with_zeros( mat: jnp.array,
                             num_classes: int,
                             axis: int = -1 ):
    """
    assumes zero is the padding fill!
    """
    pad_mask = ( mat != 0 ) 
    mat_raw_enc = nn.one_hot( x=mat, num_classes=num_classes, axis=axis ) 
    mat_oh_extra_class = mat_raw_enc * pad_mask[...,None] 
    mat_oh = mat_oh_extra_class[...,1:] 
    return mat_oh

def _preproc( unaligned_seqs: jnp.array, 
              aligned_mats: jnp.array ):
    ### unpack features
    # unaligned sequences used in __call__
    anc_seqs = unaligned_seqs[...,0] # (B, L_seq)
    desc_seqs = unaligned_seqs[...,1] # (B, L_seq)
    
    # split into prefixes and suffixes, to avoid confusion
    # prefixes: <s> A  B  C    the "a" in P(b | a, X, Y_{...j})
    #            |  |  |  |
    #            v  v  v  v
    # suffixes:  A  B  C <e>    the "b" in P(b | a, X, Y_{...j})
    aligned_mats_prefixes = aligned_mats[:,:-1,:]  # (B, L_align-1, 5)
    aligned_mats_suffixes = aligned_mats[:,1:,:] # (B, L_align-1, 5)
    
    # precomputed alignment indices
    # don't include last token, since it's not used to predict any valid input
    align_idxes = aligned_mats_prefixes[...,-2:] # (B, L_align-1, 2)
    
    
    ### prepare true_out
    # need three things: gapped ancestor, gapped descendant, and 
    # state transitions (a -> b transitions)
    gapped_anc_desc = aligned_mats_suffixes[...,:2] #(B, L_align-1, 2)
    from_states = aligned_mats_prefixes[...,2] #(B, L_align-1)
    to_states = aligned_mats_suffixes[...,2] #(B, L_align-1)
    true_out = jnp.concatenate( [ gapped_anc_desc,
                                  from_states[..., None],
                                  to_states[..., None] ],
                                axis = -1 ) #(B, L_align-1, 4)
    
    out = {'anc_seqs': anc_seqs,
           'desc_seqs': desc_seqs,
           'align_idxes': align_idxes,
           'from_states': from_states,
           'true_out': true_out}
    
    return out
             


###############################################################################
### TRAIN ON ONE BATCH    #####################################################
###############################################################################
def train_one_batch(batch, 
                    training_rngkey,
                    all_trainstates,
                    all_model_instances,
                    max_seq_len,
                    max_align_len,
                    interms_for_tboard, 
                    t_array_for_all_samples,  
                    concat_fn, 
                    record_interms_this_batch: bool = False,
                    norm_loss_by_for_reporting: str='desc_len',
                    update_grads: bool = True,
                    gap_idx: int = 43,
                    seq_padding_idx: int = 0,
                    align_idx_padding: int = -9,
                    *args,
                    **kwargs):
    """
    Jit-able function to apply the model to one batch of samples, evaluate loss
    and collect gradients, then update model parameters
    
    Arguments
    ----------
    regular inputs:
        > batch: batch from a pytorch dataloader
        > training_rngkey: the rng key
        > all_trainstates: the models + parameters

    static inputs, trigger different jit-compilations
        > max_seq_len: max length of unaligned seqs matrix (used to control 
                       number of jit-compiled versions of this function)
        > max_align_len: max length of alignment matrix (used to control 
                         number of jit-compiled versions of this function) 

    static inputs, provided by partial    
        > all_model_instances: the object instances; contain some useful 
          functions that, unfortunately, cannot be called with 
          trainstate.apply_fn
        > norm_loss_by_for_reporting: when reporting loss, normalize by some
                    sequence length; default is desc_len
        > interms_for_tboard: decide whether or not to output intermediate 
                             histograms and scalars
        > update_grads: only turn off when debugging
        > concat_fn: what function to use to concatenate embedded seq inputs
        > gap_idx, seq_padding_idx, align_idx_padding: default tokens and indices

    static inputs, specific to neural hmm:
        > t_array_for_all_samples: one time array for all samples, if 
          applicable; either a jax array of size (T,) or None
    
    Returns
    --------
    out_dict : dict
        dictionary of metrics and outputs  
    
    updated_trainstates : flax trainstate objects
        updated with new parameters
        
    """
    # which times to use 
    if t_array_for_all_samples is None:
        times_for_matrices = batch[2] #(B,)
    
    elif t_array_for_all_samples is not None:
        times_for_matrices = t_array_for_all_samples #(T,)
        
        
    #########################
    ### UNPACK FLAGS, FNS   #
    #########################
    # booleans for determining which sowed outputs to writed during training
    sow_outputs = interms_for_tboard['sow_outputs'] & record_interms_this_batch
    save_gradients = interms_for_tboard['gradients'] & record_interms_this_batch
    save_updates = interms_for_tboard['optimizer'] & record_interms_this_batch
    del interms_for_tboard
    
    
    ##############################
    ### UNPACK, CLIP, MAKE RNGS  #
    ##############################
    # unpack
    encoder_trainstate, decoder_trainstate, finalpred_trainstate = all_trainstates
    encoder_instance, decoder_instance, _ = all_model_instances
    del all_trainstates, all_model_instances
    
    # clip to max lengths, split into prefixes and suffixes
    batch_unaligned_seqs = batch[0] #(B, L, 2)
    batch_aligned_mats = batch[1] #(B, L, 5)
    del batch
    
    clipped_unaligned_seqs = batch_unaligned_seqs[:, :max_seq_len, :] #(B, L_seq, 2)
    clipped_aligned_mats = batch_aligned_mats[:, :max_align_len, :] #(B, L_align, 5)
    
    # produce new keys for each network
    all_keys = jax.random.split(training_rngkey, num=4)
    training_rngkey, enc_key, dec_key, finalpred_key = all_keys
    del all_keys
    
    
    ##########################
    ### PREPARE THE INPUTS   #
    ##########################
    # preprocess with helper
    out_dict = _preproc( unaligned_seqs = clipped_unaligned_seqs, 
                         aligned_mats = clipped_aligned_mats )
    anc_seqs = out_dict['anc_seqs']  # (B, L_seq)
    desc_seqs = out_dict['desc_seqs'] # (B, L_seq)
    align_idxes = out_dict['align_idxes'] # (B, L_align-1, 2)
    from_states = out_dict['from_states'] #(B, L_align-1)
    true_out = out_dict['true_out'] #(B, L_align-1, 4)
    del out_dict
    
    # when reporting, normalize the loss by a length (but this is NOT the 
    #   objective function)
    length_for_normalization_for_reporting = (true_out[...,1] != seq_padding_idx).sum(axis=1) #(B, )
    
    if norm_loss_by_for_reporting == 'desc_len':
        num_gaps = (true_out[...,1] == gap_idx).sum(axis=1) #(B, )
        length_for_normalization_for_reporting = length_for_normalization_for_reporting - num_gaps #(B, )
        
        
    ############################################
    ### APPLY MODEL, EVALUATE LOSS AND GRADS   #
    ############################################
    def apply_model(encoder_params, 
                    decoder_params, 
                    finalpred_params):
        ### embed with ancestor encoder
        # anc_embeddings is (B, L_seq-1, H)
        out = encoder_instance.apply_seq_embedder_in_training( seqs = anc_seqs,
                                                               tstate = encoder_trainstate,
                                                               rng_key = enc_key,
                                                               params_for_apply = encoder_params,
                                                               sow_flax_intermeds = sow_outputs )
        
        anc_embeddings, embeddings_aux_dict = out
        del out
        
        
        ### embed with descendant decoder
        # desc_embeddings is (B, L_seq-1, H)
        out = decoder_instance.apply_seq_embedder_in_training( seqs = desc_seqs,
                                                               tstate = decoder_trainstate,
                                                               rng_key = dec_key,
                                                               params_for_apply = decoder_params,
                                                               sow_flax_intermeds = sow_outputs )
        desc_embeddings, to_add = out
        del out
        
        # at minimum, embeddings_aux_dict contains:
        #     - anc_aux
        #     - anc_layer_metrics
        #     - desc_layer_metrics
        
        # could also contain
        #     - anc_attn_weights (for transformers)
        #     - desc_attn_weights (for transformers)
        #     - anc_carry (for LSTMs)
        #     - desc_carry (for LSTMs)
        embeddings_aux_dict = {**embeddings_aux_dict, **to_add}
        del to_add
        
        
        ### extract embeddings
        out = concat_fn(anc_encoded = anc_embeddings, 
                        desc_encoded = desc_embeddings,
                        idx_lst = align_idxes,
                        seq_padding_idx = seq_padding_idx,
                        align_idx_padding = align_idx_padding)
        datamat_lst, padding_mask = out
        del out
        
        # add previous alignment state, one-hot encoded
        from_states_one_hot = _one_hot_pad_with_zeros( mat=from_states,
                                                       num_classes=6,
                                                       axis=-1 ) #(B, L_align-1, 5)
        datamat_lst.append( from_states_one_hot ) 
        del from_states_one_hot
        
        
        ### forward pass through prediction head, to get scoring matrices
        mut = ['sowed_intermeds'] if sow_outputs else []
        out = finalpred_trainstate.apply_fn(variables = finalpred_params,
                                            datamat_lst = datamat_lst,
                                            t_array = times_for_matrices,
                                            padding_mask = padding_mask,
                                            training = True,
                                            sow_flax_intermeds = sow_outputs,
                                            mutable=mut,
                                            rngs={'dropout': finalpred_key})
        
        # forward_pass_scoring_matrices has the keys:
        # logprob_emit_match: (T,B,L_align-1,A,A) or (B,L_align-1,A,A)
        # logprob_emit_indel: (B,L_align-1,A)
        # logprob_transits: (T,B,L_align-1,S,S) or (B,L_align-1,S,S)
        # corr: (T,B) or (B,)
        # subs_model_params: a dictionary of things (see model code)
        # indel_model_params: a dictionary of things (see model code)
        forward_pass_scoring_matrices, pred_layer_metrics = out
        
        if sow_outputs:
            pred_layer_metrics = pred_layer_metrics['sowed_intermeds']
            
        del out, mut
        
        
        ### evaluate loglike of true alignments
        # this calculates sum of logprob over length of alignment; it could be
        # replaced with a version of the function that's scanned down the
        # length of alignments
        #
        # loss dict has:
        # logprob_perSamp_perTime; (T,B) or (B,)
        # total_seen_toks; float
        # if returning intermediates: tr, e; (T, B, L) or (B, L)
        loss_dict = finalpred_trainstate.apply_fn( variables = finalpred_params,
                                                   logprob_emit_match = forward_pass_scoring_matrices['logprob_emit_match'], 
                                                   logprob_emit_indel = forward_pass_scoring_matrices['logprob_emit_indel'],
                                                   logprob_transits = forward_pass_scoring_matrices['logprob_transits'],
                                                   corr = forward_pass_scoring_matrices['corr'],
                                                   true_out = true_out,
                                                   padding_idx = seq_padding_idx,
                                                   return_result_before_sum = False,
                                                   method = 'neg_loglike_in_scan_fn') 
        
        # sums_dict has the keys (all are float values):
        # 
        # total_seen_toks
        # rate_multiplier_sum
        # tkf_lambda_sum
        # tkf_mu_sum
        # tkf92_frag_size_sum (0, if using TKF91)
        sums_dict = finalpred_trainstate.apply_fn( variables = finalpred_params,
                                                   true_out = true_out,
                                                   subs_model_params = forward_pass_scoring_matrices['subs_model_params'],
                                                   indel_model_params = forward_pass_scoring_matrices['indel_model_params'],
                                                   method = 'accumulate_parameter_sums_in_scan_fn') 
        
        ### final loss calculation; possibly logsumexp across timepoints
        # loss
        out = finalpred_trainstate.apply_fn( variables = finalpred_params,
                                             loss_dict = loss_dict,
                                             length_for_normalization_for_reporting = length_for_normalization_for_reporting,
                                             t_array = times_for_matrices,
                                             padding_idx = seq_padding_idx,
                                             method = 'evaluate_loss_after_scan' )
        loss, aux_dict = out
        del out, loss_dict  
        
        # regularize with mean of evolutionary parameters
        out = finalpred_trainstate.apply_fn( variables = finalpred_params,
                                             raw_loss = loss,
                                             sums_dict = sums_dict,
                                             method = 'regularize_loss_using_mean_evoparams' )
        loss, all_means = out
        del out
        
        aux_dict = {**aux_dict, **all_means}
        
        
        ### return EVERYTHING
        aux_dict['embeddings_aux_dict'] = embeddings_aux_dict
        aux_dict['pred_layer_metrics'] = pred_layer_metrics
        return (loss, aux_dict)
    
    
    ### set up the grad functions, based on above loss function
    grad_fn = jax.value_and_grad(apply_model, 
                                 argnums=[0,1,2], 
                                 has_aux=True)
    
    (batch_loss, aux_dict), all_grads = grad_fn(encoder_trainstate.params, 
                                                decoder_trainstate.params, 
                                                finalpred_trainstate.params)
    
    enc_gradient, dec_gradient, finalpred_gradient = all_grads
    del all_grads
    
    
    ### evaluate metrics BEFORE updating parameters
    # batch_ave_perpl = finalpred_trainstate.apply_fn( variables = finalpred_trainstate.params,
    #                                                  loss_fn_dict = aux_dict,
    #                                                  method = 'get_perplexity_per_sample' ).mean()
    ece = jnp.exp( aux_dict['neg_logP_length_normed'].mean() )
    
    
    ###########################
    ### RECORD UPDATES MADE   #
    ###########################
    if update_grads:
        ### get new updates and optimizer states
        encoder_updates, new_encoder_opt_state = encoder_trainstate.tx.update(enc_gradient, 
                                                                              encoder_trainstate.opt_state, 
                                                                              encoder_trainstate.params)
        
        decoder_updates, new_decoder_opt_state  = decoder_trainstate.tx.update(dec_gradient, 
                                                                               decoder_trainstate.opt_state, 
                                                                               decoder_trainstate.params)
        
        finalpred_updates, new_finalpred_opt_state  = finalpred_trainstate.tx.update(finalpred_gradient, 
                                                                                     finalpred_trainstate.opt_state, 
                                                                                     finalpred_trainstate.params)
        
        ### apply updates to parameter, trainstate object
        # wrapper for encoder, in case I ever use batch norm
        new_encoder_trainstate = encoder_instance.update_seq_embedder_tstate(tstate = encoder_trainstate,
                                                                             new_opt_state = new_encoder_opt_state,
                                                                             optim_updates = encoder_updates)
        del new_encoder_opt_state
        
        # standard update recipe for decoder
        new_decoder_params = optax.apply_updates(decoder_trainstate.params, 
                                                 decoder_updates)
        new_decoder_trainstate = decoder_trainstate.replace(params = new_decoder_params,
                                                            opt_state = new_decoder_opt_state)
        del new_decoder_opt_state
        
        # standard update recipe for prediction head
        new_finalpred_params = optax.apply_updates(finalpred_trainstate.params, 
                                                   finalpred_updates)
        new_finalpred_trainstate = finalpred_trainstate.replace(params = new_finalpred_params,
                                                                opt_state = new_finalpred_opt_state)
        del new_finalpred_opt_state
    
    elif not update_grads:
        new_encoder_trainstate = encoder_trainstate
        new_decoder_trainstate = decoder_trainstate
        new_finalpred_trainstate = finalpred_trainstate
    
    
    ###############
    ### OUTPUTS   #
    ###############
    # repackage the new trainstates
    updated_trainstates = (new_encoder_trainstate, 
                           new_decoder_trainstate, 
                           new_finalpred_trainstate)
    
    
    ### always returned
    out_dict = {'anc_aux': aux_dict['embeddings_aux_dict']['anc_aux'],
                'sum_neg_logP': aux_dict['sum_neg_logP'],
                'neg_logP_length_normed': aux_dict['neg_logP_length_normed'],
                'batch_loss': batch_loss,
                'batch_ece': ece
                }
    
    ### controlled by boolean flag
    def save_to_out_dict(value_to_save, flag, varname_to_write):
        if flag:
            out_dict[varname_to_write] = value_to_save
            
    # intermediate values captured during training
    save_to_out_dict(value_to_save = aux_dict['embeddings_aux_dict']['anc_layer_intermediates/sowed_intermeds'], 
                     flag = sow_outputs, 
                     varname_to_write = 'anc_layer_metrics')
    
    save_to_out_dict(value_to_save = aux_dict['embeddings_aux_dict']['desc_layer_intermediates/sowed_intermeds'], 
                     flag = sow_outputs, 
                     varname_to_write = 'desc_layer_metrics')
    
    save_to_out_dict(value_to_save = aux_dict['pred_layer_metrics'], 
                     flag = sow_outputs, 
                     varname_to_write = 'pred_layer_metrics')
    
    # gradients
    for (varname, grad) in [('enc_gradient', enc_gradient),
                            ('dec_gradient', dec_gradient),
                            ('finalpred_gradient', finalpred_gradient)]:
        save_to_out_dict(value_to_save = grad,
                         flag = save_gradients,
                         varname_to_write = varname)
    
    # updates
    if update_grads:
        for (varname, grad) in [('encoder_updates', encoder_updates),
                                ('decoder_updates', decoder_updates),
                                ('finalpred_updates', finalpred_updates)]:
            save_to_out_dict(value_to_save = grad,
                             flag = save_updates,
                             varname_to_write = varname)
    

    # always returned from out_dict:
    #     - anc_aux (structure varies depending on model)
    #     - batch_loss; float
    #     - batch_ave_perpl; float
    #     - sum_neg_logP (B,); the loss per sample
    #     - neg_logP_length_normed (B,); the loss per sample, normalized by seq length
    
    # returned if flag active:
    #     - anc_layer_metrics
    #     - desc_layer_metrics
    #     - pred_layer_metrics
    #     - enc_gradient
    #     - dec_gradient
    #     - finalpred_gradient 
    #     - encoder_updates
    #     - decoder_updates
    #     - finalpred_updates

    return (out_dict, updated_trainstates)



###############################################################################
### EVAL ON ONE BATCH    ######################################################
###############################################################################
def eval_one_batch(batch, 
                    all_trainstates,
                    all_model_instances,
                    max_seq_len,
                    max_align_len,
                    interms_for_tboard, 
                    t_array_for_all_samples,  
                    concat_fn, 
                    norm_loss_by_for_reporting: str='desc_len',
                    gap_idx: int = 43,
                    seq_padding_idx: int = 0,
                    align_idx_padding: int = -9,
                    extra_args_for_eval: dict = dict(),
                    *args,
                    **kwargs):
    """
    Jit-able function to apply the model to one batch of samples, evaluate loss
    
    regular inputs:
        > batch: batch from a pytorch dataloader
        > all_trainstates: the models + parameters
    
    static inputs, trigger different jit-compilations:
        > max_seq_len: max length of unaligned seqs matrix (used to control 
                       number of jit-compiled versions of this function)
        > max_align_len: max length of alignment matrix (used to control 
                         number of jit-compiled versions of this function) 
    
    static inputs, provided by partial:
        > all_model_instances: the object instances; contain some useful 
          functions that, unfortunately, cannot be called with 
          trainstate.apply_fn
        > norm_loss_by_for_reporting: when reporting loss, normalize by some
                    sequence length; default is desc_len
        > interms_for_tboard: decide whether or not to output intermediate 
                             histograms and scalars
        > concat_fn: what function to use to concatenate embedded seq inputs
        > extra_args_for_eval: extra inputs for custom eval functions

    static inputs, specific to neural hmm:
        > t_array_for_all_samples: one time array for all samples, if 
          applicable; either a jax array of size (T,) or None
    
    outputs:
        > metrics_outputs: dictionary of metrics and outputs 
    """
    # which times to use 
    if t_array_for_all_samples is None:
        times_for_matrices = batch[2] #(B,)
    
    elif t_array_for_all_samples is not None:
        times_for_matrices = t_array_for_all_samples #(T,)
    
    #########################
    ### UNPACK FLAGS, FNS   #
    #########################
    # booleans for determining which intermediate arrays to return
    return_embs = interms_for_tboard['embeddings']
    return_forward_pass_outputs = interms_for_tboard['forward_pass_outputs']
    return_attn_weights = interms_for_tboard['attn_weights']
    del interms_for_tboard
    
    
    ####################
    ### UNPACK, CLIP   #
    ####################
    # unpack
    encoder_trainstate, decoder_trainstate, finalpred_trainstate = all_trainstates
    encoder_instance, decoder_instance, _ = all_model_instances
    del all_trainstates, all_model_instances
    
    # clip to max lengths, split into prefixes and suffixes
    batch_unaligned_seqs = batch[0] #(B, L, 2)
    batch_aligned_mats = batch[1] #(B, L, 5)
    del batch
    
    clipped_unaligned_seqs = batch_unaligned_seqs[:, :max_seq_len, :] #(B, L_seq, 2)
    clipped_aligned_mats = batch_aligned_mats[:, :max_align_len, :] #(B, L_align, 5)
    
    
    ##########################
    ### PREPARE THE INPUTS   #
    ##########################
    # preprocess with helper
    out_dict = _preproc( unaligned_seqs = clipped_unaligned_seqs, 
                         aligned_mats = clipped_aligned_mats )
    anc_seqs = out_dict['anc_seqs']
    desc_seqs = out_dict['desc_seqs']
    align_idxes = out_dict['align_idxes']
    from_states = out_dict['from_states']
    true_out = out_dict['true_out']
    del out_dict
    
    # when reporting, normalize the loss by a length (but this is NOT the 
    #   objective function)
    length_for_normalization_for_reporting = (true_out[...,1] != seq_padding_idx).sum(axis=1)
    
    if norm_loss_by_for_reporting == 'desc_len':
        num_gaps = (true_out[...,1] == gap_idx).sum(axis=1)
        length_for_normalization_for_reporting = length_for_normalization_for_reporting - num_gaps

    
    #######################
    ### Apply the model   #
    #######################
    ### embed with ancestor encoder
    # anc_embeddings is (B, L_seq-1, H)
    out = encoder_instance.apply_seq_embedder_in_eval( seqs = anc_seqs,
                                                       tstate = encoder_trainstate,
                                                       sow_flax_intermeds = False,
                                                       extra_args_for_eval = extra_args_for_eval)
    anc_embeddings, embeddings_aux_dict = out
    del out
    
    
    ### embed with descendant decoder
    # desc_embeddings is (B, L_seq-1, H)
    out = decoder_instance.apply_seq_embedder_in_eval( seqs = desc_seqs,
                                                       tstate = decoder_trainstate,
                                                       sow_flax_intermeds = False,
                                                       extra_args_for_eval = extra_args_for_eval )
    desc_embeddings, to_add = out
    del out
    
    # at minimum, embeddings_aux_dict contains:
    #     - anc_aux
    #     - anc_layer_metrics
    #     - desc_layer_metrics
    
    # could also contain
    #     - anc_attn_weights (for transformers)
    #     - desc_attn_weights (for transformers)
    #     - anc_carry (for LSTMs)
    #     - desc_carry (for LSTMs)
    embeddings_aux_dict = {**embeddings_aux_dict, **to_add}
    del to_add
    
    
    ### extract embeddings
    out = concat_fn(anc_encoded = anc_embeddings, 
                    desc_encoded = desc_embeddings,
                    idx_lst = align_idxes,
                    seq_padding_idx = seq_padding_idx,
                    align_idx_padding = align_idx_padding)
    datamat_lst, padding_mask = out
    del out
    
    # add previous alignment state, one-hot encoded
    from_states_one_hot = _one_hot_pad_with_zeros( mat=from_states,
                                                   num_classes=6,
                                                   axis=-1 ) #(B, L_align-1, 5)
    datamat_lst.append( from_states_one_hot ) 
    del from_states_one_hot
    
        
    ### forward pass through prediction head, to get scoring matrices
    out = finalpred_trainstate.apply_fn( variables = finalpred_trainstate.params,
                                         datamat_lst = datamat_lst,
                                         t_array = times_for_matrices,
                                         padding_mask = padding_mask,
                                         training = False,
                                         sow_flax_intermeds = False,
                                         mutable=[] )
    # forward_pass_scoring_matrices has the keys:
    # logprob_emit_match: (T,B,L_align-1,A,A) or (B,L_align-1,A,A)
    # logprob_emit_indel: (B,L_align-1,A)
    # logprob_transits: (T,B,L_align-1,S,S) or (B,L_align-1,S,S)
    # corr: (T,B) or (B,)
    # subs_model_params: a dictionary of things (see model code)
    #   > most importantly, this contained rate_multiplier!
    # indel_model_params: a dictionary of things (see model code)
    forward_pass_scoring_matrices, pred_layer_metrics = out
    
    
    ### evaluate loglike of true alignments
    # this calculates sum of logprob over length of alignment; it could be
    # replaced with a version of the function that's scanned down the
    # length of alignments
    #
    # loss dict has:
    # logprob_perSamp_perTime; (T,B) or (B,)
    # total_seen_toks; float
    # if returning intermediates: tr, e; (T, B, L) or (B, L)
    loss_dict = finalpred_trainstate.apply_fn( variables = finalpred_trainstate.params,
                                               logprob_emit_match = forward_pass_scoring_matrices['logprob_emit_match'], 
                                               logprob_emit_indel = forward_pass_scoring_matrices['logprob_emit_indel'],
                                               logprob_transits = forward_pass_scoring_matrices['logprob_transits'],
                                               corr = forward_pass_scoring_matrices['corr'],
                                               true_out = true_out,
                                               padding_idx = seq_padding_idx,
                                               return_result_before_sum = False,
                                               method = 'neg_loglike_in_scan_fn') 
    
    # sums_dict has the keys (all are float values):
    # 
    # total_seen_toks
    # rate_multiplier_sum
    # tkf_lambda_sum
    # tkf_mu_sum
    # tkf92_frag_size_sum (0, if using TKF91)
    sums_dict = finalpred_trainstate.apply_fn( variables = finalpred_trainstate.params,
                                               true_out = true_out,
                                               subs_model_params = forward_pass_scoring_matrices['subs_model_params'],
                                               indel_model_params = forward_pass_scoring_matrices['indel_model_params'],
                                               method = 'accumulate_parameter_sums_in_scan_fn') 
     
    ### final loss calculation; possibly logsumexp across timepoints
    # loss
    out = finalpred_trainstate.apply_fn( variables = finalpred_trainstate.params,
                                         loss_dict = loss_dict,
                                         length_for_normalization_for_reporting = length_for_normalization_for_reporting,
                                         t_array = times_for_matrices,
                                         padding_idx = seq_padding_idx,
                                         method = 'evaluate_loss_after_scan' )
    loss, loss_fn_dict = out
    del out, loss_dict  
    
    # regularize with mean of evolutionary parameters
    out = finalpred_trainstate.apply_fn( variables = finalpred_trainstate.params,
                                         raw_loss = loss,
                                         sums_dict = sums_dict,
                                         method = 'regularize_loss_using_mean_evoparams' )
    loss, all_means = out
    del out
    
    loss_fn_dict = {**loss_fn_dict, **all_means}
    
    
    ### evaluate metrics
    perplexity_perSamp = finalpred_trainstate.apply_fn( variables = finalpred_trainstate.params,
                                                        loss_fn_dict = loss_fn_dict,
                                                        method = 'get_perplexity_per_sample' )
    ece = jnp.exp( loss_fn_dict['neg_logP_length_normed'].mean() )
    
    
    ##########################################
    ### COMPILE FINAL DICTIONARY TO RETURN   #
    ##########################################
    ### things that always get returned
    out_dict = {'batch_loss': loss,
                'batch_ave_perpl': perplexity_perSamp.mean(), # float
                'batch_ave_acc': None, #not used here
                'batch_ece': ece,
                'sum_neg_logP': loss_fn_dict['sum_neg_logP'],
                'neg_logP_length_normed': loss_fn_dict['neg_logP_length_normed'],
                'perplexity_perSamp': perplexity_perSamp}
    
    
    ### optional things to add
    # transformer attention weights
    if return_attn_weights & ( 'anc_layer_intermediates/attn_weights' in embeddings_aux_dict.keys() ):
        out_dict['anc_attn_weights'] = embeddings_aux_dict['anc_layer_intermediates/attn_weights']
        
    if return_attn_weights & ( 'desc_layer_intermediates/attn_weights' in embeddings_aux_dict.keys() ):
        out_dict['desc_attn_weights'] = embeddings_aux_dict['desc_layer_intermediates/attn_weights']
    
    # other arrays to return
    def write_optional_outputs(value_to_save, flag, varname_to_write):
        if flag:
            out_dict[varname_to_write] = value_to_save

    write_optional_outputs(value_to_save = anc_embeddings,
                           flag = return_embs,
                           varname_to_write = 'final_ancestor_embeddings')
    
    write_optional_outputs(value_to_save = desc_embeddings,
                           flag = return_embs,
                           varname_to_write = 'final_descendant_embeddings')
    
    # write whatever comes out of forward_pass_outputs
    if return_forward_pass_outputs:
        for key, val in forward_pass_scoring_matrices.items():
            out_dict[f'scormat_{key}'] = val
    
    # always returned from out_dict:
    #     - loss; float
    #     - batch_ave_perpl; float
    #     - batch_ave_acc = None (not used here)
    #     - sum_neg_logP; (B,)
    #     - neg_logP_length_normed; (B,)
    #     - perplexity_perSamp; (B,)
    
    # returned if flag active:
    #     - anc_layer_metrics
    #     - desc_layer_metrics
    #     - pred_layer_metrics
    #     - anc_attn_weights 
    #     - desc_attn_weights 
    #     - final_ancestor_embeddings
    #     - final_descendant_embeddings
    #     - all other outputs in forward_pass_scoring_matrices
    #       > logprob_emit_match
    #       > logprob_emit_indel
    #       > logprob_transits
    #       > subs_model_params
    #       > indel_model_params
    return out_dict
