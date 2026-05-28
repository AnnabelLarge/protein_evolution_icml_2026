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
from jax import config
from flax import linen as nn
import optax

from train_eval_fns.general_training_wrapper.training_wrapper_helpers import selective_squeeze
from neural_models.sequence_embedders.concatenation_fns import extract_embs

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
              aligned_mats: jnp.array,
              use_prev_align_info: bool):
    ### unpack features
    # unaligned sequences used in __call__
    anc_seqs = unaligned_seqs[...,0] # (B, L_seq)
    desc_seqs = unaligned_seqs[...,1] # (B, L_seq)
    
    # split into prefixes and suffixes, to avoid confusion
    # prefixes: <s> A  B  C     the "a" in P(b | a, X, Y_{...j})
    #            |  |  |  |
    #            v  v  v  v
    # suffixes:  A  B  C <e>    the "b" in P(b | a, X, Y_{...j})
    aligned_mats_prefixes = aligned_mats[:,:-1,:]  # (B, L_align-1, 4)
    aligned_mats_suffixes = aligned_mats[:,1:,:] # (B, L_align-1, 4)
    
    # precomputed alignment indices
    # don't include last token, since it's not used to predict any valid input
    align_idxes = aligned_mats_prefixes[...,-2:] #(B, L_align-1, 2)
    
    
    ### true_out
    # only need the alignment-augmented descendant; dim0=0
    #   in FullLenDset pre-processing, I already removed <bos> moved all 
    #   output tokens down by one,
    true_out = aligned_mats_suffixes[...,0] #(B, L_align-1)
    
    out = {'anc_seqs': anc_seqs, # (B, L_seq)
            'desc_seqs': desc_seqs, # (B, L_seq)
            'align_idxes': align_idxes, #(B, L_align-1, 2)
            'true_out': true_out} #(B, L_align-1)
    
    # possibly add previous alignment
    if use_prev_align_info:
        prev_state_path = aligned_mats_prefixes[...,1]
        out['prev_state_path'] = prev_state_path #(B, L_align-1)
    
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
                    concat_fn,
                    norm_loss_by_for_reporting: str='desc_len',
                    record_interms_this_batch: bool = False,
                    update_grads: bool = True,
                    gap_idx: int = 43,
                    seq_padding_idx: int = 0,
                    align_idx_padding: int = -9,
                    *args,
                    **kwargs):
    """
    Jit-able function to apply the model to one batch of alignments, evaluate loss
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
    
    Returns
    --------
    out_dict : dict
        dictionary of metrics and outputs  
    
    updated_trainstates : flax trainstate objects
        updated with new parameters
        
    """
    #########################
    ### UNPACK FLAGS, FNS   #
    #########################
    # booleans for determining which sowed outputs to write during training
    sow_outputs = interms_for_tboard['sow_outputs'] & record_interms_this_batch
    save_gradients = interms_for_tboard['gradients'] & record_interms_this_batch
    save_updates = interms_for_tboard['optimizer'] & record_interms_this_batch
    del interms_for_tboard
    
    
    ####################
    ### UNPACK, CLIP   #
    ####################
    # unpack
    encoder_trainstate, decoder_trainstate, finalpred_trainstate = all_trainstates
    encoder_instance, decoder_instance, finalpred_instance = all_model_instances
    del all_model_instances, all_trainstates
    
    # flags for later
    use_prev_align_info = finalpred_instance.config['use_prev_align_info']
    use_t_per_sample = finalpred_instance.config['t_per_sample']
    
    # clip to max lengths, split into prefixes and suffixes
    # batch_unaligned_seqs is (B, L, 2)
    # batch_aligned_mats is (B, L, 4)
    # t_array could be (B,) or None
    batch_unaligned_seqs, batch_aligned_mats, t_array, _ = batch
    del batch
    
    clipped_unaligned_seqs = batch_unaligned_seqs[:, :max_seq_len, :] # (B, L_seq, 2)
    clipped_aligned_mats = batch_aligned_mats[:, :max_align_len, :] # (B, L_align, 4)
    
    # produce new keys for each network
    all_keys = jax.random.split(training_rngkey, num=4)
    training_rngkey, enc_key, dec_key, finalpred_key = all_keys
    del all_keys
    
    
    ##################
    ### PREPROCESS   #
    ##################
    # preprocess with helper
    out_dict = _preproc( unaligned_seqs = clipped_unaligned_seqs, 
                         aligned_mats = clipped_aligned_mats,
                         use_prev_align_info = use_prev_align_info )
    anc_seqs = out_dict['anc_seqs'] # (B, L_seq)
    desc_seqs = out_dict['desc_seqs'] # (B, L_seq) 
    align_idxes = out_dict['align_idxes'] #(B, L_align-1, 2)
    true_out = out_dict['true_out'] #(B, L_align-1)
    prev_state_path = out_dict.get('prev_state_path', None) #(B, L_align-1) or None
    del out_dict
    
    # when reporting, normalize the loss by a length (but this is NOT the 
    #   objective function)
    length_for_normalization_for_reporting = (true_out != seq_padding_idx).sum(axis=1) #(B, )
    
    if norm_loss_by_for_reporting == 'desc_len':
        num_gaps = ( true_out == (gap_idx) ).sum(axis=1) #(B, )
        length_for_normalization_for_reporting = length_for_normalization_for_reporting - num_gaps #(B, )
     
        
    ############################################
    ### APPLY MODEL, EVALUATE LOSS AND GRADS   #
    ############################################
    def apply_model(encoder_params, decoder_params, finalpred_params):
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
        
        # optionally, add previous alignment state, one-hot encoded
        if use_prev_align_info:
            from_states_one_hot = _one_hot_pad_with_zeros( mat=prev_state_path,
                                                           num_classes=6,
                                                           axis=-1 ) #(B, L_align-1, 5)
            datamat_lst.append( from_states_one_hot ) 
            del from_states_one_hot
            
        elif not use_prev_align_info:
            datamat_lst.append( None )
        
        
        ### forward pass through prediction head
        mut = ['sowed_intermeds'] if sow_outputs else []
        out = finalpred_trainstate.apply_fn(variables = finalpred_params,
                                            datamat_lst = datamat_lst,
                                            padding_mask = padding_mask,
                                            t_array = t_array,
                                            training = True,
                                            sow_flax_intermeds = sow_outputs,
                                            mutable=mut,
                                            rngs={'dropout': finalpred_key})

        # final_logits is (B, L_align-1, A_aug-1)
        final_logits, pred_layer_metrics = out
        
        if sow_outputs:
            pred_layer_metrics = pred_layer_metrics['sowed_intermeds']
        
        del out, mut
        
        
        ### evaluate loglike
        # loss_intermeds has the following keys:
        #
        # loss_intermeds['logprob_perSamp'] (B,)
        # loss_intermeds['correct_predictions_perSamp'] (B,)
        # loss_intermeds['valid_positions_perSamp'] (B,)
        # loss_intermeds['cm_perSamp'] (B, A_aug-1, A_aug-1)
        loss_intermeds = finalpred_trainstate.apply_fn( variables = finalpred_params,
                                                       final_logits = final_logits,
                                                       padding_mask = padding_mask,
                                                       true_out = true_out,
                                                       return_result_before_sum = False,
                                                       method = 'neg_loglike_in_scan_fn' ) 
        
        loss, loss_intermeds = finalpred_trainstate.apply_fn( variables = finalpred_params,
                                                       scan_dict = loss_intermeds,
                                                       length_for_normalization_for_reporting = length_for_normalization_for_reporting,
                                                       method = 'evaluate_loss_after_scan')
        
        # loss_intermeds NOW has the following keys:
        #
        # loss_intermeds['sum_neg_logP'] (B,)
        # loss_intermeds['neg_logP_length_normed'] (B,)
        # loss_intermeds['acc_perSamp'] (B,)
        # loss_intermeds['cm_perSamp'] (B,)
        # create aux dictionary
        aux_dict = {'sum_neg_logP': loss_intermeds['sum_neg_logP'],
                    'neg_logP_length_normed': loss_intermeds['neg_logP_length_normed'],
                    'acc_perSamp': loss_intermeds['acc_perSamp'],
                    'cm_perSamp': loss_intermeds['cm_perSamp'],
                    'final_logits': final_logits,
                    'embeddings_aux_dict': embeddings_aux_dict,
                    'pred_layer_metrics': pred_layer_metrics}
        return (loss, aux_dict)
    
    
    ### set up the grad functions, based on above loss function
    grad_fn = jax.value_and_grad(apply_model, argnums=[0,1,2], has_aux=True)
    
    (batch_loss, aux_dict), all_grads = grad_fn(encoder_trainstate.params, 
                                                decoder_trainstate.params, 
                                                finalpred_trainstate.params)
    
    enc_gradient, dec_gradient, finalpred_gradient = all_grads
    del all_grads
    
    # aux_dict contains:
    #
    # aux_dict['sum_neg_logP'] (B,)
    # aux_dict['neg_logP_length_normed'] (B,)
    # aux_dict['acc_perSamp'] (B,)
    # aux_dict['final_logits'] (B, L_align-1, A_aug-1)
    # aux_dict['embeddings_aux_dict'] (B,)
    # aux_dict['pred_layer_metrics'] (B,)
    
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
        
        # standard update for decoder
        new_decoder_params = optax.apply_updates(decoder_trainstate.params, 
                                                 decoder_updates)
        new_decoder_trainstate = decoder_trainstate.replace(params = new_decoder_params,
                                                            opt_state = new_decoder_opt_state)
        del new_decoder_opt_state
        
        # standard update for prediction head
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
    out_dict = {'batch_loss': batch_loss,
                'batch_ave_acc': jnp.mean( aux_dict['acc_perSamp'] ),
                'batch_ece': ece,
                'sum_neg_logP': aux_dict['sum_neg_logP'],
                'neg_logP_length_normed': aux_dict['neg_logP_length_normed'],
                'anc_aux': aux_dict['embeddings_aux_dict']['anc_aux'] }
    
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
    for (varname, grad) in [('encoder_updates', encoder_updates),
                            ('decoder_updates', decoder_updates),
                            ('finalpred_updates', finalpred_updates)]:
        save_to_out_dict(value_to_save = grad,
                               flag = save_updates,
                               varname_to_write = varname)
    

    # always returned from out_dict:
    #     - batch_loss; float
    #     - batch_ave_perpl; float
    #     - batch_ave_acc; float 
    #     - sum_neg_logP (B,); the loss per sample BEFORE normalizing by length
    #     - neg_logP_length_normed (B,); the loss per sample
    #     - anc_aux 
    
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
                   concat_fn,
                   norm_loss_by_for_reporting: str='desc_len',
                   gap_idx: int = 43,
                   seq_padding_idx: int = 0,
                   align_idx_padding: int = -9,
                   extra_args_for_eval: dict = dict(),
                   *args,
                   **kwargs):
    """
    Jit-able function to evaluate a model on a batch of alignments
    
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
        > concat_fn: what function to use to concatenate embedded seq inputs
        > gap_idx, seq_padding_idx, align_idx_padding: default tokens and indices
        > extra_args_for_eval: additional arguments, as needed
    
    Returns
    --------
    out_dict : dict
        dictionary of metrics and outputs   
    """
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
    encoder_instance, decoder_instance, finalpred_instance = all_model_instances
    del all_model_instances, all_trainstates
    
    # flags for later
    use_prev_align_info = finalpred_instance.config['use_prev_align_info']
    use_t_per_sample = finalpred_instance.config['t_per_sample']
    
    # clip to max lengths, split into prefixes and suffixes
    # batch_unaligned_seqs is (B, L, 2)
    # batch_aligned_mats is (B, L, 4)
    # t_array could be (B,) or None
    batch_unaligned_seqs, batch_aligned_mats, t_array, _ = batch
    del batch
    
    clipped_unaligned_seqs = batch_unaligned_seqs[:, :max_seq_len, :] # (B, L_seq, 2)
    clipped_aligned_mats = batch_aligned_mats[:, :max_align_len, :] # (B, L_align, 4)
    
    
    
    ##################
    ### PREPROCESS   #
    ##################
    # preprocess with helper
    out_dict = _preproc( unaligned_seqs = clipped_unaligned_seqs, 
                         aligned_mats = clipped_aligned_mats,
                         use_prev_align_info = use_prev_align_info )
    anc_seqs = out_dict['anc_seqs'] # (B, L_seq)
    desc_seqs = out_dict['desc_seqs'] # (B, L_seq) 
    align_idxes = out_dict['align_idxes'] #(B, L_align-1, 2)
    true_out = out_dict['true_out'] #(B, L_align-1)
    prev_state_path = out_dict.get('prev_state_path', None) #(B, L_align-1) or None
    del out_dict
    
    # when reporting, normalize the loss by a length (but this is NOT the 
    #   objective function)
    length_for_normalization_for_reporting = (true_out != seq_padding_idx).sum(axis=1) #(B, )
    
    if norm_loss_by_for_reporting == 'desc_len':
        num_gaps = ( true_out == (gap_idx) ).sum(axis=1) #(B, )
        length_for_normalization_for_reporting = length_for_normalization_for_reporting - num_gaps #(B, )
    
        
    ##################################
    ### APPLY MODEL, EVALUATE LOSS   #
    ##################################
    ### embed with ancestor encoder
    # anc_embeddings is (B, L_seq-1, H)
    out = encoder_instance.apply_seq_embedder_in_eval(seqs = anc_seqs,
                                                      tstate = encoder_trainstate,
                                                      sow_flax_intermeds = False,
                                                      extra_args_for_eval = extra_args_for_eval)
    
    anc_embeddings, embeddings_aux_dict = out
    del out
    
    
    ### embed with descendant decoder
    # desc_embeddings is (B, L_seq-1, H)
    out = decoder_instance.apply_seq_embedder_in_eval(seqs = desc_seqs,
                                                      tstate = decoder_trainstate,
                                                      sow_flax_intermeds = False,
                                                      extra_args_for_eval = extra_args_for_eval)
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
    
    # optionally, add previous alignment state, one-hot encoded
    if use_prev_align_info:
        from_states_one_hot = _one_hot_pad_with_zeros( mat=prev_state_path,
                                                       num_classes=6,
                                                       axis=-1 ) #(B, L_align-1, 5)
        datamat_lst.append( from_states_one_hot ) 
        del from_states_one_hot
        
    elif not use_prev_align_info:
        datamat_lst.append( None )
    
    
    ### forward pass through prediction head
    out = finalpred_trainstate.apply_fn(variables = finalpred_trainstate.params,
                                        datamat_lst = datamat_lst,
                                        t_array = t_array,
                                        padding_mask = padding_mask,
                                        training = False,
                                        sow_flax_intermeds = False,
                                        mutable=[])

    # final_logits is (B, L_align-1, A_aug-1)
    final_logits, pred_layer_metrics = out
    
    
    ### evaluate loglike
    # loss_intermeds has the following keys:
    #
    # loss_intermeds['logprob_perSamp'] (B,)
    # loss_intermeds['correct_predictions_perSamp'] (B,)
    # loss_intermeds['valid_positions_perSamp'] (B,)
    # loss_intermeds['cm_perSamp'] (B, A_aug-1, A_aug-1)
    loss_intermeds = finalpred_trainstate.apply_fn( variables = finalpred_trainstate.params,
                                                   final_logits = final_logits,
                                                   true_out = true_out,
                                                   padding_mask = padding_mask,
                                                   return_result_before_sum = return_forward_pass_outputs,
                                                   method = 'neg_loglike_in_scan_fn' ) 
    
    # after this line, loss_intermeds has NEW keys and values:
    #
    # loss_intermeds['sum_neg_logP'] (B,)
    # loss_intermeds['neg_logP_length_normed'] (B,)
    # loss_intermeds['acc_perSamp'] (B,)
    # loss_intermeds['cm_perSamp'] (B,)
    loss, loss_intermeds = finalpred_trainstate.apply_fn( variables = finalpred_trainstate.params,
                                                   scan_dict = loss_intermeds,
                                                   length_for_normalization_for_reporting = length_for_normalization_for_reporting,
                                                   method = 'evaluate_loss_after_scan')
    
    # evaluate perplexity 
    perplexity_perSamp = finalpred_trainstate.apply_fn( variables = finalpred_trainstate.params,
                                                     loss_fn_dict = loss_intermeds,
                                                     method = 'get_perplexity_per_sample' )
    ece = jnp.exp( loss_intermeds['neg_logP_length_normed'].mean() )
    
    
    ##########################################
    ### COMPILE FINAL DICTIONARY TO RETURN   #
    ##########################################
    ### things that always get returned
    out_dict = {'batch_loss': loss, # float
                'batch_ave_perpl': perplexity_perSamp.mean(), # float
                'batch_ave_acc': loss_intermeds['acc_perSamp'].mean(), # float
                'batch_ece': ece,
                'sum_neg_logP': loss_intermeds['sum_neg_logP'], #(B,)
                'neg_logP_length_normed': loss_intermeds['neg_logP_length_normed'], #(B,)
                'acc_perSamp': loss_intermeds['acc_perSamp'], #(B,)
                'cm_perSamp': loss_intermeds['cm_perSamp'], #(B, A_aug-1, A_aug-1)
                'perplexity_perSamp': perplexity_perSamp}  #(B,)
    
    
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

    write_optional_outputs(value_to_save = final_logits,
                           flag = return_forward_pass_outputs,
                           varname_to_write = 'scormat_final_logits')
    
    # always returned from out_dict:
    #     - loss; float
    #     - batch_ave_perpl; float
    #     - batch_ave_acc; float
    #     - sum_neg_logP; (B,)
    #     - neg_logP_length_normed; (B,)
    #     - perplexity_perSamp; (B,)
    #     - acc_perSamp; (B,)
    #     - cm_perSamp; (B, A_aug-1, A_aug-1)
        
    # returned if flag active:
    #     - anc_layer_metrics
    #     - desc_layer_metrics
    #     - pred_layer_metrics
    #     - anc_attn_weights 
    #     - desc_attn_weights 
    #     - final_ancestor_embeddings
    #     - final_descendant_embeddings
    #     - scormat_final_logits (i.e. BEFORE log_softmax)
         
    return out_dict
