#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 12:50:04 2025

"""
# regular python
import numpy as np
from collections.abc import MutableMapping
import pickle
import math
from functools import partial
from tqdm import tqdm
import os

# flax, jax, and optax
import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
                            
def train_one_batch(batch, 
                    training_rngkey,
                    all_trainstates,
                    t_array,
                    max_align_len,
                    interms_for_tboard,
                    record_interms_this_batch: bool = False,
                    update_grads: bool = True,
                    *args,
                    **kwargs):
    pairhmm_trainstate = all_trainstates[0]
    sow_outputs = interms_for_tboard['sow_outputs'] & record_interms_this_batch
    save_gradients = interms_for_tboard['gradients'] & record_interms_this_batch
    save_updates = interms_for_tboard['optimizer'] & record_interms_this_batch
    
    
    ### batch has 4 entries:
    ### 0.) unaligned seqs: (B, L, 2)
    ### 1.) aligned matrices: (B, L, 2)
    ### 2.) time (optional): (B,) or None
    ### 3.) dataloader idx (B,)
    batch_aligned_mats = batch[1]
    clipped_aligned_mats = batch_aligned_mats[:, :max_align_len, :]
    new_batch_inputs = [clipped_aligned_mats, batch[2]]
    del batch
    
    def apply_model(pairhmm_params):
        ### new_batch_inputs only has 2 entries:
        ### 0.) aligned matrices: (B, max_align_len, 2)
        ### 1.) time (optional): (B,) or None
        # in training, only evaluate joint loglike i.e. use default __call__
        (loss_NLL, aux_dict), sow_dict = pairhmm_trainstate.apply_fn(variables = pairhmm_params,
                                          batch = new_batch_inputs,
                                          t_array = t_array,
                                          sow_flax_intermeds = sow_outputs,
                                          mutable=['sowed_intermeds'] if sow_outputs else [])
        
        if sow_outputs:
            aux_dict['pred_layer_metrics'] = sow_dict['sowed_intermeds']

        return loss_NLL, aux_dict
    
    grad_fn = jax.value_and_grad(apply_model, has_aux=True)
    (batch_loss_NLL, aux_dict), grad = grad_fn(pairhmm_trainstate.params)
    
    ### only turn this off during debug
    if update_grads:
        updates, new_opt_state = pairhmm_trainstate.tx.update(grad,
                                                           pairhmm_trainstate.opt_state,
                                                           pairhmm_trainstate.params)
        new_params = optax.apply_updates(pairhmm_trainstate.params,
                                         updates)
        new_trainstate = pairhmm_trainstate.replace(params = new_params,
                                                    opt_state = new_opt_state)
    else:
        new_trainstate = pairhmm_trainstate
    
    
    ### Outputs
    joint_neg_logP_length_normed = aux_dict['joint_neg_logP_length_normed']
    joint_ece = jnp.exp( joint_neg_logP_length_normed.mean() )
    
    # main output
    out_dict = {'joint_neg_logP_length_normed': joint_neg_logP_length_normed,
                'joint_neg_logP': aux_dict['joint_neg_logP'],
                'batch_ece': joint_ece,
                'batch_loss': batch_loss_NLL}
    
    
    ### other intermediates
    # sowed values
    if sow_outputs:
        out_dict['pred_layer_metrics'] = aux_dict['pred_layer_metrics']
        
    # gradients
    if save_gradients:
        out_dict['finalpred_gradient'] = grad
    
    # updates
    if save_updates:
        out_dict['finalpred_updates'] = updates
    
    
    return out_dict, [new_trainstate]


def eval_one_batch( batch, 
                    t_array,
                    all_trainstates,
                    pairhmm_instance,
                    max_align_len,
                    interms_for_tboard,
                    return_all_loglikes: bool,
                    *args,
                    **kwargs):
    pairhmm_trainstate = all_trainstates[0]
    
    ### batch has 4 entries:
    ### 0.) unaligned seqs: (B, L, 2)
    ### 1.) aligned matrices: (B, L, 2)
    ### 2.) time (optional): (B,) or None
    ### 3.) dataloader idx (B,)
    batch_aligned_mats = batch[1]
    clipped_aligned_mats = batch_aligned_mats[:, :max_align_len, :]
    new_batch_inputs = [clipped_aligned_mats, batch[2]] #aligned inputs, time per sample (if applicable)
    del batch
    
    if not return_all_loglikes:
        ### new_batch_inputs only has 2 entries:
        ### 0.) aligned matrices: (B, max_align_len, 2)
        ### 1.) time (optional): (B,) or None
        (loss_NLL, aux_dict), _ = pairhmm_trainstate.apply_fn(variables = pairhmm_trainstate.params,
                                          batch = new_batch_inputs,
                                          t_array = t_array,
                                          sow_flax_intermeds = False,
                                          mutable=[])
    
    elif return_all_loglikes:
        aux_dict, _ = pairhmm_trainstate.apply_fn(variables = pairhmm_trainstate.params,
                                          batch = new_batch_inputs,
                                          t_array = t_array,
                                          mutable=[],
                                          method=pairhmm_instance.calculate_all_loglikes)
        
        # specifically use joint prob for loss
        loss_NLL = jnp.mean( aux_dict['joint_neg_logP'] )
        
        
    ### joint probability metrics
    joint_neg_logP_length_normed = aux_dict['joint_neg_logP_length_normed']
    joint_perplexity_perSamp = jnp.exp(joint_neg_logP_length_normed)
    joint_ece = jnp.exp( joint_neg_logP_length_normed.mean() )
    
    out_dict = {'batch_loss': loss_NLL,
                'batch_ave_joint_perpl': jnp.mean(joint_perplexity_perSamp),
                'batch_ece': joint_ece,
                'joint_neg_logP': aux_dict['joint_neg_logP'],
                'joint_neg_logP_length_normed': joint_neg_logP_length_normed,
                'joint_perplexity_perSamp': joint_perplexity_perSamp}
    
    ### other metrics
    if return_all_loglikes:
        # cond
        cond_neg_logP_length_normed = aux_dict['cond_neg_logP_length_normed']
        cond_perplexity_perSamp = jnp.exp(cond_neg_logP_length_normed)
        
        out_dict['cond_neg_logP'] = aux_dict['cond_neg_logP']
        out_dict['cond_neg_logP_length_normed'] = cond_neg_logP_length_normed
        out_dict['cond_perplexity_perSamp'] = cond_perplexity_perSamp
        
        # anc marginal
        anc_neg_logP_length_normed = aux_dict['anc_neg_logP_length_normed']
        anc_perplexity_perSamp = jnp.exp(anc_neg_logP_length_normed)
        
        out_dict['anc_neg_logP'] = aux_dict['anc_neg_logP']
        out_dict['anc_neg_logP_length_normed'] = anc_neg_logP_length_normed
        out_dict['anc_perplexity_perSamp'] = anc_perplexity_perSamp
        
        # desc marginal
        desc_neg_logP_length_normed = aux_dict['desc_neg_logP_length_normed']
        desc_perplexity_perSamp = jnp.exp(desc_neg_logP_length_normed)
        
        out_dict['desc_neg_logP'] = aux_dict['desc_neg_logP']
        out_dict['desc_neg_logP_length_normed'] = desc_neg_logP_length_normed
        out_dict['desc_perplexity_perSamp'] = desc_perplexity_perSamp
    
    return out_dict
    

def final_eval_wrapper(dataloader, 
                       dataset, 
                       eval_fn_jitted,
                       jitted_determine_alignlen_bin,
                       save_per_sample_losses: bool,
                       logfile_dir: str,
                       out_arrs_dir: str, 
                       outfile_prefix: str, 
                       tboard_writer = None,
                       **kwargs):
    """
    WARNING: might have to pull trainstate and instance out of parted+jit 
      function; if so, then make them arguments to this final function
    
    eval_fn_jitted should have already been parted by providing:
        - t_array = given time array
        - pairhmm_trainstate = best trainstate
        - pairhmm_instance = model instance
        - interms_for_tboard = (value from config)
        - return_all_loglike = True
    """
    
    summary_stats = {'sum_joint_loglikes': 0,
                   'joint_ave_loss': 0,
                   'joint_ave_loss_seqlen_normed': 0,
                   'joint_perplexity': 0,
                   
                   'sum_cond_loglikes': 0,
                   'cond_ave_loss': 0,
                   'cond_ave_loss_seqlen_normed': 0,
                   'cond_perplexity': 0,
                   
                   'sum_anc_loglikes': 0,
                   'anc_ave_loss': 0,
                   'anc_ave_loss_seqlen_normed': 0,
                   'anc_perplexity': 0,
                   
                   'sum_desc_loglikes': 0,
                   'desc_ave_loss': 0,
                   'desc_ave_loss_seqlen_normed': 0,
                   'desc_perplexity': 0,
                   }
    
    for batch_idx, batch in tqdm( enumerate(dataloader), total=len(dataloader) ): 
        batch_max_alignlen = jitted_determine_alignlen_bin(batch = batch)
        batch_max_alignlen = batch_max_alignlen.item()
        
        eval_metrics = eval_fn_jitted(batch=batch, 
                                      max_align_len=batch_max_alignlen)
        
        #########################################
        ### start df; record metrics per sample #
        #########################################
        final_loglikes = dataset.retrieve_sample_names(batch[-1])
        
        for prefix in ['joint','cond','anc','desc']:
            final_loglikes[f'{prefix}_logP'] = eval_metrics[f'{prefix}_neg_logP']
            final_loglikes[f'{prefix}_logP/normlength'] = eval_metrics[f'{prefix}_neg_logP_length_normed']
            final_loglikes[f'{prefix}_perplexity'] = eval_metrics[f'{prefix}_perplexity_perSamp']
        
        final_loglikes['dataloader_idx'] = batch[-1]
        num_samples_in_batch = eval_metrics['joint_neg_logP'].shape[0]
        
        # record mean values to buckets
        wf = ( num_samples_in_batch / len(dataset) )
        
        for prefix in ['joint','cond','anc','desc']:
            # loglikelihood of interest; don't weight this one!
            to_add = final_loglikes[f'{prefix}_logP'].sum()
            summary_stats[f'sum_{prefix}_loglikes'] += to_add
            del to_add

            # loglikelihood, averaged over samples
            to_add = final_loglikes[f'{prefix}_logP'].mean() * wf
            summary_stats[f'{prefix}_ave_loss'] += to_add
            del to_add
            
            # loglikelihood normalized by some sequence length, then averaged over samples
            to_add = final_loglikes[f'{prefix}_logP/normlength'].mean() * wf
            summary_stats[f'{prefix}_ave_loss_seqlen_normed'] += to_add
            del to_add
            
            # perplexity
            to_add = final_loglikes[f'{prefix}_perplexity'].mean() * wf
            summary_stats[f'{prefix}_perplexity'] += to_add
            del to_add
            
        # write dataframe
        if save_per_sample_losses:
            final_loglikes.to_csv((f'{logfile_dir}/{outfile_prefix}_pt{batch_idx}_'+
                                  'FINAL-LOGLIKES.tsv'), sep='\t')
    
    
    ######################
    ### POST EVAL LOOP   #
    ######################
    # add ECE for all
    for prefix in ['joint','cond','anc','desc']:
        to_add = jnp.exp( summary_stats[f'{prefix}_ave_loss_seqlen_normed'] )
        summary_stats[f'{prefix}_ece'] = to_add
        del to_add
    
    return summary_stats
