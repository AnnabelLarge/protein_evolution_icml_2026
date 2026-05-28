#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 20:45:05 2025

"""
# general python
import os
import shutil
from tqdm import tqdm
from time import process_time
from time import time as wall_clock_time
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None # this is annoying
import pickle
from functools import partial
import platform
import argparse
import json
from datetime import datetime

# jax/flax stuff
import jax
import jax.numpy as jnp
import flax
from flax import linen as nn
import optax

# pytorch imports
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

# custom function/classes imports
from utils.build_optimizer import build_optimizer
from utils.edit_argparse import enforce_valid_defaults
from train_eval_fns.general_training_wrapper.training_wrapper_helpers import (jit_compile_determine_seqlen_bin,
                                                                              jit_compile_determine_alignlen_bin)
from utils.end_training import write_final_eval_results

# specific to training this model
from neural_models.neural_shared.neural_initializer import create_all_tstates 
from utils.edit_argparse import feedforward_fill_with_default_values as fill_with_default_values
from utils.edit_argparse import feedforward_share_top_level_args as share_top_level_args
from train_eval_fns.feedforward_predict_train_eval_one_batch import eval_one_batch
from train_eval_fns.neural_final_eval_wrapper import final_eval_wrapper


def eval_feedforward( args, 
                      dataloader_dict: dict,
                      training_argparse,
                      **kwargs ):
    ###########################################################################
    ### 0: CHECK CONFIG; IMPORT APPROPRIATE MODULES   #########################
    ###########################################################################
    err = (f"{training_argparse.pred_model_type} is not feedforward; "+
           f"using the wrong training script")
    assert training_argparse.pred_model_type == 'feedforward', err
    del err
    
    # model param filenames
    encoder_save_model_filename = f'ANC_ENC_BEST.pkl'
    decoder_save_model_filename = f'DESC_DEC_BEST.pkl'
    finalpred_save_model_filename = f'FINAL_PRED_BEST.pkl'
    all_save_model_filenames = [encoder_save_model_filename, 
                                decoder_save_model_filename,
                                finalpred_save_model_filename]
    
    fill_with_default_values(training_argparse)
    enforce_valid_defaults(training_argparse)
    share_top_level_args(training_argparse)

    # fill in extra
    args.seq_padding_idx = training_argparse.seq_padding_idx
    if not hasattr(args, 'output_attn_weights'):
        args.output_attn_weights = False

    
    ###########################################################################
    ### 1: SETUP   ############################################################
    ###########################################################################
    ### create the eval working directory, if it doesn't exist
    args.logfile_dir = f'{args.eval_wkdir}/logfiles'
    args.logfile_name = f'{args.logfile_dir}/PROGRESS.log'
    args.out_arrs_dir = f'{args.eval_wkdir}/out_arrs'    
    args.model_ckpts_dir = f'{args.eval_wkdir}/model_ckpts'
    if args.eval_wkdir not in os.listdir():
        os.mkdir(args.eval_wkdir)
        os.mkdir(args.logfile_dir)
        os.mkdir(args.out_arrs_dir)
        os.mkdir(args.model_ckpts_dir)
    
    # create a new logfile
    with open(args.logfile_name,'w') as g:
        g.write(f"{datetime.now()}\n\n")
        
        g.write( f'Loading from {args.training_wkdir} to eval new data\n\n' )
        
        g.write( f'Feedforward network to predict alignment-augmented descendant\n' )
        g.write( f'Ancestor sequence embedder (FULL-CONTEXT): {training_argparse.anc_model_type}\n' )
        g.write( f'Descendant sequence embedder (CAUSAL): {training_argparse.desc_model_type}\n' )
        g.write( f'Combine embeddings with: {training_argparse.pred_config["postproc_model_type"]}\n' )
        g.write( f'when reporting, normalizing losses by: descendant length\n\n' )
    
    
    ### extract data from dataloader_dict
    test_dset = dataloader_dict['test_dset']
    test_dl = dataloader_dict['test_dl']
    
    
    ###########################################################################
    ### 2: INITIALIZE MODEL PARTS, OPTIMIZER  #################################
    ###########################################################################
    print('2: model init')
    with open(args.logfile_name,'a') as g:
        g.write('\n')
        g.write(f'2: model init\n')
    
    # need to intialize an optimizer for compatibility when restoring the state, 
    #   but we're not training so this doesn't really matter?
    tx = build_optimizer(training_argparse)
    
    
    ### determine shapes for init
    # unaligned sequences sizes
    global_seq_max_length = test_dset.global_seq_max_length
    largest_seqs = (args.batch_size, global_seq_max_length)
    
    # aligned datasets sizes
    if args.use_scan_fns:
        max_dim1 = args.chunk_length
    
    elif not args.use_scan_fns:
        max_dim1 = test_dset.global_align_max_length - 1
      
    largest_aligns = (args.batch_size, max_dim1)
    del max_dim1
    
    seq_shapes = [largest_seqs, largest_aligns]
    
    # time
    t_per_sample = training_argparse.pred_config['t_per_sample']
    
    if t_per_sample:
        dummy_t_for_each_sample = jnp.empty( (args.batch_size,) )
    
    elif not t_per_sample:
        dummy_t_for_each_sample = None
    
    # batch provided to train/eval functions consist of:
    # 1.) unaligned sequences (B, L_seq, 2)
    # 2.) aligned data matrices (B, L_align, 4)
    # 3.) time per sample; (B,) if present, None otherwise
    # 4, not used.) sample index (B,)
    seq_shapes = [largest_seqs, largest_aligns, dummy_t_for_each_sample]
    
    
    ### initialize functions, determine concat_fn
    out = create_all_tstates( seq_shapes = seq_shapes, 
                              tx = tx, 
                              model_init_rngkey = jax.random.key(0),
                              tabulate_file_loc = args.model_ckpts_dir,
                              anc_model_type = training_argparse.anc_model_type, 
                              desc_model_type = training_argparse.desc_model_type, 
                              pred_model_type = training_argparse.pred_model_type, 
                              anc_enc_config = training_argparse.anc_enc_config, 
                              desc_dec_config = training_argparse.desc_dec_config, 
                              pred_config = training_argparse.pred_config,
                              t_array_for_all_samples = None )  
    blank_trainstates, all_model_instances, concat_fn = out
    del out
    
    # load parameters
    saved_at = f'{args.training_wkdir}/model_ckpts'
    best_trainstates = []
    for i in range(3):
        param_fname = all_save_model_filenames[i]
        blank_tstate = blank_trainstates[i]
        with open(f'{saved_at}/{param_fname}', 'rb') as f:
            state_dict = pickle.load(f)
        ts = flax.serialization.from_state_dict( blank_tstate, state_dict )
        best_trainstates.append(ts)
        del param_fname, blank_tstate, f, state_dict, ts
    
    del i, blank_trainstates, saved_at
    
    
    ### jit-compilations
    # manage sequence lengths
    jitted_determine_seqlen_bin = jit_compile_determine_seqlen_bin(args)
    jitted_determine_alignlen_bin = jit_compile_determine_alignlen_bin(args)
    
    # pass arguments into eval_one_batch; make a parted_eval_fn 
    interms_for_tboard = {k: False for k in training_argparse.interms_for_tboard.keys()}
    interms_for_tboard['embeddings'] = args.embeddings
    interms_for_tboard['forward_pass_outputs'] = args.forward_pass_outputs
    interms_for_tboard['attn_weights'] = args.output_attn_weights
    extra_args_for_eval = {'output_attn_weights': args.output_attn_weights}

    parted_eval_fn = partial( eval_one_batch,
                              all_model_instances = all_model_instances,
                              interms_for_tboard = interms_for_tboard,
                              concat_fn = concat_fn,
                              norm_loss_by_for_reporting = 'desc_len',  
                              extra_args_for_eval = extra_args_for_eval )
    del extra_args_for_eval

    # jit compile this eval function
    eval_fn_jitted = jax.jit( parted_eval_fn, 
                              static_argnames = ['max_seq_len', 'max_align_len'])
    del parted_eval_fn

    ###########################################################################
    ### 3: EVAL   #############################################################
    ###########################################################################
    print(f'BEGIN EVAL')
    # write to logfile
    with open(args.logfile_name,'a') as g:
        g.write('\n')
        g.write(f'BEGIN EVAL\n')
        
    test_summary_stats = final_eval_wrapper(dataloader = test_dl, 
                                             dataset = test_dset, 
                                             best_trainstates = best_trainstates, 
                                             jitted_determine_seqlen_bin = jitted_determine_seqlen_bin,
                                             jitted_determine_alignlen_bin = jitted_determine_alignlen_bin,
                                             eval_fn_jitted = eval_fn_jitted,
                                             out_alph_size = training_argparse.out_alph_size, 
                                             save_arrs = args.save_arrs,
                                             save_per_sample_losses = args.save_per_sample_losses,
                                             interms_for_tboard = interms_for_tboard, 
                                             logfile_dir = args.logfile_dir,
                                             out_arrs_dir = args.out_arrs_dir,
                                             outfile_prefix = f'test-set')

    # save the trainstate again
    for i in range(len(best_trainstates)):
        new_outfile = f'{args.model_ckpts_dir}/{all_save_model_filenames[i]}'
        new_outfile = new_outfile.replace('.pkl',f'_BEST.pkl')
        with open(new_outfile, 'wb') as g:
            model_state_dict = flax.serialization.to_state_dict(best_trainstates[i])
            pickle.dump(model_state_dict, g)   
    
    # output average losses
    write_final_eval_results(args = args, 
                             summary_stats = test_summary_stats,
                             filename = 'AVE-LOSSES.tsv')
    