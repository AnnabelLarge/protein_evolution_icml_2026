#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 20:45:05 2025

"""
# general python
import os
import shutil
import glob
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
from utils.write_config import write_config
from utils.edit_argparse import enforce_valid_defaults
from utils.setup_training_dir import setup_training_dir
from train_eval_fns.general_training_wrapper.training_wrapper_helpers import (timers,
                                                                record_postproc_time_table)
from utils.end_training import write_final_eval_results

# specific to training this model
from neural_models.neural_shared.neural_initializer import create_all_tstates 
from utils.edit_argparse import neural_hmm_fill_with_default_values as fill_with_default_values
from utils.edit_argparse import neural_hmm_share_top_level_args as share_top_level_args
from train_eval_fns.neural_hmm_predict_train_eval_one_batch import ( train_one_batch,
                                                                     eval_one_batch )
from train_eval_fns.neural_final_eval_wrapper import final_eval_wrapper
from train_eval_fns.general_training_wrapper.TrainingWrapper import NeuralTKFTrainingWrapper as TrainingWrapper




def train_neural_hmm(args, dataloader_dict: dict):
    ###########################################################################
    ### 0: CHECK CONFIG; IMPORT APPROPRIATE MODULES   #########################
    ###########################################################################
    err = (f"{args.pred_model_type} is not neural_hmm; "+
           f"using the wrong training script")
    assert args.pred_model_type == 'neural_hmm', err
    del err
    
    ### edit the argparse object in-place
    fill_with_default_values(args)
    enforce_valid_defaults(args)
    share_top_level_args(args)
    
    if not args.update_grads:
        print('DEBUG MODE: DISABLING GRAD UPDATES')
    
    
    ###########################################################################
    ### 1: SETUP   ############################################################
    ###########################################################################
    ### initial setup of misc things
    # setup the working directory (if not done yet) and this run's sub-directory
    setup_training_dir(args)
    
    # initial random key, to carry through execution
    rngkey = jax.random.key(args.rng_seednum)
    
    # setup tensorboard writer
    writer = SummaryWriter(args.tboard_dir)
    
    # create a new logfile
    with open(args.logfile_name,'a') as g:
        if not args.update_grads:
            g.write('DEBUG MODE: DISABLING GRAD UPDATES\n\n')
            
        g.write( f'Neural sequence embedders with Markovian alignment assumption\n' )
        g.write( f'Substitution model: {args.pred_config["subst_model_type"]}\n' )
        g.write( f'Indel model: {args.pred_config["indel_model_type"]}\n' )
        g.write( f'when reporting, normalizing losses by: descendant length\n\n' )
        
        g.write( f'Evolutionary model parameters (global vs local):\n' )
        
        if not args.pred_config['load_all']:
            for key, val in args.pred_config["global_or_local"].items():
                g.write(f'{key}: {val}\n')
                
        g.write('\n')
        
        g.write(f'Ancestor sequence embedder (FULL-CONTEXT): {args.anc_model_type}\n')
        g.write(f'Descendant sequence embedder (CAUSAL): {args.desc_model_type}\n\n')
        
        
    ### save updated config, provide filename for saving model parameters
    encoder_save_model_filename = args.model_ckpts_dir + '/'+ f'ANC_ENC.pkl'
    decoder_save_model_filename = args.model_ckpts_dir + '/'+ f'DESC_DEC.pkl'
    finalpred_save_model_filename = args.model_ckpts_dir + '/'+ f'FINAL_PRED.pkl'
    all_save_model_filenames = [encoder_save_model_filename, 
                                decoder_save_model_filename,
                                finalpred_save_model_filename]
    write_config(args = args, out_dir = args.model_ckpts_dir)
    
    
    ### extract data from dataloader_dict
    # use this to update model parameters
    training_dset = dataloader_dict['training_dset']
    training_dl = dataloader_dict['training_dl']
    
    # use this to decide early stopping
    dev_dset = dataloader_dict['dev_dset']
    dev_dl = dataloader_dict['dev_dl']
    
    # use this as final held-out test set
    final_test_dset = dataloader_dict['test_dset']
    final_test_dl = dataloader_dict['test_dl']
    
    # time
    t_array_for_all_samples = dataloader_dict['t_array_for_all_samples']
    
    # share to lower-level configs
    args.pred_config['training_dset_emit_counts'] = training_dset.emit_counts
    args.pred_config['emissions_postproc_config']['training_dset_emit_counts'] = training_dset.emit_counts
    
    
    
    ###########################################################################
    ### 2: MODEL INIT, TRAINING  ##############################################
    ###########################################################################
    print('2: model init')
    with open(args.logfile_name,'a') as g:
        g.write('\n')
        g.write(f'2: model init\n')
    
    # init the optimizer, split a new rng key
    tx = build_optimizer(args)
    rngkey, model_init_rngkey = jax.random.split(rngkey, num=2)
    
    
    ### determine shapes for init
    # unaligned sequences sizes
    global_seq_max_length = max([training_dset.global_seq_max_length,
                                 dev_dset.global_seq_max_length,
                                 final_test_dset.global_seq_max_length])
    largest_seqs = (args.batch_size, global_seq_max_length)
    
    # aligned datasets sizes
    if args.use_scan_fns:
        max_dim1 = args.chunk_length
    
    elif not args.use_scan_fns:
        max_dim1 = max([training_dset.global_align_max_length,
                        dev_dset.global_align_max_length,
                        final_test_dset.global_align_max_length]) - 1
      
    largest_aligns = (args.batch_size, max_dim1)
    del max_dim1
    
    # time
    if t_array_for_all_samples is not None:
        dummy_t_array_for_all_samples = jnp.empty( (t_array_for_all_samples.shape[0], ) )
        dummy_t_for_each_sample = None
    
    else:
        dummy_t_array_for_all_samples = None
        dummy_t_for_each_sample = jnp.empty( (args.batch_size,) )
    
    # batch provided to train/eval functions consist of:
    # 1.) unaligned sequences (B, L_seq, 2)
    # 2.) aligned data matrices (B, L_align, 5)
    # 3.) time per sample (if applicable) (B,)
    # 4, not used.) sample index (B,)
    seq_shapes = [largest_seqs, largest_aligns, dummy_t_for_each_sample]
    
    
    ### initialize trainstate objects, concat_fn
    out = create_all_tstates( seq_shapes = seq_shapes, 
                              tx = tx, 
                              model_init_rngkey = model_init_rngkey,
                              tabulate_file_loc = args.model_ckpts_dir,
                              anc_model_type = args.anc_model_type, 
                              desc_model_type = args.desc_model_type, 
                              pred_model_type = args.pred_model_type, 
                              anc_enc_config = args.anc_enc_config, 
                              desc_dec_config = args.desc_dec_config, 
                              pred_config = args.pred_config,
                              t_array_for_all_samples = dummy_t_array_for_all_samples,
                              )  
    all_trainstates, all_model_instances, concat_fn = out
    del out
    
    
    ### jit-compilations
    # training function
    parted_train_fn = partial( train_one_batch,
                               all_model_instances = all_model_instances,
                               interms_for_tboard = args.interms_for_tboard,
                               t_array_for_all_samples = t_array_for_all_samples,
                               concat_fn = concat_fn,
                               norm_loss_by_for_reporting = 'desc_len',
                               update_grads = args.update_grads )
    
    train_fn_jitted = jax.jit(parted_train_fn, 
                              static_argnames = ['max_seq_len', 
                                                 'max_align_len',
                                                 'record_interms_this_batch'])
    del parted_train_fn
    
    
    ### eval_fn used in training loop (to monitor progress)
    # pass arguments into eval_one_batch; make a parted_eval_fn that doesn't
    #   return any intermediates
    no_returns = {k: False for k in args.interms_for_tboard.keys()}
    extra_args_for_eval = {'output_attn_weights': False}
    
    parted_eval_fn = partial( eval_one_batch,
                              all_model_instances = all_model_instances,
                              interms_for_tboard = no_returns,
                              t_array_for_all_samples = t_array_for_all_samples,  
                              concat_fn = concat_fn,
                              norm_loss_by_for_reporting = 'desc_len',                  
                              extra_args_for_eval = extra_args_for_eval )
    del no_returns, extra_args_for_eval
    
    # jit compile this eval function
    eval_fn_jitted = jax.jit( parted_eval_fn, 
                              static_argnames = ['max_seq_len','max_align_len'])
    del parted_eval_fn
    
    
    ### initialize training wrapper
    training_wrapper = TrainingWrapper( args = args,
                                        initial_training_rngkey = rngkey,
                                        dataloader_dict = dataloader_dict,
                                        train_fn_jitted = train_fn_jitted,
                                        eval_fn_jitted = eval_fn_jitted,
                                        all_save_model_filenames = all_save_model_filenames,
                                        writer = writer)
    
    
    ### train
    print(f'3: main training loop')
    with open(args.logfile_name,'a') as g:
        g.write('\n')
        g.write(f'3: main training loop\n')
    
    out = training_wrapper.run_train_loop( all_trainstates = all_trainstates )
    early_stop, best_epoch, best_trainstates = out
    del out
    
    
    ###########################################################################
    ### FINAL EVAL   ##########################################################
    ###########################################################################
    print(f'4: post-training actions')
    # write to logfile
    with open(args.logfile_name,'a') as g:
        g.write('\n')
        g.write(f'4: post-training actions\n')
    
    # don't accidentally use old trainstates or eval fn
    del all_trainstates, eval_fn_jitted
    
    # new timer for these steps
    postproc_timer_class = timers( num_epochs = 1 )
    postproc_timer_class.start_timer()


    ### write to output logfile
    with open(args.logfile_name,'a') as g:
        # if early stopping was never triggered, record results at last epoch
        if not early_stop:
            g.write(f'Regular stopping after {args.num_epochs} full epochs:\n\n')
        
        # finish up logfile, regardless of early stopping or not
        g.write(f'Epoch with lowest average dev set loss ("best epoch"): {best_epoch}\n')
        g.write(f'RE-EVALUATING ALL DATA WITH BEST PARAMS\n\n')
    

    ### save the argparse object by itself
    args.epoch_idx = best_epoch
    with open(f'{args.model_ckpts_dir}/TRAINING_ARGPARSE.pkl', 'wb') as g:
        pickle.dump(args, g)
        
    del best_epoch, early_stop


    ### jit compile new eval function
    # if this is a transformer model, will have extra arguments for eval function
    extra_args_for_eval = {'output_attn_weights': args.interms_for_tboard.get('attn_weights', False)}

    parted_eval_fn = partial( eval_one_batch,
                              all_model_instances = all_model_instances,
                              interms_for_tboard = args.interms_for_tboard,
                              t_array_for_all_samples = t_array_for_all_samples,  
                              concat_fn = concat_fn,
                              norm_loss_by_for_reporting = 'desc_len',  
                              extra_args_for_eval = extra_args_for_eval )
    del extra_args_for_eval

    # jit compile this eval function
    eval_fn_jitted = jax.jit( parted_eval_fn, 
                              static_argnames = ['max_seq_len', 'max_align_len'])
    del parted_eval_fn

    ###########################################
    ### loop through training dataloader and  #
    ### score with best params                #
    ###########################################
    with open(args.logfile_name,'a') as g:
        g.write(f'SCORING ALL TRAIN SEQS\n')
        
    # DON'T save arrays yet; takes up too much memory
    train_summary_stats = final_eval_wrapper(dataloader = training_dl, 
                                             dataset = training_dset, 
                                             best_trainstates = best_trainstates, 
                                             jitted_determine_seqlen_bin = training_wrapper.seqlen_bin_fn,
                                             jitted_determine_alignlen_bin = training_wrapper.alignlen_bin_fn,
                                             eval_fn_jitted = eval_fn_jitted,
                                             out_alph_size = None,
                                             save_arrs = False,
                                             save_per_sample_losses = args.save_per_sample_losses,
                                             interms_for_tboard = args.interms_for_tboard, 
                                             logfile_dir = args.logfile_dir,
                                             out_arrs_dir = args.out_arrs_dir,
                                             outfile_prefix = f'train-set')
    
    
    ##########################################
    ### loop through dev set dataloader and  #
    ### score with best params               #
    ##########################################
    with open(args.logfile_name,'a') as g:
        g.write(f'SCORING ALL DEV SEQS\n')
        
        # DON'T save arrays yet; takes up too much memory
        dev_summary_stats = final_eval_wrapper( dataloader = dev_dl, 
                                                dataset = dev_dset, 
                                                best_trainstates = best_trainstates, 
                                                jitted_determine_seqlen_bin = training_wrapper.seqlen_bin_fn,
                                                jitted_determine_alignlen_bin = training_wrapper.alignlen_bin_fn,
                                                eval_fn_jitted = eval_fn_jitted,
                                                out_alph_size = None,
                                                save_arrs = False,
                                                save_per_sample_losses = args.save_per_sample_losses,
                                                interms_for_tboard = args.interms_for_tboard, 
                                                logfile_dir = args.logfile_dir,
                                                out_arrs_dir = args.out_arrs_dir,
                                                outfile_prefix = f'dev-set')
        

    ###########################################
    ### loop through test dataloader and      #
    ### score with best params                #
    ###########################################
    with open(args.logfile_name,'a') as g:
        g.write(f'SCORING ALL HELD-OUT TEST SEQS\n')
        
    final_test_summary_stats = final_eval_wrapper(dataloader = final_test_dl, 
                                             dataset = final_test_dset, 
                                             best_trainstates = best_trainstates, 
                                             jitted_determine_seqlen_bin = training_wrapper.seqlen_bin_fn,
                                             jitted_determine_alignlen_bin = training_wrapper.alignlen_bin_fn,
                                             eval_fn_jitted = eval_fn_jitted,
                                             out_alph_size = None, 
                                             save_arrs = args.save_arrs,
                                             save_per_sample_losses = args.save_per_sample_losses,
                                             interms_for_tboard = args.interms_for_tboard, 
                                             logfile_dir = args.logfile_dir,
                                             out_arrs_dir = args.out_arrs_dir,
                                             outfile_prefix = f'final-test-set')


    ###########################################
    ### update the logfile with final losses  #
    ###########################################
    write_final_eval_results(args = args, 
                             summary_stats = train_summary_stats,
                             filename = 'TRAIN_AVE-LOSSES.tsv')

    write_final_eval_results(args = args, 
                             summary_stats = dev_summary_stats,
                             filename = 'DEV_AVE-LOSSES.tsv')

    write_final_eval_results(args = args, 
                             summary_stats = final_test_summary_stats,
                             filename = 'FINAL-TEST_AVE-LOSSES.tsv')

    # record total time spent on post-training actions; write this to a table
    #   instead of a scalar collection
    record_postproc_time_table( already_started_timer_class = postproc_timer_class,
                                writer = writer )

    writer.close()
    
    # clean up intermediates
    for file_path in glob.glob(f"{args.model_ckpts_dir}/*_INPROGRESS.pkl"):
        try:
            os.remove(file_path)
        except FileNotFoundError:
            pass  # File might have been deleted already
    
