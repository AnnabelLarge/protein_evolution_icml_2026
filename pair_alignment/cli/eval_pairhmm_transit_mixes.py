#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 11:55:46 2025

Load parameters and evaluate likelihoods for an markovian
  site class model

"""
# general python
import os
import pandas as pd
pd.options.mode.chained_assignment = None # this is annoying
import pickle
from functools import partial
import platform
import argparse
import json
from tqdm import tqdm
from datetime import datetime

# jax/flax stuff
import jax
import jax.numpy as jnp
import flax

# pytorch imports
from torch.utils.data import DataLoader

# custom function/classes imports (in order of appearance)
from utils.build_optimizer import build_optimizer
from utils.edit_argparse import enforce_valid_defaults
from train_eval_fns.general_training_wrapper.training_wrapper_helpers import jit_compile_determine_alignlen_bin
from utils.end_training import write_final_eval_results

# specific to this model
from utils.edit_argparse import pairhmm_frag_and_site_classes_fill_with_default_values as fill_with_default_values
from utils.edit_argparse import pairhmms_share_top_level_args as share_top_level_args
from latent_class_mixtures.initializers import init_pairhmm_transit_mixes as init_pairhmm
from train_eval_fns.transit_mixes_training_fns import ( eval_one_batch,
                                                        final_eval_wrapper )


def eval_pairhmm_transit_mixes( args, 
                                dataloader_dict: dict,
                                training_argparse,
                                **kwargs ):
    ###########################################################################
    ### 0: CHECK CONFIG; IMPORT APPROPRIATE MODULES   #########################
    ###########################################################################
    # final where model pickles, previous argparses are
    err = (f"Pred model type: {training_argparse.pred_model_type}; "+
           f"this is the eval script for pairHMM with mixtures of transit classes!")
    assert training_argparse.pred_model_type in ['pairhmm_frag_and_site_classes', 'pairhmm_nested_tkf'], err
    del err
        
    prev_model_ckpts_dir = f'{os.getcwd()}/{args.training_wkdir}/model_ckpts'
    pairhmm_savemodel_filename = prev_model_ckpts_dir + '/'+ f'FINAL_PRED_BEST.pkl'
    
    fill_with_default_values(training_argparse)
    enforce_valid_defaults(training_argparse)
    share_top_level_args(training_argparse)


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
    
    # new place to save final pred outputs
    finalpred_save_model_filename = args.model_ckpts_dir + '/'+ f'FINAL_PRED.pkl'
    
    # create a new logfile
    with open(args.logfile_name,'w') as g:
        g.write(f"{datetime.now()}\n\n")
        
        g.write( f'Loading from {training_argparse.training_wkdir} to eval new data\n' )
        
        # standard header
        g.write( f'PairHMM TKF92 with mixtures of transit classes: {training_argparse.pred_model_type}\n' )
        g.write( f'Substitution model: {training_argparse.pred_config["subst_model_type"]}\n' )
        g.write( f'Indel model: TKF92\n\n' )
        
        g.write( f'Number of domain mixes: {training_argparse.pred_config["num_domain_mixtures"]}\n' )
        g.write( f'Number of fragment mixes: {training_argparse.pred_config["num_fragment_mixtures"]}\n' )
        g.write( f'Number of site mixes: {training_argparse.pred_config["num_site_mixtures"]}\n' )
        g.write( f'Number of rate multipliers: {training_argparse.pred_config["k_rate_mults"]}\n' )
                
        # note if rates are independent
        if training_argparse.pred_config['indp_rate_mults']:
            g.write( f'  - Rates are independent of site class label: ( P(k | c) = P(k) )\n' )
                    
        elif not training_argparse.pred_config['indp_rate_mults']:
            g.write( f'  - Rates depend on class labels\n' )
        
        # how to normalize reported metrics (usually by descendant length)
        g.write(f'  - When reporting, normalizing losses by: descendant length\n')
        
        # write source of times
        g.write( f'Times from: {training_argparse.pred_config["times_from"]}\n' )
    
    
    ### extract data from dataloader_dict
    test_dset = dataloader_dict['test_dset']
    test_dl = dataloader_dict['test_dl']
    t_array_for_all_samples = dataloader_dict['t_array_for_all_samples']
    
    
    ###########################################################################
    ### 2: INITIALIZE MODEL PARTS, OPTIMIZER  #################################
    ###########################################################################
    print('MODEL INIT')
    with open(args.logfile_name,'a') as g:
        g.write('\n')
        g.write(f'1: model init\n')
    
    
    # need to intialize an optimizer for compatibility when restoring the state, 
    #   but we're not training so this doesn't really matter?
    tx = build_optimizer(training_argparse)
    
    
    ### determine shapes for init
    # time
    if t_array_for_all_samples is not None:
        dummy_t_array_for_all_samples = jnp.empty( (t_array_for_all_samples.shape[0], ) )
        dummy_t_for_each_sample = None
    
    else:
        dummy_t_array_for_all_samples = None
        dummy_t_for_each_sample = jnp.empty( (args.batch_size,) )
        
    
    ### init sizes
    # (B, L, 3)
    max_dim1 = test_dset.global_align_max_length 
    largest_aligns = jnp.empty( (args.batch_size, max_dim1, 3), dtype=int )
    del max_dim1
    
    
    ### initialize functions
    seq_shapes = [largest_aligns,
                  dummy_t_for_each_sample]
    
    out = init_pairhmm( pred_model_type = training_argparse.pred_model_type,
                        seq_shapes = seq_shapes, 
                        dummy_t_array = dummy_t_array_for_all_samples,
                        tx = tx, 
                        model_init_rngkey = jax.random.key(0),
                        pred_config = training_argparse.pred_config,
                        tabulate_file_loc = args.model_ckpts_dir)
    blank_tstate, pairhmm_instance = out
    del out
    
    # load values
    with open(pairhmm_savemodel_filename, 'rb') as f:
        state_dict = pickle.load(f)
        
    best_pairhmm_trainstate = flax.serialization.from_state_dict( blank_tstate, 
                                                                  state_dict )
    del blank_tstate, state_dict
    
    
    ### part+jit functions
    # manage sequence lengths
    jitted_determine_alignlen_bin = jit_compile_determine_alignlen_bin(training_argparse)
    
    no_outputs = {k: False for k in training_argparse.interms_for_tboard.keys()}
    parted_eval_fn = partial( eval_one_batch,
                              t_array = t_array_for_all_samples,
                              all_trainstates = [best_pairhmm_trainstate],
                              pairhmm_instance = pairhmm_instance,
                              interms_for_tboard = no_outputs,
                              return_all_loglikes = True )
    eval_fn_jitted = jax.jit(parted_eval_fn, 
                              static_argnames = ['max_align_len'])
    del parted_eval_fn
    
    
    ### un-transform parameters and write to numpy arrays
    if args.save_arrs:
        # if using a set grid, write for all values in the grid
        if t_array_for_all_samples is not None:
            best_pairhmm_trainstate.apply_fn( variables = best_pairhmm_trainstate.params,
                                              t_array = t_array_for_all_samples,
                                              prefix = 't_grid',
                                              out_folder = args.out_arrs_dir,
                                              write_time_static_objs = True,
                                              method = pairhmm_instance.write_params )
            
        # if using one branch length per sample, write arrays with t=1.0
        elif t_array_for_all_samples is None:
            best_pairhmm_trainstate.apply_fn( variables = best_pairhmm_trainstate.params,
                                              t_array = jnp.array([1.0]),
                                              prefix = 't=1',
                                              out_folder = args.out_arrs_dir,
                                              write_time_static_objs = True,
                                              method = pairhmm_instance.write_params )
    
    
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
                                            eval_fn_jitted = eval_fn_jitted,
                                            save_per_sample_losses = args.save_per_sample_losses,
                                            jitted_determine_alignlen_bin = jitted_determine_alignlen_bin,
                                            logfile_dir = args.logfile_dir,
                                            out_arrs_dir = args.out_arrs_dir,
                                            outfile_prefix = f'test-dset')
    
    
    ###########################################
    ### update the logfile with final losses  #
    ###########################################
    # save the trainstate again
    with open(f'{args.model_ckpts_dir}/FINAL_PRED.pkl', 'wb') as g:
        model_state_dict = flax.serialization.to_state_dict(best_pairhmm_trainstate)
        pickle.dump(model_state_dict, g)
        
    write_final_eval_results(args = args, 
                             summary_stats = test_summary_stats,
                             filename = 'AVE-LOSSES.tsv')
    