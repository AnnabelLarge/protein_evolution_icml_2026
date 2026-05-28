#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 12:33:01 2025


Load parameters and evaluate likelihoods for an independent
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
from utils.end_training import write_final_eval_results
                                 
# specific to training this model
from utils.edit_argparse import pairhmm_indp_sites_fill_with_default_values as fill_with_default_values
from utils.edit_argparse import pairhmms_share_top_level_args as share_top_level_args


def eval_pairhmm_indp_sites(args, 
                            training_argparse,
                            dataloader_dict: dict,
                            override_with_pred_model_type = None):
    ###########################################################################
    ### 0: CHECK CONFIG; IMPORT APPROPRIATE MODULES   #########################
    ###########################################################################
    # final where model pickles, previous argparses are
    err = (f"{training_argparse.pred_model_type} is not pairhmm_indp_sites or "+
           f"old_style_pairhmm_indp_sites; using the wrong training script!")
    assert training_argparse.pred_model_type in ['pairhmm_indp_sites', 'old_style_pairhmm_indp_sites'], err
    del err


    ### decide whether to train with current code, or code to run old-style models
    # override pred_model_type, possibly
    if override_with_pred_model_type is None:
        pred_model_type = training_argparse.pred_model_type 
    
    elif override_with_pred_model_type is not None:
        pred_model_type = override_with_pred_model_type
        training_argparse.pred_config['tie_params'] = False # manually set this to false, I guess
    
    # import functions
    if pred_model_type == 'pairhmm_indp_sites':
        from latent_class_mixtures.initializers import init_pairhmm_indp_sites as init_pairhmm
        from train_eval_fns.indp_site_classes_training_fns import ( eval_one_batch,
                                                                    final_eval_wrapper )
    
    elif pred_model_type == 'old_style_pairhmm_indp_sites':
        from older_indel_models.initializers import init_pairhmm
        from train_eval_fns.old_style_indp_site_classes_training_fns import ( eval_one_batch,
                                                                              final_eval_wrapper )
    
    
    # get file handles
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
    
    # create a new logfile
    with open(args.logfile_name,'w') as g:
        g.write(f"{datetime.now()}\n\n")
        
        g.write( f'Loading from {args.training_wkdir} to eval new data\n\n' )
        
        # standard header
        g.write( f'PairHMM with independent site classes over emissions\n' )
        g.write( f'Substitution model: {training_argparse.pred_config["subst_model_type"]}\n' )
        g.write( f'Indel model: {training_argparse.pred_config.get("indel_model_type","None")}\n\n' )
        
        g.write( f'Number of domain mixes: 1\n' )
        g.write( f'Number of fragment mixes: 1\n' )
        g.write( f'Number of site mixes: {training_argparse.pred_config["num_site_mixtures"]}\n' )
        g.write( f'Number of rate multipliers: {training_argparse.pred_config["k_rate_mults"]}\n' )
                
        # note if rates are independent
        if training_argparse.pred_config['indp_rate_mults']:
            g.write( f'  - Rates are independent of site class label: ( P(k | c) = P(k) )\n' )
                    
        elif not training_argparse.pred_config['indp_rate_mults']:
            g.write( f'  - Rates depend on class labels\n' )
        
        # how to normalize reported metrics (usually by descendant length)
        if training_argparse.pred_config["indel_model_type"] is not None:
            g.write( f'  - When reporting, normalizing losses by: descendant length\n' )
        
        elif training_argparse.pred_config["indel_model_type"] is None:
            g.write( f'  - When reporting, normalizing losses by: align length '+
                     f'(same as desc length, because we remove gap '+
                     f'positions) \n' )
        
        # write source of times
        g.write( f'Times from: {training_argparse.pred_config["times_from"]}\n' )
    
    with open(f'{args.out_arrs_dir}/FINAL-EVAL_tkf_approx.tsv','w') as g:
        g.write('Used tkf approximations in the following locations:\n')
        
    
    ### extract data from dataloader_dict
    test_dset = dataloader_dict['test_dset']
    test_dl = dataloader_dict['test_dl']
    t_array_for_all_samples = dataloader_dict['t_array_for_all_samples']
    
    
    ###########################################################################
    ### 1: INITIALIZE MODEL PARTS, OPTIMIZER  #################################
    ###########################################################################
    print('MODEL INIT')
    with open(args.logfile_name,'a') as g:
        g.write('\n')
        g.write(f'MODEL INIT\n')
    
    
    # need to intialize an optimizer for compatibility when restoring the state, 
    #   but we're not training so this doesn't really matter
    tx = build_optimizer(training_argparse)
    
    
    ### determine shapes for init
    B = training_argparse.batch_size
    A = training_argparse.emission_alphabet_size
    S = test_dset.num_transitions
    
    # time
    if t_array_for_all_samples is not None:
        dummy_t_array_for_all_samples = jnp.empty( (t_array_for_all_samples.shape[0], ) )
        dummy_t_for_each_sample = None
    
    else:
        dummy_t_array_for_all_samples = None
        dummy_t_for_each_sample = jnp.empty( (B,) )
        
    # counts array
    dummy_subCounts = jnp.empty( (B, A, A) )
    dummy_insCounts = jnp.empty( (B, A) )
    dummy_delCounts = jnp.empty( (B, A) )
    dummy_transCounts = jnp.empty( (B, S, S) )
    
    fake_batch = [dummy_subCounts,
                  dummy_insCounts,
                  dummy_delCounts,
                  dummy_transCounts,
                  dummy_t_for_each_sample]
    
    
    ### initialize functions
    out = init_pairhmm( seq_shapes = fake_batch, 
                        dummy_t_array = dummy_t_array_for_all_samples,
                        tx = tx, 
                        model_init_rngkey = jax.random.key(0),
                        pred_config = training_argparse.pred_config,
                        tabulate_file_loc = args.model_ckpts_dir
                        )
    blank_tstate, pairhmm_instance = out
    del out
    
    # load values
    with open(pairhmm_savemodel_filename, 'rb') as f:
        state_dict = pickle.load(f)
        
    best_pairhmm_trainstate = flax.serialization.from_state_dict( blank_tstate, 
                                                                  state_dict )
    del blank_tstate, state_dict
    
    
    ### part+jit eval function
    no_outputs = {k: False for k in training_argparse.interms_for_tboard.keys()}
    parted_eval_fn = partial( eval_one_batch,
                              t_array = t_array_for_all_samples,
                              all_trainstates = [best_pairhmm_trainstate],
                              pairhmm_instance = pairhmm_instance,
                              interms_for_tboard = no_outputs,
                              return_all_loglikes = True )
    eval_fn_jitted = jax.jit(parted_eval_fn)
    del parted_eval_fn
    
    
    ### write the parameters again
    if args.save_arrs:
        # if using a set grid, write for all values in the grid
        if t_array_for_all_samples is not None:
            best_pairhmm_trainstate.apply_fn( variables = best_pairhmm_trainstate.params,
                                              t_array = t_array_for_all_samples,
                                              prefix = 't_grid',
                                              out_folder = args.out_arrs_dir,
                                              write_time_static_objs = True,
                                              method = pairhmm_instance.write_params )
            
        # if using one branch length per sample, write parameters at t=1.0
        elif t_array_for_all_samples is None:
            best_pairhmm_trainstate.apply_fn( variables = best_pairhmm_trainstate.params,
                                              t_array = jnp.array([1.0]),
                                              prefix = 't=1',
                                              out_folder = args.out_arrs_dir,
                                              write_time_static_objs = True,
                                              method = pairhmm_instance.write_params )
    
    
    ###########################################################################
    ### 2: EVAL   #############################################################
    ###########################################################################
    print(f'BEGIN eval')
    # write to logfile
    with open(args.logfile_name,'a') as g:
        g.write('\n')
        g.write(f'BEGIN eval\n')

    test_summary_stats = final_eval_wrapper(dataloader = test_dl, 
                                            dataset = test_dset,  
                                            eval_fn_jitted = eval_fn_jitted,
                                            save_per_sample_losses = args.save_per_sample_losses,
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
    