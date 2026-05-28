#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 16:13:55 2023


"""
import json
import os
import argparse
import jax
import pickle
import shutil
import sys
import gc

# allow sibling-package imports (dloaders, cli, utils, etc.) when run as a package
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dloaders.init_dataloader import init_dataloader


def main():
    ### for now, running models on single GPU
    err_ms = 'SELECT GPU TO RUN THIS COMPUTATION ON with CUDA_VISIBLE_DEVICES=DEVICE_NUM'
    assert len(jax.devices()) == 1, err_ms
    del err_ms
    
    ###########################################################################
    ### INITIALIZE PARSER   ###################################################
    ###########################################################################
    parser = argparse.ArgumentParser(prog='Pair_Alignment')
    
    ### which program do you want to run?
    valid_tasks = ['train',
                   'continue_train',
                   'eval']
    
    parser.add_argument('-task',
                        type=str,
                        required=True,
                        choices = valid_tasks,
                        help=f'What do you want to do? Pick from: {valid_tasks}')
    
    parser.add_argument('-configs',
                        type = str,
                        required=True,
                        help='Load configs from file or folder of files, in json format.')
    
    # only needed when continuing training
    parser.add_argument('-new_training_wkdir',
                        type = str,
                        help='FOR CONTINUE_TRAIN OPTION; Name for a new training working dir')
    
    parser.add_argument('-prev_model_ckpts_dir',
                        type = str,
                        help='FOR CONTINUE_TRAIN OPTION; Path to previous trainstate, argparse object')
    
    parser.add_argument('-tstate_to_load',
                        type = str,
                        help='FOR CONTINUE_TRAIN OPTION; The suffix (not including file extension) of the tstate object to load')
  
    # parse the arguments
    top_level_args = parser.parse_args()
    
    
    ### helper functions 
    # open a single config file and extract additional arguments
    def read_config_file(config_file):
        with open(config_file, 'r') as f:
            contents = json.load(f)
            t_args = argparse.Namespace()
            t_args.__dict__.update(contents)
            args = parser.parse_args(namespace=t_args)
        return args
    

    ###########################################################################
    ### TRAINING: basic function   ############################################
    ###########################################################################
    if top_level_args.task == 'train':
        # read argparse
        assert top_level_args.configs.endswith('.json'), "input is one JSON file"
        print(f'TRAINING WITH: {top_level_args.configs}')
        args = read_config_file(top_level_args.configs)
        pred_model_type = args.pred_model_type
        
        # import correct wrappers, dataloader initializers
        if 'pairhmm_indp_sites' in pred_model_type:
            from cli.train_pairhmm_indp_sites import train_pairhmm_indp_sites as train_fn
            from dloaders.init_counts_dset import init_counts_dset as init_datasets
            from dloaders.CountsDset import jax_collator as collate_fn
            
        elif pred_model_type in ['pairhmm_frag_and_site_classes',
                                 'pairhmm_nested_tkf',
                                 'neural_hmm',
                                 'feedforward']:
            from dloaders.init_full_len_dset import init_full_len_dset as init_datasets
            from dloaders.FullLenDset import jax_collator as collate_fn
            
            if pred_model_type in ['pairhmm_frag_and_site_classes', 'pairhmm_nested_tkf']:
                from cli.train_pairhmm_transit_mixes import train_pairhmm_transit_mixes as train_fn
                
            elif pred_model_type == 'neural_hmm':
                from cli.train_neural_hmm import train_neural_hmm as train_fn
    
            elif pred_model_type == 'feedforward':
                from cli.train_feedforward import train_feedforward as train_fn
                
        # make dataloder list
        dload_dict = init_datasets( args,
                                    'train',
                                    training_argparse = None,
                                    include_dataloader = True )
            
        # train model
        train_fn( args, dload_dict )


    
    ###########################################################################
    ### TRAINING: continue one training experiment   ##########################
    ###########################################################################
    elif top_level_args.task == 'continue_train':
        # read argparse
        assert top_level_args.configs.endswith('.json'), "input is one JSON file"
        print(f'CONTINUE TRAINING WITH: {top_level_args.configs}, IN NEW DIR {top_level_args.new_training_wkdir}')
        args_from_training_config = read_config_file(top_level_args.configs)
        pred_model_type = args_from_training_config.pred_model_type
        
        # import correct wrappers, dataloader initializers
        if 'pairhmm_indp_sites' in pred_model_type:
            from cli.cont_training_pairhmm_indp_sites import cont_training_pairhmm_indp_sites as cont_train_fn
            from dloaders.init_counts_dset import init_counts_dset as init_datasets
            from dloaders.CountsDset import jax_collator as collate_fn
        
        elif pred_model_type in ['pairhmm_frag_and_site_classes',
                                 'pairhmm_nested_tkf',
                                 'neural_hmm',
                                 'feedforward']:
            from dloaders.init_full_len_dset import init_full_len_dset as init_datasets
            from dloaders.FullLenDset import jax_collator as collate_fn
            
            if pred_model_type in ['pairhmm_frag_and_site_classes', 'pairhmm_nested_tkf']:
                from cli.cont_training_pairhmm_transit_mixes import cont_training_pairhmm_transit_mixes as cont_train_fn
                
            elif pred_model_type == 'neural_hmm':
                from cli.cont_training_neural_hmm import cont_training_neural_hmm as cont_train_fn
    
            elif pred_model_type == 'feedforward':
                from cli.cont_training_feedforward import cont_training_feedforward as cont_train_fn

        # make dataloader objects
        dload_dict = init_datasets( args_from_training_config,
                                    'train',
                                    training_argparse = None,
                                    include_dataloader = True )
            
        # train model
        cont_train_fn( args=args_from_training_config, 
                        dataloader_dict=dload_dict,
                        new_training_wkdir=top_level_args.new_training_wkdir,
                        prev_model_ckpts_dir=top_level_args.prev_model_ckpts_dir,
                        tstate_to_load=top_level_args.tstate_to_load
                        )
    
    
    ###########################################################################
    ### EVAL: basic function   ################################################
    ###########################################################################
    elif top_level_args.task == 'eval':
        # read argparse
        assert top_level_args.configs.endswith('.json'), "input is one JSON file"
        print(f'EVALUATING WITH: {top_level_args.configs}')
        args = read_config_file(top_level_args.configs)
        
        # find and read training argparse
        model_ckpts_dir = f'{os.getcwd()}/{args.training_wkdir}/model_ckpts'
        training_argparse_filename = model_ckpts_dir + '/' + 'TRAINING_ARGPARSE.pkl'
        
        with open(training_argparse_filename,'rb') as g:
            training_argparse = pickle.load(g)
        
        
        ### determine pred_model_type
        # automatically detect
        pred_model_type = training_argparse.pred_model_type
        override_with_pred_model_type = None
        
        # uncomment to override; useful for debugging
        # pred_model_type = override_with_pred_model_type
        # override_with_pred_model_type = override_with_pred_model_type
        
        
        ### import correct wrappers, dataloader initializers
        if 'pairhmm_indp_sites' in pred_model_type:
            from cli.eval_pairhmm_indp_sites import eval_pairhmm_indp_sites as eval_fn
            from dloaders.init_counts_dset import init_counts_dset as init_datasets
            from dloaders.CountsDset import jax_collator as collate_fn

        elif pred_model_type in ['pairhmm_frag_and_site_classes',
                                 'pairhmm_nested_tkf',
                                 'neural_hmm',
                                 'feedforward']:
            from dloaders.init_full_len_dset import init_full_len_dset as init_datasets
            from dloaders.FullLenDset import jax_collator as collate_fn

            if pred_model_type in ['pairhmm_frag_and_site_classes', 'pairhmm_nested_tkf']:
                from cli.eval_pairhmm_transit_mixes import eval_pairhmm_transit_mixes as eval_fn

            elif pred_model_type == 'neural_hmm':
                from cli.eval_neural_hmm import eval_neural_hmm as eval_fn
            
            elif pred_model_type == 'feedforward':
                from cli.eval_feedforward import eval_feedforward as eval_fn

        # load data; saved under trianing_wkdir name
        dload_dict = init_datasets( args,
                                    'eval',
                                    training_argparse,
                                    include_dataloader = True )
            
        # evaluate model
        eval_fn( args = args, 
                 training_argparse = training_argparse,
                 dataloader_dict = dload_dict, 
                 override_with_pred_model_type = override_with_pred_model_type)
    
    
            
if __name__ == '__main__':
    main()