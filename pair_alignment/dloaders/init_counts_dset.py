#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 17:36:42 2025


About:
=======
initialize dataloaders and pytorch datasets

"""
import pickle
import torch
from torch.utils.data import DataLoader
import numpy as np

from dloaders.CountsDset import CountsDset 
from dloaders.CountsDset import jax_collator as collator
from dloaders.init_dataloader import init_dataloader
from dloaders.init_time_array import init_time_array

   
def init_counts_dset( args, 
                      task, 
                      training_argparse=None,
                      include_dataloader=True ):
    """
    initialize the dataloaders
    """
    #################################
    ### training-specific options   #
    #################################
    if task in ['train', 
                'resume_train']:
        only_test = False
        t_per_sample = args.pred_config['times_from'] == 't_per_sample'
        t_array_for_all_samples = init_time_array(args)
        subs_only = (args.pred_config['indel_model_type'] is None)
        
        if args.pred_config['subst_model_type'].lower() == 'hky85':
            emission_alphabet_size = 4
        else:
            emission_alphabet_size = 20
            
    
    #############################
    ### eval-specific options   #
    #############################
    elif task in ['eval']:
        only_test = True
        t_per_sample = training_argparse.pred_config['times_from'] == 't_per_sample'
        t_array_for_all_samples = init_time_array(training_argparse)
        subs_only = (training_argparse.pred_config['indel_model_type'] is None)
        
        if training_argparse.pred_config['subst_model_type'].lower() == 'hky85':
            emission_alphabet_size = 4
        else:
            emission_alphabet_size = 20
            
            
    #################
    ### LOAD DATA   #
    #################
    # test data
    assert type(args.test_dset_splits) == list

    print('Test dset:')
    for s in args.test_dset_splits:
        print(s)
    print()

    test_dset = CountsDset( data_dir = args.data_dir, 
                            split_prefixes = args.test_dset_splits,
                            emission_alphabet_size = emission_alphabet_size,
                            t_per_sample = t_per_sample,
                            subs_only = subs_only,
                            toss_alignments_longer_than = args.toss_alignments_longer_than)
    out = {'test_dset': test_dset,
           't_array_for_all_samples': t_array_for_all_samples}


    # training data
    if not only_test:
        assert type(args.train_dset_splits) == list

        print('Training dset:')
        for s in args.train_dset_splits:
            print(s)
        print()
        
        training_dset = CountsDset( data_dir = args.data_dir, 
                                    split_prefixes = args.train_dset_splits,
                                    emission_alphabet_size = emission_alphabet_size,
                                    t_per_sample = t_per_sample,
                                    subs_only = subs_only,
                                    toss_alignments_longer_than = args.toss_alignments_longer_than)
        out['training_dset'] = training_dset
        
        
    ############################################
    ### create dataloaders, output dictionary  #
    ############################################
    if include_dataloader:
        test_dl = init_dataloader(args = args, 
                                  pytorch_custom_dset = test_dset,
                                  shuffle = False,
                                  collate_fn = collator)
        out['test_dl'] = test_dl
        
        if not only_test:
            training_dl = init_dataloader(args = args, 
                                          pytorch_custom_dset = training_dset,
                                          shuffle = True,
                                          collate_fn = collator)
            out['training_dl'] = training_dl
            
    return out
