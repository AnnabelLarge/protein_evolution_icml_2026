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
from argparse import Namespace

from dloaders.FullLenDset import FullLenDset
from dloaders.FullLenDset import jax_collator as collator
from dloaders.init_dataloader import init_dataloader
from dloaders.init_time_array import init_time_array




def _determine_emission_alphabet_size(argparse_obj, 
                                      pred_model_type):
    if pred_model_type == "feedforward":
        return 20
    else: 
        return 4 if (argparse_obj.pred_config.get("subst_model_type") == "hky85") else 20

def _maybe_return_time_grid(argparse_obj):
    """
    returns grid of times if desired, otherwise None
    t_per_sample is True if this returns None, False otherwise
    """
    # if either of these are true, return None
    cond1 = argparse_obj.pred_config['times_from'] is None
    cond2 = argparse_obj.pred_config['times_from'] == 't_per_sample'
    t_per_sample = cond1 or cond2
    
    if t_per_sample:
        return None
        
    # init a grid if t_per_sample is False
    elif not t_per_sample:
        return init_time_array( argparse_obj )

def _make_dataset(args, 
                  splitname,
                  split_prefixes, 
                  pred_model_type, 
                  use_scan_fns, 
                  t_per_sample,
                  emission_alphabet_size = 20, 
                  gap_idx = 43,
                  seq_padding_idx = 0,
                  align_padding_idx = -9):
    
    # make sure this is a list of names
    assert isinstance(split_prefixes, list)
    print(f'{splitname} dset:')
    for s in split_prefixes:
        print(s)
    print()
    
    # init
    return FullLenDset( data_dir = args.data_dir,
                        split_prefixes = split_prefixes,
                        pred_model_type = pred_model_type,
                        use_scan_fns = use_scan_fns,
                        t_per_sample = t_per_sample,
                        emission_alphabet_size = emission_alphabet_size,
                        chunk_length = args.chunk_length,
                        toss_alignments_longer_than = args.toss_alignments_longer_than,
                        seq_padding_idx = seq_padding_idx,
                        align_padding_idx = align_padding_idx,
                        gap_idx = gap_idx )


def init_full_len_dset( args: Namespace,
                        task: str,
                        training_argparse = None,
                        include_dataloader: bool = True ):
    # Determine context (train vs eval)
    if task in ["train", "resume_train"]:
        argparse_obj = args
        only_test = False
    elif task == "eval":
        argparse_obj = training_argparse
        only_test = True

    pred_model_type = argparse_obj.pred_model_type
    gap_idx = getattr( argparse_obj, "gap_idx", 43 )
    
    # determine if classical mixture or neural model
    is_neural = pred_model_type in ['feedforward', 'neural_hmm']


    ### Special defaults for model types
    if not is_neural:
        argparse_obj.use_scan_fns = False
        
    elif is_neural and pred_model_type == "feedforward":
        # enforce feedforward defaults
        if argparse_obj.pred_config["t_per_sample"]:
            argparse_obj.pred_config["times_from"] = "t_per_sample"
        else:
            argparse_obj.pred_config["times_from"] = None

    # other defaults: out alphabet size
    emission_alphabet_size = _determine_emission_alphabet_size(argparse_obj = argparse_obj, 
                                                               pred_model_type = pred_model_type) 
    
    # other defaults: grid of times (could be either (T,) array, or none)
    t_array_for_all_samples = _maybe_return_time_grid(argparse_obj = argparse_obj)
    
    # if t_array_for_all_samples is None, then there's no time grid; use one
    # branch length per sample
    t_per_sample = t_array_for_all_samples is None


    ### Build dataset objects
    out = {"t_array_for_all_samples": t_array_for_all_samples}

    # Test dataset
    out["test_dset"] = _make_dataset(args = args, 
                                     splitname = 'Test',
                                     split_prefixes = args.test_dset_splits,
                                     pred_model_type = pred_model_type, 
                                     use_scan_fns = args.use_scan_fns,
                                     t_per_sample = t_per_sample, 
                                     emission_alphabet_size = emission_alphabet_size, 
                                     gap_idx = gap_idx,
                                     seq_padding_idx = 0,
                                     align_padding_idx = -9)

    # Dev dataset: if training a neural model, use this for early stopping criteria
    if is_neural and not only_test:
        out["dev_dset"] = _make_dataset(args = args, 
                                        splitname = 'Dev',
                                        split_prefixes = args.dev_dset_splits,
                                        pred_model_type = pred_model_type, 
                                        use_scan_fns = args.use_scan_fns,
                                        t_per_sample = t_per_sample, 
                                        emission_alphabet_size = emission_alphabet_size, 
                                        gap_idx = gap_idx,
                                        seq_padding_idx = 0,
                                        align_padding_idx = -9)

    # Training set: if training any model
    if not only_test:
        out["training_dset"] = _make_dataset(args = args, 
                                             splitname = 'Train',
                                             split_prefixes = args.train_dset_splits,
                                             pred_model_type = pred_model_type, 
                                             use_scan_fns = args.use_scan_fns,
                                             t_per_sample = t_per_sample, 
                                             emission_alphabet_size = emission_alphabet_size, 
                                             gap_idx = gap_idx,
                                             seq_padding_idx = 0,
                                             align_padding_idx = -9)
    
    
    ### Dataloaders
    if include_dataloader:
        # Test dataset
        out["test_dl"] = init_dataloader(args = args, 
                                  pytorch_custom_dset = out["test_dset"],
                                  shuffle = False,
                                  collate_fn = collator)
        
        # Dev dataset: if training a neural model, use this for early stopping criteria
        if is_neural and not only_test:
            out["dev_dl"] = init_dataloader(args = args, 
                                            pytorch_custom_dset = out["dev_dset"],
                                            shuffle = False,
                                            collate_fn = collator)
            
        # Training set: if training any model
        if not only_test:
            out["training_dl"] = init_dataloader(args = args, 
                                                 pytorch_custom_dset = out["training_dset"],
                                                 shuffle = True,
                                                 collate_fn = collator)

    return out
