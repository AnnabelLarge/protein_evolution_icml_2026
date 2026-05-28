#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 15:59:45 2025

"""
import torch
import random
from torch.utils.data import DataLoader
import numpy as np

def init_dataloader(args, 
                    shuffle,
                    pytorch_custom_dset,
                    collate_fn):
    if shuffle:
        torch.manual_seed(args.rng_seednum)
        random.seed(args.rng_seednum)
        np.random.seed(args.rng_seednum)
        
    dl = DataLoader( pytorch_custom_dset, 
                     batch_size = args.batch_size, 
                     shuffle = shuffle,
                     collate_fn = collate_fn
                     )
    
    return dl

