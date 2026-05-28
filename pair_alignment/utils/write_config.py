#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 11:05:01 2024

"""
import os
import json
import argparse
from collections.abc import MutableMapping
import numpy as np
import pickle

def flatten_convert(dictionary, 
                    parent_key=None, 
                    separator = '/'):
    items = []
    for key, value in dictionary.items():
        new_key = parent_key + separator + key if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(flatten_convert(value, new_key, separator=separator).items())
        else:
            items.append((new_key, value)) #write value, not np.array(value)
    return dict(items)


def write_config(args, out_dir):
    with open(f'{out_dir}/TRAINING_ARGPARSE.pkl', 'wb') as g:
        pickle.dump(args, g)
        
    args = vars(args)
    
    runname = args['training_wkdir']
    ignore_keys = ['interms_for_tboard', 
                   'save_arrs', 
                   'histogram_output_freq',
                   'anc_enc_config', 
                   'desc_dec_config', 
                   'pred_config',
                   'optimizer_config', 
                   'tboard_dir',
                   'model_ckpts_dir',
                   'logfile_dir',
                   'logfile_name',
                   'out_arrs_dir']
    
    ### write general table
    keys_to_keep = [key for key in list( args.keys() ) if key not in ignore_keys] 
    
    with open(f'{out_dir}/CONFIG-TABLE.tsv','w') as g:
        for key in keys_to_keep:
            g.write(f'{key}\t')
            g.write(f'{args[key]}\n')
        
    
    ### write individual tables
    def write_indv_table(key, prefix):
        sub_dict = args.get(key, None)
        
        if sub_dict is not None:
            sub_dict = flatten_convert(sub_dict)
            
            with open(f'{out_dir}/{prefix}.tsv','w') as g:
                g.write('training_wkdir' + '\t' + runname + '\n')
                for key, val in sub_dict.items():
                    g.write(f'{key}\t')
                    g.write(f'{val}\n')
        
    
    write_indv_table('optimizer_config', 'OPTIM-CONFIG')
    write_indv_table('anc_enc_config', 'ANC-ENC-CONFIG')
    write_indv_table('desc_dec_config', 'DESC-DEC-CONFIG')
    write_indv_table('pred_config', 'PRED-CONFIG')
    