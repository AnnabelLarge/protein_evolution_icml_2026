#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ABOUT:
=======
preprocess pfam data

"""
import sys
import argparse
import subprocess

from initial_cleaning.initial_cleaning import main as initial_cleaning_fn
from prepare_for_featurization.prepare_for_featurization import main as split_n_pick 
from generate_inputs.make_features import main as make_features
from generate_inputs.precalculate_counts_for_pairHMM import precalculate_counts_for_pairHMM
from concatenate_parts.concatenate_parts import main as concat_parts
from utils.utils import make_sub_folder

def main():
    parser = argparse.ArgumentParser(
                        prog='data_preproc',
                        description='Preprocess data into cherries')
    
    parser.add_argument('-pfam_seed_file',
                        required=True,
                        type = str,
                        help = '(str) Name of the original single seed file; if in a folder, provide the path too')
    
    parser.add_argument('-tree_dir',
                        required=True,
                        type = str,
                        help = '(str) the folder of .tree files from PFam+FastTree; if in a folder, provide the path too')
    
    parser.add_argument('-num_splits',
                        type = int,
                        default = 10,
                        help = '(int) number of splits (not including OOD valid)')
    
    parser.add_argument('-metadata_header',
                        type = str,
                        default = 'metadata',
                        help = '(str) Header to add to output stats file')
    
    parser.add_argument('-rand_key',
                        type = int,
                        default = 6,
                        help = '(int) random key for randomly selecting data splits')
    
    parser.add_argument('-topk1_valid',
                        type = int,
                        default = 3,
                        help = '(int) number of widest pfams for OOD valid')
    
    parser.add_argument('-topk2_valid',
                        type = int,
                        default = 8,
                        help = '(int) number of gappiest pfams for OOD valid')
    
    parser.add_argument('-alphabet_size',
                        type=int,
                        default=20,
                        help ='(int) base alphabet size; 20 for amino acids')
    
    parser.add_argument('-max_len',
                        type=int,
                        default=5000,
                        help ='(int) maximum length to pad all inputs to')
    
    parser.add_argument('-batch_size',
                        type=int,
                        default=1000,
                        help ='(int) when precalculating event counts, whats the batch size to do so')

    args = parser.parse_args()

    # 1.) clean
    initial_cleaning_fn(pfam_seed_file = args.pfam_seed_file,
                        tree_dir = args.tree_dir,
                        header = args.metadata_header)
    
    # 2.) split into cherries
    split_n_pick(pfam_seed_file = args.pfam_seed_file,
                  tree_dir = args.tree_dir,
                  num_splits = args.num_splits,
                  rand_key = args.rand_key,
                  topk1_valid = args.topk1_valid,
                  topk2_valid = args.topk2_valid)
    
    # 3.) make features (not including summary counts)
    cherries_folder = 'CHERRIES-FROM_' + args.tree_dir.replace('/trees','')
    make_features(num_splits = args.num_splits,
                  max_len = args.max_len,
                  seed_folder = 'seed_alignments',
                  trees_folder = args.tree_dir,
                  cherries_folder = cherries_folder)
    
    # 4.) precalculate counts; this can be slow
    precalculate_counts_for_pairHMM(splitname = 'CHERRIES_valid', 
                                    batch_size = args.batch_size)
    for i in range(args.num_splits):
        precalculate_counts_for_pairHMM(splitname = f'CHERRIES_split{i}',
                                        batch_size = args.batch_size)
    
    # 5.) concatenate everything (per folder)
    concat_parts(splitname = 'CHERRIES_valid',
                  alphabet_size = args.alphabet_size)
    
    for i in range(args.num_splits):
        concat_parts(splitname = f'CHERRIES_split{i}',
                      alphabet_size = args.alphabet_size)
    
    # 6.) clean up
    subprocess.run(["bash", "tear_down.sh"], check=True)


if __name__ == '__main__':
    main()
    
    
    # # example inputs
    # args.pfam_seed_file = 'EXAMPLE_INPUTS/EXAMPLE_Pfam-A.seed'
    # args.tree_dir = 'EXAMPLE_INPUTS/trees'
    # args.num_splits = 2
    # args.metadata_header = 'header'
    # args.rand_key = 42
    # args.topk1_valid = 0
    # args.topk2_valid = 0
    # args.alphabet_size = 20
    # args.max_len = 5000
    # args.batch_size = 10
    