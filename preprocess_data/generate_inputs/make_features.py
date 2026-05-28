#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from prepare_for_featurization.prepare_for_featurization import main as split_n_pick 
from generate_inputs.generate_inputs import samples_from_file
from utils.utils import make_sub_folder

def main(num_splits,
         max_len,
         seed_folder,
         trees_folder,
         cherries_folder):
    split_file_lst = ['pfams_in_OOD_valid.tsv'] + [f'pfams_in_split{i}.tsv' for i in range(num_splits)]
    for pfam_filename in split_file_lst:
        splitname = pfam_filename.replace('.tsv','').split('_')[-1]
        cherries_dset_prefix = f'CHERRIES_{splitname}'
            
        with open(f'{pfam_filename}','r') as f:
            pfams_in_split = [line.strip() for line in f]
        
        file_lst = [f'{cherries_folder}/{pf}_cherries.tsv' 
                    for pf in pfams_in_split]
        
        for suffix in ['full_length', 'summarized_counts', 'all_metadata']:
            this_split_cherries_folder = f'{cherries_dset_prefix}_{suffix}'
            make_sub_folder(in_dir = '.', sub_folder=this_split_cherries_folder)
        
        for i in range(len(pfams_in_split)):
            if i % 10 == 0:
                print(f'{i}/{len(pfams_in_split)}')
                
            pfam = pfams_in_split[i]
            file_of_cherries = file_lst[i]
            
            samples_from_file(pfam = pfam, 
                              seed_folder = seed_folder,
                              trees_folder = trees_folder,
                              filename = file_of_cherries,
                              dset_prefix = cherries_dset_prefix,
                              max_len = max_len)
            