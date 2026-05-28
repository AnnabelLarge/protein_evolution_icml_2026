#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np


def sort_pfams(folder_name: str):
    all_files = [f for f in os.listdir(folder_name) if
                 f.startswith('PF')]
    
    pfams_sorted = [f.split('_')[0].split('-')[0] for f in all_files]
    pfams_sorted.sort()
    
    return pfams_sorted


def sort_pfam_parts(folder_name: str,
                    pfam: str,
                    file_suffix: str):
    pfam_parts_in_folder = [f for f in os.listdir(folder_name) 
                            if f.startswith(pfam) and f.endswith(file_suffix)]
    num_parts = len(pfam_parts_in_folder)
    
    if num_parts > 1:
        sorted_pfams_prefixes = [f'{pfam}-pt{i}' for i in range(num_parts)]
    
    elif num_parts == 1:
        sorted_pfams_prefixes = [pfam]
    
    elif num_parts == 0:
        raise ValueError(f'No files found for {folder_name}, {pfam}, {file_suffix}')
    
    return sorted_pfams_prefixes


def combine_metadata(splitname: str):
    """
    concatenate the metadata for all training pairs
    
    usually the metadata dataframes for each PFam aren't split into parts...
      so don't worry about that yet?
    
    input:
    ------
        - splitname (str): the folder prefix, usually setname+foldname
    
    returns:
    --------
        - pfams_in_order (list): the sorted list of pfams; use this
                                 to establish the concatenation order for all
                                 other files
    
    outputs:
    --------
        - concatenated metadata about the training set
    """
    folder = splitname+ '_all_metadata'
    
    out_meta = []
    pfams_in_order = sort_pfams(folder_name = folder)
    
    if len(pfams_in_order) > 0:
        for pfam_prefix in pfams_in_order:
            sub_df = pd.read_csv(f'{folder}/{pfam_prefix}_metadata.tsv',
                                 sep='\t',
                                 index_col=0)
            sub_df['original_loc'] = f'{folder}/{pfam_prefix}_metadata.tsv'
            out_meta.append(sub_df)
        
        out_meta = pd.concat(out_meta)
        out_meta = out_meta.reset_index(drop=True)
        out_meta.to_csv(f'{splitname}_metadata.tsv', sep='\t')
    
    return pfams_in_order


def combine_unaligned_inputs(splitname: str,
                          pfams_in_order: list):
    """
    concatenate the unaligned inputs (ungapped ancestor and descendant seqs)
    
    unaligned_seqs_matrix = (B, L_seq, 2)
      - dim2=0: ancestor, unaligned
      - dim2=1: descendant, unaligned
      - L_seq INCLUDES <bos>, <eos>
    {prefix}_seqs_unaligned.npy
    
    
    input:
    ------
        - splitname (str): the folder prefix, usually setname+foldname
        - pfams_in_order (list): the list of file prefixes in order, 
                                 usually pfam+part_id
    
    returns:
    --------
        (None)
    
    outputs:
    --------
        - concatenated inputs 
    """
    folder = splitname + '_full_length'
    
    out_seqs_paths = []
    out_align_idxes = []
    for pfam in pfams_in_order:
        sorted_pfam_in_parts = sort_pfam_parts(folder_name = folder,
                                               pfam = pfam,
                                               file_suffix = '_seqs_unaligned.npy')
        
        for pfam_prefix in sorted_pfam_in_parts:
            with open(f'{folder}/{pfam_prefix}_seqs_unaligned.npy','rb') as f:
                out_seqs_paths.append( np.load(f) )
    
    out_seqs_paths = np.concatenate(out_seqs_paths, axis=0)
    longest_anc, longest_desc = np.where(out_seqs_paths != 0,
                                         True,
                                         False).sum(axis=1).max(axis=0)
    
    with open(f'{splitname}_seqs_unaligned.npy','wb') as g:
        np.save(g, out_seqs_paths)
    
    with open(f'{splitname}_longest_seqs.txt','w') as g:
        g.write(f'split\tlongest_anc\tlongest_desc\n')
        g.write(f'{splitname}\t{longest_anc}\t{longest_desc}\n')
        

def combine_aligned_inputs(splitname: str,
                           pfams_in_order: list):
    """
    concatenate the aligned inputs (ungapped ancestor and descendant seqs)
    
    aligned_seqs_matrix = (B, L_align, 4)
      - dim2=0: ancestor GAPPED (aligned)
      - dim2=1: descendant GAPPED (aligned)
      - dim2=2: precomputed m indexes for neural models
      - dim2=3: precomputed n indexes for neural models
    {prefix}_aligned_mats.npy
    
    
    input:
    ------
        - splitname (str): the folder prefix, usually setname+foldname
        - pfams_in_order (list): the list of file prefixes in order, 
                                 usually pfam+part_id
    
    returns:
    --------
        (None)
    
    outputs:
    --------
        - concatenated inputs 
    """
    folder = splitname + '_full_length'
    
    out_seqs_paths = []
    out_align_idxes = []
    for pfam in pfams_in_order:
        sorted_pfam_in_parts = sort_pfam_parts(folder_name = folder,
                                               pfam = pfam,
                                               file_suffix = '_aligned_mats.npy')
        
        for pfam_prefix in sorted_pfam_in_parts:
            with open(f'{folder}/{pfam_prefix}_aligned_mats.npy','rb') as f:
                out_seqs_paths.append( np.load(f) )
    
    out_seqs_paths = np.concatenate(out_seqs_paths, axis=0)
    longest_aligns = np.where(out_seqs_paths[:,:,0] != 0,
                              True,
                              False).sum(axis=1).max()
    
    
    with open(f'{splitname}_aligned_mats.npy','wb') as g:
        np.save(g, out_seqs_paths)
    
    with open(f'{splitname}_longest_alignment.txt','w') as g:
        g.write(f'split\tlongest_align\n')
        g.write(f'{splitname}\t{longest_aligns}\n')

def combine_hmm_precalc_inputs(splitname: str,
                               pfams_in_order: list,
                               alphabet_size: int = 20):
    """
    concatenate/sum the inputs for EvolPairHMM
    
    input:
    ------
        - splitname (str): the folder prefix, usually setname+foldname
        - pfams_in_order (list): the list of file prefixes in order, 
                                 usually pfam+part_id
        - alphabet_size (int=20): the base alphabet size; 20 for proteins
    
    returns:
    --------
        (None)
    
    outputs:
    --------
        - concatenated/combined inputs for EvolPairHMM
    """
    folder = f'{splitname}_summarized_counts'
    
    all_out = {'subCounts': [],
               'insCounts': [],
               'delCounts': [],
               'transCounts': [],
               'AAcounts': np.zeros( (alphabet_size,) ),
               'AAcounts_subsOnly': np.zeros( (alphabet_size,) )
               }
    
    for pfam in pfams_in_order:
        sorted_pfam_in_parts = sort_pfam_parts(folder_name = folder,
                                               pfam = pfam,
                                               file_suffix = '_transCounts.npy')
        
        for pfam_prefix in sorted_pfam_in_parts:
            # these get concatenated
            with open(f'{folder}/{pfam_prefix}_subCounts.npy','rb') as f:
                all_out['subCounts'].append( np.load(f) )
            
            with open(f'{folder}/{pfam_prefix}_insCounts.npy','rb') as f:
                all_out['insCounts'].append( np.load(f) )
            
            with open(f'{folder}/{pfam_prefix}_delCounts.npy','rb') as f:
                all_out['delCounts'].append( np.load(f) )
                
            with open(f'{folder}/{pfam_prefix}_transCounts.npy','rb') as f:
                all_out['transCounts'].append( np.load(f) )
            
            # these get added
            with open(f'{folder}/{pfam_prefix}_AAcounts.npy','rb') as f:
                all_out['AAcounts'] = all_out['AAcounts'] + np.load(f)
            
            with open(f'{folder}/{pfam_prefix}_AAcounts_subsOnly.npy','rb') as f:
                all_out['AAcounts_subsOnly'] = (all_out['AAcounts_subsOnly'] + 
                                                np.load(f) )
        
    for suffix in ['subCounts','insCounts','delCounts','transCounts']:
        to_write = np.concatenate(all_out[suffix])
        with open(f'{splitname}_{suffix}.npy','wb') as g:
            np.save(g, to_write)
    
    for suffix in ['AAcounts','AAcounts_subsOnly']:
        with open(f'{splitname}_{suffix}.npy','wb') as g:
            np.save(g, to_write)
        


def main(splitname: str,
         alphabet_size: int = 20):
    """
    run the functions above to combine file parts
    
    input:
    ------
        - splitname (str): the folder prefix, usually setname+foldname
        - alphabet_size (int=20): the base alphabet size; 20 for proteins
        - include_pair_align (bool=False): whether or not to concatenate this 
                                           folder of intermediates (usually no)
    
    returns:
    --------
        (None)
    
    outputs:
    --------
        - concatenated/combined inputs for EvolPairHMM and DogShow
    """
    
    # concatenate metadata, and use this to get the file ordering
    pfams_in_order = combine_metadata(splitname = splitname)
    
    # if files are found, concatenate everything else
    if len(pfams_in_order) > 0:
        combine_unaligned_inputs(splitname = splitname,
                             pfams_in_order = pfams_in_order)
    
        combine_aligned_inputs(splitname = splitname,
                               pfams_in_order = pfams_in_order)
        
        
        # assuming you've also precalculated pairHMM inputs, concatenate those too
        combine_hmm_precalc_inputs(splitname = splitname,
                                   pfams_in_order = pfams_in_order,
                                   alphabet_size = alphabet_size)
        
