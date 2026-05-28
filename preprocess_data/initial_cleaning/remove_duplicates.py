#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import os
from tqdm import tqdm

from utils.utils import (make_orig_folder, 
                   move_file_to_originals, 
                   rename_file_in_place)


def find_repeats(seed_alignment_dir: str):
    """
    find names and sequences of repeats WITHIN and ACROSS seed files
    
    Will iterate through every line of every seed file, so this might
      take a while...
     
    inputs:
    -------
        - seed_alignment_dir (str): where the .seed files are
    
    returns:
    --------
        (None)
    
    outputs:
    --------
        - repeats_ACROSS_families.tsv: names and sequences of repeats across 
                                       multiple pfams
        - repeats_ACROSS_families.tsv: names and sequences of repeats within 
                                       single pfam
    """
    ### a container for all the within-family repeats
    # key = prot seq (ungapped)
    # value = pfam/seqname
    RAW_repeats = {}
    seen_seqs = set()
    
    ### scan all msa files for repeats
    for idx,filename in tqdm(enumerate(os.listdir('seed_alignments'))):            
        if filename.endswith('.seed'):    
            pfam_name = filename.replace('.seed','')
            
            with open(f'./seed_alignments/{filename}','r', 
                      encoding='latin') as f:
                for line in f:
                    if not line.startswith('#'):
                        samp_name, gapped_seq = line.strip().split()
                        seq = gapped_seq.replace('.','')
                        valname = f'{pfam_name}:{samp_name}'
                        
                        # check if ungapped sequence is seen WITHIN this MSA
                        if seq in seen_seqs:
                            RAW_repeats[seq].append(valname)
                        else:
                            RAW_repeats[seq] = [valname]
                            seen_seqs.add(seq)
    
    
    ### remove anything that doesn't repeat
    repeats = {}
    for key, val_lst in RAW_repeats.items():
        if len(val_lst) > 1:
            repeats[key] = val_lst
    
    # remove the raw dict and the intermediate set
    del RAW_repeats, seen_seqs
    
    
    ### keep repeats across families in their own dictionary
    ###   otherwise, place in repeats within familities
    repeats_within_fams = {}
    repeats_across_fams = {}
    
    for key, val_lst in repeats.items():
        pfams_in_lst = set()
        for entry in val_lst:
            pfam, _ = entry.split(':')
            pfams_in_lst.add(pfam)
        
        if len(pfams_in_lst) > 1:
            repeats_across_fams[key] = val_lst
        
        else:
            repeats_within_fams[key] = val_lst
    
    del repeats
    
    
    ### separately output these
    if len(repeats_within_fams) > 0:
        with open('repeats_WITHIN_families.tsv', 'w') as g:
            for key, val_lst in repeats_within_fams.items():
                pfam_name = val_lst[0].split(':')[0]
                val_lst_without_pfam = ';'.join([elem.split(':')[1] for elem in val_lst])
                
                g.write(f'{pfam_name}\t{val_lst_without_pfam}\t{key}\n')
    
    if len(repeats_across_fams) > 0:
        with open('repeats_ACROSS_families.tsv', 'w') as g:
            for key, val_lst in repeats_across_fams.items():
                to_write = ';'.join(val_lst)
                g.write(f'{to_write}\t{key}\n')



def parse_within_families_file():
    """
    read "repeats_WITHIN_families" to figure out which repeats to remove
    
    inputs:
    -------
        (None)
    
    returns:
    --------
        - to_remove_dict: dictionary of pfam values to remove
    """
    # don't do anything if you don't generate this file
    if 'repeats_WITHIN_families.tsv' not in os.listdir():
        return dict()
    
    all_pfams = []
    all_samp_names = []
    with open('repeats_WITHIN_families.tsv','r') as f:
        for line in f:
            pfam, samp_names, _ = line.strip().split('\t')
            all_pfams.append(pfam)
            all_samp_names.append(samp_names.split(';'))
    
    # build dictionary from lists
    to_remove_dict = {}
    for i in range(len(all_pfams)):
        pfam = all_pfams[i]
        all_duplicates = all_samp_names[i]
        
        # only keep the first instance
        remove_samps = all_duplicates[1:]
        
        if pfam in to_remove_dict.keys():
            to_remove_dict[pfam] = to_remove_dict[pfam] + remove_samps
        elif pfam not in to_remove_dict.keys():
            to_remove_dict[pfam] = remove_samps
    
    return to_remove_dict
    
    
    
def parse_across_families_file():
    """
    read "repeats_ACROSS_families" to figure out which repeats to remove
    
    inputs:
    -------
        (None)
    
    returns:
    --------
        - to_remove_dict: dictionary of pfam values to remove
    """
    # don't do anything if you don't generate this file
    if 'repeats_ACROSS_families.tsv' not in os.listdir():
        return dict()
    
    to_remove_dict = {}
    with open('repeats_ACROSS_families.tsv', 'r') as f:
        for line in f:
            line = line.strip().split('\t')[0]
            raw_lst = line.split(';')
            
            # only keep the first instance
            remove_samps = raw_lst[1:]
            
            for entry in remove_samps:
                pfam, sample = entry.split(':')
                
                if pfam not in to_remove_dict.keys():
                    to_remove_dict[pfam] = [sample]
                
                elif pfam in to_remove_dict.keys():
                    to_remove_dict[pfam].append(sample)
        
    return to_remove_dict
    
    
def remove_samples(seed_alignment_dir: str,
                   to_remove_dict: dict):
    """
    given a list of pfams and samples to remove, trim samples from seed files
    
    
    inputs:
    -------
        - seed_alignment_dir (str): where the .seed files are
        - to_remove_dict: samples to remove from every pfam
            > keys: pfam
            > values: list of samples to remove
    
    returns:
    --------
        (None)
    
    outputs:
    --------
        - de-duplicated pfam files, new 'originals' folder
        
    """
    ### quit function, if there's nothing to be done
    if len(to_remove_dict) == 0:
        print('No duplicates found')
        return

    # make a folder to store originals, if it doesn't already exist
    make_orig_folder(in_dir = seed_alignment_dir)

    ### start iterating through
    for pfam, to_remove in tqdm(to_remove_dict.items()):
        to_remove = set(to_remove)
        
        # get the filenames
        msa_file = f'{pfam}.seed'
        assert msa_file in os.listdir(seed_alignment_dir), f'{msa_file} missing!'
        
        # open the msa and remove duplicate sequences
        with open(f'./{seed_alignment_dir}/DEDUPED_{msa_file}','w') as g_new:
            with open(f'./{seed_alignment_dir}/{msa_file}','r') as f_msa:
                for line in f_msa:
                    # filter
                    if line.startswith('#'):
                        g_new.write(line)
                    
                    else:
                        this_samp_name = line.split()[0]
                        if this_samp_name not in to_remove:
                            g_new.write(line)
        
        # move the original file, if it's not already there
        move_file_to_originals(filename = msa_file,
                               in_dir = seed_alignment_dir)
        
        # rename the new version (will overwrite original, if that's left 
        #   in the folder)
        rename_file_in_place(filename = msa_file,
                             in_dir = seed_alignment_dir,
                             prefix = 'DEDUPED')
