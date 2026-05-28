#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import os
from tqdm import tqdm

from utils.utils import (make_orig_folder, 
                   move_file_to_originals, 
                   rename_file_in_place)


INVALID_CHARS = set(['B','O','U','X','Z'])
SHORT_PEP_THRESHOLD = 10


def clean_short_peps_invalid_chars(seed_alignment_dir: str):
    """
    Remove any (ungapped) sequences that are less than 10 amino acids or
      contain invalid characters
    
    Slow because you have to loop through all the seed alignments 
      line-by-line again...
    
    inputs:
    -------
        - seed_alignment_dir (str): where the .seed files are
    
    returns:
    --------
        (None)
    
    outputs:
    --------
        - SHORT-PEPS_INVALID-CHARS.tsv: anything removed at this step
    """
    # record which pfams and samples you removed at this step
    # key = pfam
    # value = list of samples taken out
    removed = {}
    removed_whole_pfam = []
    
    ### start by looping through MSA files
    for idx,msa_file in tqdm( enumerate(os.listdir('seed_alignments')) ):
        
        # do this for every individual pfam
        if (msa_file.startswith('PF')) and (msa_file.endswith('.seed')):
        
            pfam_name = msa_file.replace('.seed','')
            
            
            ### 1.) scan through the MSA to find short peptides, invalid chars 
            samples_to_remove = []
            new_content_to_write = []
            num_seqs_to_write = 0
            with open(f'./seed_alignments/{msa_file}','r', 
                      encoding='latin') as f_msa:
                for line in f_msa:
                    # automatically write all annotation lines to new file
                    if line.startswith('#'):
                        new_content_to_write.append(line)
                    
                    # aligned sequences are on these lines
                    elif not line.startswith('#'):
                        samp_name, gapped_seq = line.strip().split()
                        seq = gapped_seq.replace('.','').upper()
                        
                        cond1 = len(seq) < SHORT_PEP_THRESHOLD
                        cond2 = any( [char in seq for char in INVALID_CHARS] )
                        
                        if cond1 or cond2:
                            samples_to_remove.append(samp_name)
                        
                        else:
                            new_content_to_write.append(line.upper())
                            num_seqs_to_write += 1
                            
                            
            ### 2.) ONLY DO THESE ACTIONS IF YOU NEED TO REWRITE THE SEED FILE
            cond1 = ( len(samples_to_remove) > 0 )
            cond2 = ( num_seqs_to_write == 1 )
            
            ### first check: if cond2 is ever activated, you'll remove the 
            ###   whole pfam
            if cond2:
                removed_whole_pfam.append(pfam_name)
                        
            ### second check: figure out why you have to remove sequences, if 
            ###   at all
            # if this is true, you found sequences to remove b/c too short or 
            #   had invalid chars
            if cond1:
                make_orig_folder(in_dir = seed_alignment_dir)
                removed[pfam_name] = samples_to_remove
                
                # move the original file, if it's not already there
                move_file_to_originals(filename = msa_file, 
                                       in_dir = seed_alignment_dir)
                  
                # if there's more than one sequence to write, output new .seed 
                #   file
                if not cond2:
                    new_msa_file = f'./seed_alignments/CLEANED_{msa_file}'
                    with open(new_msa_file,'w') as g_newmsa:
                        [g_newmsa.write(elem) for elem in new_content_to_write]
                    
                    # rename the new version (will overwrite original, if that's left 
                    #   in the folder)
                    rename_file_in_place(filename = msa_file,
                                         in_dir = seed_alignment_dir,
                                         prefix = 'CLEANED')
               
            # no short or invalid char sequences found, but original MSA has 
            #  only one sequence; move it to "originals"
            elif cond2:
                make_orig_folder(in_dir = seed_alignment_dir)
                move_file_to_originals(filename = msa_file, 
                                       in_dir = seed_alignment_dir)
            
    
    ### record anything removed
    removed_pfams = []
    
    if len(removed) > 0:
        with open(f'SHORT-PEPS_INVALID-CHARS.tsv','w') as g:
            for key, val in removed.items():
                removed_pfams.append(key)
                val_as_str = ';'.join(val)
                g.write(f'{key}\t{val_as_str}\n')
    
    if len(removed_whole_pfam) > 0:
        with open(f'REMOVED-PFAMS.tsv','w') as g:
            [g.write(elem + '\n') for elem in removed_whole_pfam]
    
    return removed_pfams + removed_whole_pfam
        
    
            