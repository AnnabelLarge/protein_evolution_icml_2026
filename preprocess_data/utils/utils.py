#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os


def find_missing_info(seed_alignment_dir: str,
                      tree_dir: str):
    """
    find if pfams are missing either tree or MSA seed alignment files
     
    inputs:
    -------
        - seed_alignment_dir: contains seed alignments
        - tree_dir: contains trees
     
    returns:
    --------
    """
    pfams_in_msas = set( [f.replace('.seed','') for f in os.listdir(seed_alignment_dir) 
                          if f.startswith('PF') and f.endswith('.seed')] )
    pfams_in_trees = set( [f.replace('.tree','') for f in os.listdir(tree_dir)
                           if f.startswith('PF') and f.endswith('.tree')] )
    
    missing_trees = pfams_in_msas - pfams_in_trees
    missing_msas = pfams_in_trees - pfams_in_msas
    return list(missing_trees), list(missing_msas)
    

def rename_file_in_place(filename: str,
                         in_dir:str,
                         prefix: str):
    """
    rename a file in place
    
    inputs:
    -------
        - filename: name of file (usually pfam.seed or pfam.tree)
        - in_dir: name of folder
        - prefix: the temporary prefix 
    
    returns:
    --------
        (None)
        
    outputs:
    --------
        - renames file in place
    """
    os.rename(f'./{in_dir}/{prefix}_{filename}',
              f'./{in_dir}/{filename}')


def move_file_to_originals(filename: str, 
                           in_dir: str):
    """
    move the file from in_dir to in_dir/originals
    
    inputs:
    -------
        - filename: name of file (usually pfam.seed or pfam.tree)
        - in_dir: name of folder
    
    returns:
    --------
        (None)
        
    outputs:
    --------
        - new directory at {in_dir}/originals
    """
    if filename not in os.listdir(f'{in_dir}/originals'):
        os.rename(f'./{in_dir}/{filename}',
                  f'./{in_dir}/originals/{filename}')


def make_sub_folder(in_dir, sub_folder):
    """
    try making a folder inside in_dir
    
    inputs:
    -------
        - in_dir: name of directory to contain sub_folder
        - sub_folder: new sub_folder
    
    returns:
    --------
        (None)
        
    outputs:
    --------
        - new directory at {in_dir}/{sub_folder}
    """
    if sub_folder not in os.listdir(in_dir):
        os.mkdir(f'{in_dir}/{sub_folder}')


def make_orig_folder(in_dir: str):
    make_sub_folder(in_dir = in_dir, 
                    sub_folder = 'originals') 
    

def msa_dimensions(pfam_seed_file: str):
    """
    Open a pfam seed file and count the width and depth
    
    Doesn't depend on info in given PFam annotation line, since sequences
      could be removed during processing
    
    inputs:
    -------
        - pfam_seed_file: the single Pfam seed file to split
    
    returns:
    --------
        - out_dict: dictionary of MSA dimensions
    """
    pfam_name = pfam_seed_file.split('/')[-1].replace('.seed','')
    msa_width = -1
    num_seqs = 0
    with open(pfam_seed_file, 'r', encoding='latin') as f:
        for line in f:
            if not line.startswith('#'):
                num_seqs += 1
            
                if msa_width == -1:
                    msa_width = len( line.strip().split()[-1] )
            
    out_dict = {'name': pfam_name,
                'num_seqs': num_seqs,
                'msa_width': msa_width}
    
    return out_dict

