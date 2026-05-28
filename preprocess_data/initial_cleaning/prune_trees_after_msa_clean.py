#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from Bio import Phylo

from initial_cleaning.remove_duplicates import (parse_within_families_file,
                                                parse_across_families_file)
from utils.utils import (make_orig_folder,
                   move_file_to_originals,
                   rename_file_in_place)


def prune_trees_after_msa_clean(tree_dir: str):
    """
    After cleaning up MSAs, remove whole trees (or samples from trees)
    
    inputs:
    -------
        - tree_dir: directory of trees
    
    returns:
    --------
        (None)
        
    outputs:
    --------
        - renames/moves tree files in place
    """
    ### combine results from all intermediate files
    if 'REMOVED-PFAMS.tsv' in os.listdir():
        with open(f'REMOVED-PFAMS.tsv','r') as f:
            pfams_to_remove = [line.strip() for line in f]
    
    remove_some_samps = {}
    pt1_dict = parse_within_families_file()
    pt2_dict = parse_across_families_file()
    pt3_dict = dict()
    if 'SHORT-PEPS_INVALID-CHARS.tsv' in os.listdir():
        with open('SHORT-PEPS_INVALID-CHARS.tsv','r') as f:
            for line in f:
                pfam, raw_lst = line.strip().split('\t')
                lst = raw_lst.split(';')
                pt3_dict[pfam] = lst
    all_keys = list( set( list( pt1_dict.keys() ) + 
                          list( pt2_dict.keys() ) + 
                          list( pt3_dict.keys() ) 
                          ) 
                    )
    for key in all_keys:
        if key not in pfams_to_remove:
            remove_some_samps[key] = (pt1_dict.get(key, []) +
                                      pt2_dict.get(key, []) +
                                      pt3_dict.get(key, []) )
    
    if (len(pfams_to_remove) > 0) or (len(remove_some_samps) > 0):
        make_orig_folder(tree_dir)
        
    
    ### if pfams were removed entirely, move those tree files
    [move_file_to_originals(filename = f'{pfam}.tree', in_dir = tree_dir) 
     for pfam in pfams_to_remove]
    
    
    ### otherwise, comb through and prune trees
    for pfam, to_remove in remove_some_samps.items():
        tree = Phylo.read(f'./{tree_dir}/{pfam}.tree', format='newick')
        avail_leafs = [cl.name for cl in tree.get_terminals()]
        
        for leaf in to_remove:
            if leaf in avail_leafs:
                tree.prune(leaf)
            
        Phylo.write(tree, f'./{tree_dir}/CLEANED_{pfam}.tree', format='newick')
        
        move_file_to_originals(filename = f'{pfam}.tree', 
                               in_dir = tree_dir)
        
        rename_file_in_place(filename = f'{pfam}.tree', 
                             in_dir = tree_dir,
                             prefix = 'CLEANED')
