#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

from initial_cleaning.split_seed_file import split_seed_file
from initial_cleaning.remove_duplicates import (find_repeats,
                                                parse_within_families_file,
                                                parse_across_families_file,
                                                remove_samples)
from initial_cleaning.clean_short_peps_invalid_chars import clean_short_peps_invalid_chars
from initial_cleaning.prune_trees_after_msa_clean import prune_trees_after_msa_clean
from initial_cleaning.stats_after_cleaning import (serially_pfam_level_metadata,
                                                   clan_level_metadata)
from utils.utils import find_missing_info


def main(pfam_seed_file: str,
         header: str,
         seed_alignment_dir: str = 'seed_alignments',
         tree_dir: str = 'trees'):
    """
    after downloading pfam seed alignments file, clean and split into .seed
      and .tree files per acceptable PFam
      
    inputs:
    -------
        - pfam_seed_file: the single Pfam seed file to split
        - header: information about the general pfam set
        - seed_alignment_dir (str): where individual .seed files should go
        - tree_dir (str): where individual .tree files should go
    
    returns:
    --------
        (None)
    
    outputs:
    --------
        - cleaned .seed and .tree files per pfam, ready to be split
    """
    ### split the pfam seed file
    split_seed_file(seed_alignment_dir = seed_alignment_dir,
                    pfam_seed_file = pfam_seed_file,
                    header = header)
    
    
    ### remove repeated sequences within and across families
    find_repeats(seed_alignment_dir = seed_alignment_dir)
    dict1 = parse_within_families_file()
    dict2 = parse_across_families_file()
    remove_samples(seed_alignment_dir = seed_alignment_dir, 
                   to_remove_dict = dict1)
    remove_samples(seed_alignment_dir = seed_alignment_dir, 
                   to_remove_dict = dict2)
    
    # start recording what pfams are removed
    removed_pfams = list(dict1.keys()) + list(dict2.keys())
    del dict1, dict2
    
    
    ### remove short peptides, sequences with invalid chars
    removed_this_step = clean_short_peps_invalid_chars(seed_alignment_dir = seed_alignment_dir)
    removed_pfams = removed_pfams + removed_this_step
    del removed_this_step
    
    
    ### whatever samples/pfams were removed, do the same with the trees 
    prune_trees_after_msa_clean(tree_dir)
    
    
    ### last check for missing samples
    missing_trees, missing_msas = find_missing_info(seed_alignment_dir = seed_alignment_dir,
                                                    tree_dir = tree_dir)
    
    err_msg = f'Have tree files without matching seed alignments?\n{missing_msas}'
    assert len(missing_msas) == 0, err_msg

    # if any pfams were permanently removed, remove from missing_trees;
    #   won't need to calculate these trees!    
    missing_trees = list( set(missing_trees) - set(removed_pfams) )
    
    # these need to manually be aligned in FastTree
    if len(missing_trees) > 0:
        with open(f'ALIGN_IN_FASTTREE.tsv', 'w') as g:
            [g.write(elem + '\n') for elem in missing_trees]
    
    
    ### get finer-grained stats after cleaning
    prefix_for_files = pfam_seed_file.split('.')[0]

    # pfam-level metadata
    pfam_meta_df = serially_pfam_level_metadata(pfam_seed_file = pfam_seed_file,
                                                seed_alignment_dir = seed_alignment_dir)
    pfam_meta_df = pfam_meta_df[~pfam_meta_df['pfam'].isin(removed_pfams)]
    pfam_level_metadata_file = f'{prefix_for_files}_PFAM-METADATA.tsv'  
    pfam_meta_df.to_csv(pfam_level_metadata_file, sep='\t')
    del pfam_level_metadata_file
    
    # clan-level metadata
    clan_meta_df = clan_level_metadata(pfam_seed_file = pfam_seed_file)
    for bad_pfam in removed_pfams:
        clan_meta_df = clan_meta_df[~clan_meta_df['pfams'].str.contains(bad_pfam)]
    clan_level_metadata_file = f'{prefix_for_files}_CLAN-METADATA.tsv'
    clan_meta_df.to_csv(clan_level_metadata_file, sep='\t')
    del clan_level_metadata_file
    
    # write the final stats 
    out_dict = {'Number of Pfams': len(pfam_meta_df),
                'Number of Pfams in clans': clan_meta_df['num_pfams'].sum(),
                'Number of Unique clans': len(clan_meta_df)}
    
    with open(f'{prefix_for_files}_STATS-AFTER-CLEANING.tsv', 'w') as g:
        g.write(f'{header}\n')
        [g.write(f'{key}\t{val}\n') for key, val in out_dict.items()]
    
    with open(f'{prefix_for_files}_ALL-REMOVED-PFAMS.tsv','w') as g:
        [g.write(elem + '\n') for elem in removed_pfams]
