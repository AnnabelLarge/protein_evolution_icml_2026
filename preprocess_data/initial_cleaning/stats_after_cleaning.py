#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm


def pfam_level_metadata(seed_alignment_dir: str,
                        pfam: str):
    """
    get pfam-level metadata, including:
        - clan: clan that pfam belongs in (if any)
        - type: type of pfam (domain, family, etc)
        - width: width of the MSA
        - depth: how many sequences per pfam
        - percent_gaps: out of every charcter in the MSA, how many
                       are gap chars? (NOT normalized by any particular
                       sequence length)
    
    inputs:
    -------
        - seed_alignment_dir: where pfam MSA seed alignments are
        - pfam: name of the PFam (PF#####)
    
    returns:
    --------
        - out_dict: PFam-level metadata
    """
    
    out_dict = {'pfam': pfam}
    msa_width_found = False
    seq_count = 0
    char_count = 0
    gaps_count = 0
    with open(f'{seed_alignment_dir}/{pfam}.seed', 'r') as f:
        for line in f:
            if line.startswith('#=GF CL'):
                clan_name = line.strip().split()[-1]
                out_dict['clan_name'] = clan_name
            
            elif line.startswith('#=GF TP'):
                pfam_type = line.strip().split()[-1]
                out_dict['type'] = pfam_type
            
            elif not line.startswith('#'):
                gapped_seq = line.strip().split()[-1]
                gapped_seq = gapped_seq.upper()
                
                # MSA depth
                seq_count += 1
                
                # MSA width
                if not msa_width_found:
                    out_dict['msa_width'] = len(gapped_seq)
                    msa_width_found = True
                
                # gappiness
                char_count += len(gapped_seq)
                gaps_count += gapped_seq.count('.')
    
    out_dict['msa_depth'] = seq_count
    out_dict['percent_gaps'] = gaps_count / char_count
    out_dict['clan_name'] = out_dict.get('clan_name','')
    out_dict['type'] = out_dict.get('type','')
    
    return out_dict


def serially_pfam_level_metadata(pfam_seed_file: str,
                                 seed_alignment_dir: str):
    """
    use pfam_level_metadata() on all files in a folder
    
    inputs:
    -------
        - pfam_seed_file: the original seed file (for figuring out prefix)
        - seed_alignment_dir: where seed files are located
    
    returns:
    --------
        - all_meta: the dataframe of stats
    
    outputs:
    --------
        - pfam_level_metadata_file: f'{prefix}_PFAM-METADATA.tsv
    """
    # prefix = pfam_seed_file.split('.')[0]
    # pfam_level_metadata_file = f'{prefix}_PFAM-METADATA.tsv'
    
    pfam_lst = [file.replace('.seed','') for file in os.listdir(seed_alignment_dir)
                if file.startswith('PF') and file.endswith('.seed')]

    all_meta = []
    for pfam in tqdm(pfam_lst):
        out_dict = pfam_level_metadata(seed_alignment_dir = seed_alignment_dir,
                                       pfam = pfam)
        all_meta.append(out_dict)
        del out_dict
        
    all_meta = pd.DataFrame(all_meta)
    return all_meta


def clan_level_metadata(pfam_seed_file: str):
    """
    get clan-level metadata, including:
        - total pfams
        - total sequences
        - list of pfams in each clan
    
    inputs:
    -------
        - pfam_seed_file: the original seed file (for figuring out prefix)
    
    returns:
    --------
        - clan_metadata: the dataframe of stats
    
    outputs:
    --------
        - clan_level_metadata_file: f'{prefix}_CLAN-METADATA.tsv
    
    """
    prefix = pfam_seed_file.split('.')[0]
    pfam_level_metadata_file = f'{prefix}_PFAM-METADATA.tsv'
    # clan_level_metadata_file = f'{prefix}_CLAN-METADATA.tsv'
    
    path = "/".join( prefix.split('/')[:-1] )
    pfam_level_metadata_file_without_path = pfam_level_metadata_file.split('/')[-1]
    err = f'{pfam_level_metadata_file} not found!'
    assert pfam_level_metadata_file_without_path in os.listdir(path), err
    del path, pfam_level_metadata_file_without_path
    
    df = pd.read_csv(pfam_level_metadata_file, sep='\t', index_col = 0)
    df = df.fillna('')
    clan_counts = dict(Counter(df['clan_name']))
    if '' in clan_counts.keys():
        del clan_counts['']
    
    # how big are the clans?
    clan_metadata = []
    for clan in clan_counts.keys():
        sub_df = df[df['clan_name'] == clan]
        num_seqs = sub_df['msa_depth'].sum()
        num_pfams = len(sub_df)
        
        out_dict = {'clan_name': clan,
                    'num_pfams': num_pfams,
                    'num_seqs': num_seqs,
                    'pfams': '; '.join( sub_df['pfam'].tolist() ) 
                    }
        clan_metadata.append(out_dict)
    
    clan_metadata = pd.DataFrame(clan_metadata)
    return clan_metadata

