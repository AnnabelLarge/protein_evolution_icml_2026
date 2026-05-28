#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re
import os
import pandas as pd

def split_seed_file(seed_alignment_dir: str,
                    pfam_seed_file: str,
                    header: str):
    """
    Pfam seed alignments are in one giant file, split into per-pfam 
      alignment files
    
    inputs:
    -------
        - seed_alignment_dir (str): where individual .seed files should go
        - pfam_seed_file: the single Pfam seed file to split
        - header: information to add at the top of the stats file
    
    returns:
    --------
        (None)
    
    outputs:
    --------
        - f'{prefix}_INIT-STATS.tsv': some initial stats after splitting
    """
    # make the folder
    if seed_alignment_dir not in os.listdir():
        os.mkdir(seed_alignment_dir)
    
    # split the contents of the main seed file
    with open(pfam_seed_file,'r', encoding='latin') as f:
        contents = [entry for entry in f.read().split('//\n') if entry != '']
    
    # parse through the contents, gather stats, and output
    num_pfams = len(contents)
    num_pfams_with_clans = 0
    seen_clans = set()
    for entry in contents:
        match = re.search(rf"^.*#=GF AC.*$", entry, re.MULTILINE)
        line_with_pfam_name = match.group(0) if match else None
        pfam = line_with_pfam_name.strip().split()[2].split('.')[0]
        del match
        
        match = re.search(rf"^.*#=GF CL.*$", entry, re.MULTILINE)
        line_with_clan_name = match.group(0) if match else None
        if line_with_clan_name != None:
            num_pfams_with_clans += 1
            
            clan = line_with_clan_name.strip().split()[-1]
            if clan not in seen_clans:
                seen_clans.add(clan)
                
        with open(f'./{seed_alignment_dir}/{pfam}.seed','w') as g:
            g.write(entry)
    
    # write the final stats 
    out_dict = {'Number of Pfams': num_pfams,
                'Number of Pfams in clans': num_pfams_with_clans,
                'Number of Unique clans': len(seen_clans)}
    
    prefix = pfam_seed_file.split('.')[0]
    
    with open(f'{prefix}_INIT-STATS.tsv', 'w') as g:
        g.write(f'{header}\n')
        [g.write(f'{key}\t{val}\n') for key, val in out_dict.items()]
