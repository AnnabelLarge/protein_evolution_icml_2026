#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'



def make_splits(pfam_seed_file: str,
                num_splits: int = 10,
                rand_key: int = 6,
                topk1_valid: int = 3,
                topk2_valid: int = 8):
    """
    split pfams into OOD validation set + {num_splits} folds
    
    when I made splits:
        num_splits = 10
        rand_key = 6
        topk1_valid = 3
        topk2_valid = 8
    
    inputs:
    -------
        - pfam_seed_file (str): the original seed file; use for getting file 
                                prefix
        - num_splits (int): number of folds for in-distribution train+test set
        - rand_key (int): random key for numpy
        - topk1_valid (int): when making OOD Valid set, how many of the 
                             widest pfams to hold out 
        - topk2_valid (int): when making OOD Valid set, how many of the 
                             widest pfams to hold out
    
    returns:
    --------
        (None)
    
    outputs:
    --------
        - split_pfam_meta_file: a version of the previous pfam_level_metadata_file,
                                but with split information added
        - split_metadata_file: metadata about the splits
        - plain text files for each fold, with names of pfams in the folds
    """
    prefix = pfam_seed_file.split('.')[0]
    pfam_level_metadata_file = f'{prefix}_PFAM-METADATA.tsv'
        
    df = pd.read_csv(pfam_level_metadata_file, 
                     sep='\t', 
                     index_col=0)
    
    def num_pairs(n):
        return (n * (n-1) ) / 2
    
    df['possible_pairs'] = num_pairs(df['msa_depth'])
    
    
    ############################
    ### PFAMS WITH CLAN LABELS #
    ############################
    pfams_with_clans = df[~df['clan_name'].isna()]
    
    def add_stat(colname, fn, new_lab):
        clans_to_stat = {}
        for clan in list(set(pfams_with_clans['clan_name'])):
            subdf = pfams_with_clans[pfams_with_clans['clan_name'] == clan]
            stat = fn(subdf[colname])
            clans_to_stat[clan] = stat
            del subdf, stat, clan
        
        pfams_with_clans[new_lab] = pfams_with_clans['clan_name'].apply(lambda x: clans_to_stat[x])
    
    
    ### clan stats
    # sizes
    add_stat(colname = 'possible_pairs',
             fn = sum,
             new_lab = 'CLAN_possible_pairs')
    
    # add average MSA width for the clan
    add_stat(colname = 'msa_width',
             fn = lambda x: x.mean(),
             new_lab = 'CLAN-AVE_msa_width')
    
    
    # add average percent indels for the clan
    add_stat(colname = 'percent_gaps',
             fn = lambda x: x.mean(),
             new_lab = 'CLAN-AVE_percent_gaps')
    
    
    # clan-level dataframe; remove any particularly large clans
    col_lst = ['clan_name',
               'CLAN_possible_pairs',
               'CLAN-AVE_msa_width', 
               'CLAN-AVE_percent_gaps']
    clan_meta = pfams_with_clans.drop_duplicates(subset = 'clan_name', 
                                                 keep = 'first')[col_lst]
    clan_meta = clan_meta[clan_meta['CLAN_possible_pairs'] < 800000]
    
    # widest MSAs
    widest = clan_meta.nlargest(n=topk1_valid, 
                                columns='CLAN-AVE_msa_width', 
                                keep='all')
    
    # gappiest MSAs
    gappiest =  clan_meta.nlargest(n=topk2_valid, 
                                   columns='CLAN-AVE_percent_gaps', 
                                   keep='all')
    
    
    ##############################
    ### SPLIT OFF VALIDATION SET #
    ##############################
    # how many pairs, if I extract these pfams?
    validation_clans = list(set(widest['clan_name'].tolist() + gappiest['clan_name'].tolist()))
    sub_df = clan_meta[clan_meta['clan_name'].isin(validation_clans)]
    num_validation_pairs = sub_df['CLAN_possible_pairs'].sum()
    perc_data = num_validation_pairs / df['possible_pairs'].sum()
    
    
    validation_pfams = df[df['clan_name'].isin(validation_clans)]['pfam'].tolist()
    remaining = pfams_with_clans[~pfams_with_clans['clan_name'].isin(validation_clans)]
    del col_lst, clan_meta, widest, gappiest, sub_df
    del perc_data
    
    
    #################################################
    ### START SPILTTING PFAMS WITH CLANS INTO FOLDS #
    #################################################
    col_lst = ['clan_name',
                'CLAN_possible_pairs',
                'CLAN-AVE_msa_width', 
                'CLAN-AVE_percent_gaps']
    clan_meta = remaining.drop_duplicates(subset = 'clan_name', 
                                          keep = 'first')[col_lst]
    
    total_samples = clan_meta['CLAN_possible_pairs'].sum()
    max_size = math.ceil(total_samples / num_splits)
    
    train_test_clans = {i: [] for i in range(num_splits)}
    train_test_pfams = {i: [] for i in range(num_splits)}
    split_sizes = {i:0 for i in range(num_splits)}
    
    # shuffle before assigning splits
    clan_meta = clan_meta.sample(frac=1, random_state=rand_key).reset_index(drop=True)
    to_place = []
    split_ids = list(range(num_splits))
    for clan in clan_meta['clan_name']:
        sub_df = clan_meta[clan_meta['clan_name'] == clan]
        pfams = pfams_with_clans[pfams_with_clans['clan_name']==clan]['pfam'].tolist()
        num_pairs = sub_df['CLAN_possible_pairs'].sum()
        
        # try placing in one of num_splits splits
        placed = False
        for i in split_ids:
            if split_sizes[i] < max_size:
                placed = True
                train_test_clans[i].append(clan)
                train_test_pfams[i] += pfams
                split_sizes[i] += num_pairs
                break
        
        if not placed:
            to_place.append(clan)
        
        # rotate list before next iteration
        split_ids = split_ids[1:] + split_ids[:1]
    
    assert len(to_place) == 0
    
    del col_lst, clan_meta, total_samples, max_size, to_place, split_ids
    
    
    ##############################################
    ### START SPILTTING PFAMS WITH NO CLAN LABEL #
    ##############################################
    pfams_without_clans = df[df['clan_name'].isna()]
    
    total_samples = pfams_without_clans['possible_pairs'].sum()
    max_size = math.ceil(total_samples / num_splits)
    
    more_train_test_pfams = {i: [] for i in range(num_splits)}
    more_split_sizes = {i:0 for i in range(num_splits)}
    
    # shuffle before assigning splits
    pfams_without_clans = pfams_without_clans.sample(frac=1, 
                                                     random_state=rand_key).reset_index(drop=True)
    if len(pfams_without_clans) > 0:
        to_place = []
        split_ids = list(range(num_splits))
        for _, row in pfams_without_clans.iterrows():
            pfam_name = row['pfam']
            num_pairs = row['possible_pairs']
            
            # try placing in one of 5 splits
            placed = False
            for i in split_ids:
                if more_split_sizes[i] < max_size:
                    placed = True
                    more_train_test_pfams[i].append( pfam_name )
                    more_split_sizes[i] += num_pairs
                    break
            
            if not placed:
                to_place.append(pfam_name)
            
            # rotate list before next iteration
            split_ids = split_ids[1:] + split_ids[:1]
        
        assert len(to_place) == 0
        del pfams_without_clans, total_samples, max_size, to_place, split_ids, row
        del pfam_name, num_pairs, placed, i
    
    
    #############################################################
    ### COMBINE PFAM AND CLAN LISTS; MAKE FINAL META DATAFRAMES #
    #############################################################
    checksum = 0
    for i in range(num_splits):
        train_test_pfams[i] = train_test_pfams[i]+more_train_test_pfams[i]
        split_sizes[i] = split_sizes[i]+more_split_sizes[i]
        checksum += split_sizes[i]
    
    checksum +=num_validation_pairs
    
    # make sure all pairs are accounted for
    assert checksum == df['possible_pairs'].sum()
    
    # add one-hot label to main dataframe (easier to check for duplication)
    for i in range(num_splits):
        pfams_in_split = train_test_pfams[i]
        one_hot = df['pfam'].isin(pfams_in_split)
        df[f'in_split{i}'] = one_hot
    
    df['in_OOD_valid'] = df['pfam'].isin(validation_pfams)
    
    del checksum, i, train_test_pfams, split_sizes, more_train_test_pfams
    del more_split_sizes, pfams_in_split, one_hot, validation_pfams
    del num_validation_pairs, validation_clans, train_test_clans
    
    
    ####################################################################
    ### FINAL CHECK OF SPLITS, OUTPUT METADATA AT PFAM AND CLAN LEVELS #
    ####################################################################
    # make sure all pfams only belong to one split
    col_lst = ['in_OOD_valid'] + [f'in_split{i}' for i in range(num_splits)]
    sub_df = df[col_lst]
    rowsum = sub_df.sum(axis=1)
    assert (rowsum == 1).all()
    print('no data bleed :)')
    
    # metadata per split: num_pfams, num_clans (if any), num_seqs, 
    #   num_possible_pairs, percent_of_dataset
    split_meta = []
    for col in col_lst:
        sub_df = df[df[col]]
        
        num_pfams = len(sub_df)
        num_clans = len(list(set(sub_df['clan_name'].tolist())))
        num_seqs = sub_df['msa_depth'].sum()
        num_possible_pairs = sub_df['possible_pairs'].sum()
        percent_by_seqs = num_seqs / df['msa_depth'].sum()
        percent_by_pairs = num_possible_pairs / df['possible_pairs'].sum()
        
        ave_msa_depth = sub_df['msa_depth'].mean()
        ave_msa_width = sub_df['msa_width'].mean()
        ave_percent_gaps = sub_df['percent_gaps'].mean()
        
        out_dict = {'split': col,
                    'num_pfams': num_pfams,
                    'num_clans': num_clans,
                    'num_seqs': num_seqs,
                    'num_possible_pairs': num_possible_pairs,
                    'ave_msa_depth': ave_msa_depth,
                    'ave_msa_width': ave_msa_width,
                    'ave_percent_gaps': ave_percent_gaps,
                    'percent_by_seqs': percent_by_seqs,
                    'percent_by_pairs': percent_by_pairs}
        
        split_meta.append(out_dict)
        
        # also write pfams to separate file
        sub_df['pfam'].to_csv(f'pfams_{col}.tsv', sep='\t', header=False, index=False)
            
    
    split_meta = pd.DataFrame(split_meta)
    
    del col_lst, sub_df, rowsum, col, num_pfams, num_clans, num_seqs
    del num_possible_pairs, percent_by_seqs, percent_by_pairs
    del ave_msa_depth, ave_msa_width, ave_percent_gaps, out_dict
    
    
    # output metadata with split annotations
    pfam_level_metadata_file = f'{prefix}_PFAM-METADATA.tsv'
    split_pfam_meta_file = pfam_level_metadata_file.replace('METADATA','METADATA_withSplitLabels')
    split_metadata_file = f'{prefix}_SPLIT-METADATA.tsv'
    
    df.to_csv(split_pfam_meta_file, sep='\t')
    split_meta.to_csv(split_metadata_file, sep='\t')
