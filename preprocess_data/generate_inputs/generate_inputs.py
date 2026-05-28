#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from Bio import Phylo
import pandas as pd
import numpy as np
from itertools import combinations
import random
import math
from copy import deepcopy

from utils.utils import make_sub_folder


def safe_int16(mat):
    if mat.dtype == 'int16':
        return mat
    assert mat.min() >= -32768
    assert mat.max() <= 32767
    return mat.astype('int16')

def safe_int8(mat):
    if mat.dtype == 'int8':
        return mat
    assert mat.min() >= -128
    assert mat.max() <= 127
    return mat.astype('int8')

def safe_int32(mat):
    if mat.dtype == 'int32':
        return mat
    assert mat.min() >= -2147483648
    assert mat.max() <= 2147483647
    return mat.astype('int32')
  
def read_inputs(pfam, seed_folder, trees_folder):
    raw_msa = {}
    pfam_level_meta = {'pfam': pfam,
                       'clan': '',
                       'type': ''}
    
    num_seqs = 0
    with open(f'{seed_folder}/{pfam}.seed','r') as f:
        for line in f:
            if line.startswith('#=GF CL'):
                pfam_level_meta['clan'] = line.strip().split()[-1]
            
            elif line.startswith('#=GF TP'):
                pfam_level_meta['type'] = line.strip().split()[-1]
            
            if not line.startswith('#'):
                num_seqs += 1
                name, seq = line.strip().split()
                seq = seq.upper()
                raw_msa[name] = seq
    
    pfam_level_meta['pfam_Nseqs'] = num_seqs
    
    tree = Phylo.read(f'{trees_folder}/{pfam}.tree', 'newick')
    return raw_msa, tree, pfam_level_meta

def dedup(tuple_list):
    return list( set( tuple( sorted(t) ) for t in tuple_list ) )

def read_pairs_from_file(filename):
    df = pd.read_csv(filename, sep='\t')
    
    # added this bit to specifically handle my inputs
    df = df[['seq1','seq2']]
    
    pairs = df.itertuples(index=False, name=None)
    pairs = dedup(pairs)
    return pairs

def generate_random_pairs(seqnames, percent_of_pairs, filename_of_cherries):
    cherries = read_pairs_from_file(filename_of_cherries)
    reverse_cherries = []
    for (seq1, seq2) in cherries:
        reverse_cherries.append( (seq2, seq1) )
    
    banned_tuples = set( cherries + reverse_cherries )
    
    all_possible_pairs = [tup for tup in combinations(seqnames, 2) if 
                          tup not in banned_tuples]
    
    num_to_sample = math.ceil(percent_of_pairs * len(all_possible_pairs))
    pairs = random.sample(all_possible_pairs, num_to_sample)
    return pairs
    
def extract_alignment(ancestor, descendant, raw_msa):
    anc_gapped = raw_msa[ancestor]
    desc_gapped = raw_msa[descendant]
    alignment = []
    num_matches = 0
    num_subs = 0
    num_ins = 0
    num_dels = 0
    for tup in zip(anc_gapped, desc_gapped):
        if tup != ('.','.'):
            alignment.append(tup)
            
            # ins
            if tup[0] == '.' and tup[1] != '.':
                num_ins += 1
            
            # del
            elif tup[0] != '.' and tup[1] == '.':
                num_dels += 1
            
            # exact match
            elif tup[0] == tup[1]:
                num_matches += 1
            
            # subs
            else:
                num_subs += 1
    
    anc_seq_len = len( anc_gapped.replace('.','') )
    desc_seq_len = len( desc_gapped.replace('.','') )
    alignment_len = len( alignment )
    psi = num_matches/min(anc_seq_len, desc_seq_len)
    
    out_dict = {'perc_seq_id': psi,
                'anc_seq_len': anc_seq_len,
                'desc_seq_len': desc_seq_len,
                'alignment_len': alignment_len,
                'num_matches': num_matches,
                'num_subs': num_subs,
                'num_ins': num_ins,
                'num_dels': num_dels}
    return alignment, out_dict

def get_alphabet():
    special = ['<pad>', '<bos>', '<eos>']
    aas = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 
           'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    mapping = {elem:i for i,elem in enumerate(special + aas)}
    mapping['.'] = 43
    return mapping

def reversible_featurizer(str_alignment, mapping, max_len):
    """
    unaligned_seqs_matrix = (B, L_seq, 2)
      - dim2=0: ancestor, unaligned
      - dim2=1: descendant, unaligned
      - L_seq INCLUDES <bos>, <eos>
    
    aligned_seqs_matrix = (B, L_align, 4)
      - dim2=0: ancestor GAPPED (aligned)
      - dim2=1: descendant GAPPED (aligned)
      - dim2=2: precomputed m indexes for neural models
      - dim2=3: precomputed n indexes for neural models
    
    """
    # dim0=0 is forward pair, dim0=1 is reverse pair
    unaligned_seqs_matrix = np.zeros( (2, max_len+2, 2) )
    aligned_seqs_matrix = np.zeros( (2, max_len+2, 4) )
    
    # padding token for aligned_seqs_matrix[:,:,[2,3]] should be -9
    aligned_seqs_matrix[:,:,[2,3]] = -9
    
    
    ################################
    ### initialize first positions #
    ################################
    # <bos> to start each unaligned sequence
    unaligned_seqs_matrix[:, 0, :] = 1
    
    # <bos> to start each aligned sequence
    aligned_seqs_matrix[:, 0, [0,1]] = 1
    
    # precomputed counts start with (m=1, n=0)
    aligned_seqs_matrix[:, 0, 2] = 1
    aligned_seqs_matrix[:, 0, 3] = 0
    
    
    ##########################################################
    ### step through alignment to fill from string_alignment #
    ##########################################################
    def update_buckets( which, 
                        align_idx, 
                        anc_char, 
                        desc_char, 
                        anc_pos, 
                        desc_pos):
        
        ##############
        ### deletion #
        ##############
        if (desc_char == '.') & (anc_char != '.'):
            ### add to unaligned seq features
            # ancestor
            unaligned_seqs_matrix[which, anc_pos, 0] = mapping[anc_char]
            
            # (no descendant sequence to add)
            
            
            ### add to aligned seq features
            # gapped ancestor
            aligned_seqs_matrix[which, align_idx, 0] = mapping[anc_char]
            
            # gapped descendant
            aligned_seqs_matrix[which, align_idx, 1] =mapping['.']
            
            # at delete site: (m+1, n)
            # precomputed m for NEXT ALIGN IDX
            prev_m = aligned_seqs_matrix[which, align_idx-1, 2]
            aligned_seqs_matrix[which, align_idx, 2] = prev_m + 1
            
            # precomputed n for NEXT ALIGN IDX
            prev_n = aligned_seqs_matrix[which, align_idx-1, 3]
            aligned_seqs_matrix[which, align_idx, 3] = prev_n
            
            
            ### update buckets for next iter
            anc_pos += 1
            
            
        ###############
        ### insertion #
        ###############
        elif (anc_char == '.') & (desc_char != '.'):
            ### add to unaligned seq features
            # (no ancestor sequence to add)
            
            # descendant
            unaligned_seqs_matrix[which, desc_pos, 1] = mapping[desc_char]
            
            
            ### add to aligned seq features
            # gapped ancestor
            aligned_seqs_matrix[which, align_idx, 0] = mapping['.']
            
            # gapped descendant
            aligned_seqs_matrix[which, align_idx, 1] =mapping[desc_char]
            
            # at insert site: (m, n+1)
            # precomputed m for NEXT ALIGN IDX
            prev_m = aligned_seqs_matrix[which, align_idx-1, 2]
            aligned_seqs_matrix[which, align_idx, 2] = prev_m 
            
            # precomputed n for NEXT ALIGN IDX
            prev_n = aligned_seqs_matrix[which, align_idx-1, 3]
            aligned_seqs_matrix[which, align_idx, 3] = prev_n + 1
            
            
            ### update buckets for next iter
            desc_pos += 1
            
            
        ###########
        ### match #
        ###########
        elif (anc_char != '.') & (desc_char != '.'):
            ### add to unaligned seq features
            # ancestor
            unaligned_seqs_matrix[which, anc_pos, 0] = mapping[anc_char]
            
            # descendant
            unaligned_seqs_matrix[which, desc_pos, 1] = mapping[desc_char]
            
            
            ### add to aligned seq features
            # gapped ancestor
            aligned_seqs_matrix[which, align_idx, 0] = mapping[anc_char]
            
            # gapped descendant
            aligned_seqs_matrix[which, align_idx, 1] =mapping[desc_char]
            
            # at match site: (m+1, n+1)
            # precomputed m for NEXT ALIGN IDX
            prev_m = aligned_seqs_matrix[which, align_idx-1, 2]
            aligned_seqs_matrix[which, align_idx, 2] = prev_m  + 1
            
            # precomputed n for NEXT ALIGN IDX
            prev_n = aligned_seqs_matrix[which, align_idx-1, 3]
            aligned_seqs_matrix[which, align_idx, 3] = prev_n + 1
            
            
            ### update buckets for next iter
            anc_pos += 1
            desc_pos += 1
            
        return anc_pos, desc_pos
    
    
    assert len(str_alignment) <= max_len

    fw_anc_pos = 1
    fw_desc_pos = 1
    rv_anc_pos = 1
    rv_desc_pos = 1
    for i, (seq1_char, seq2_char) in enumerate(str_alignment):
        # increment up by one, since you've already initialized first
        # positions
        align_idx = i+1
        
        # forward: (seq1, seq2)
        fw_out = update_buckets(which = 0,
                                align_idx = align_idx,
                                anc_char = seq1_char, 
                                desc_char = seq2_char, 
                                anc_pos = fw_anc_pos, 
                                desc_pos = fw_desc_pos)
        fw_anc_pos, fw_desc_pos = fw_out
        del fw_out
        
        # reverse: (seq2, seq1)
        rv_out = update_buckets(which = 1,
                                align_idx = align_idx,
                                anc_char = seq2_char, 
                                desc_char = seq1_char, 
                                anc_pos = rv_anc_pos, 
                                desc_pos = rv_desc_pos)
        rv_anc_pos, rv_desc_pos = rv_out
        del rv_out
    
    
    ###################################
    ### Add <eos> to end of sequences #
    ###################################
    ### updated unaligned_seqs_matrix 
    # forward: fw_anc_pos, fw_desc_pos
    unaligned_seqs_matrix[0, fw_anc_pos, 0] = 2
    unaligned_seqs_matrix[0, fw_desc_pos, 1] = 2
    
    # reverse: rv_anc_pos, rv_desc_pos
    unaligned_seqs_matrix[1, rv_anc_pos, 0] = 2
    unaligned_seqs_matrix[1, rv_desc_pos, 1] = 2
    
    
    ### update aligned_seqs_matrix at align_idx + 1
    aligned_seqs_matrix[:, align_idx+1, [0,1]] = 2


    ### try encoding
    unaligned_seqs_matrix = safe_int8(unaligned_seqs_matrix)
    aligned_seqs_matrix = safe_int16(aligned_seqs_matrix)
        
    return unaligned_seqs_matrix, aligned_seqs_matrix


def encode_one_pair(i, seq1, seq2, tree, raw_msa, pfam, max_len):
    dist = tree.distance(seq1,seq2)
    fw_pair_level_metadata = {'pairID': f'FW_{pfam}_p{i}',
                              'ancestor': seq1,
                              'descendant': seq2,
                              'TREEDIST_anc-to-desc': dist}
    
    rv_pair_level_metadata = {'pairID': f'RV_{pfam}_p{i}',
                              'ancestor': seq2,
                              'descendant': seq1,
                              'TREEDIST_anc-to-desc': dist}
    
    str_alignment, add_to_fw = extract_alignment(ancestor = seq1, 
                                                 descendant = seq2, 
                                                 raw_msa = raw_msa)
    fw_pair_level_metadata = {**fw_pair_level_metadata, **add_to_fw}
    
    # swap info between anc and desc for reverse pair
    add_to_rv = {'perc_seq_id': add_to_fw['perc_seq_id'],
                  'anc_seq_len': add_to_fw['desc_seq_len'],
                  'desc_seq_len': add_to_fw['anc_seq_len'],
                  'alignment_len': add_to_fw['alignment_len'],
                  'num_matches': add_to_fw['num_matches'],
                  'num_subs': add_to_fw['num_subs'],
                  'num_ins': add_to_fw['num_dels'],
                  'num_dels': add_to_fw['num_ins']
                  }
    rv_pair_level_metadata = {**rv_pair_level_metadata, **add_to_rv}
    del add_to_fw, add_to_rv
    
    
    # generate neural and hmm pair alignment inputs in one go
    mapping = get_alphabet()
    unaligned_seqs_matrix, aligned_seqs_matrix = reversible_featurizer(str_alignment = str_alignment, 
                                                                mapping = mapping, 
                                                                max_len = max_len)
    
    return (fw_pair_level_metadata, 
            rv_pair_level_metadata, 
            unaligned_seqs_matrix, 
            aligned_seqs_matrix)


def featurize_one_pfam(pfam, 
                       seed_folder, 
                       trees_folder, 
                       filename,
                       max_len,
                       pairs_from = 'file', 
                       percent_of_pairs = None):
    ### read inputs, get pairs
    raw_msa, tree, pfam_level_metadata = read_inputs(pfam = pfam, 
                                                      seed_folder = seed_folder, 
                                                      trees_folder = trees_folder)
    pairs = read_pairs_from_file(filename = filename)
    
    
    ### if you don't find any, exit function
    if len(pairs) == 0:
        return None
    
    
    ### iterate through pairs
    metadata = []
    unaligned_outputs = []
    aligned_outputs = []
    
    for pair_id, (seq1, seq2) in enumerate(pairs):
        out = encode_one_pair(i = pair_id, 
                              seq1 = seq1, 
                              seq2 = seq2, 
                              tree = tree,
                              raw_msa = raw_msa,
                              pfam = pfam,
                              max_len = max_len)
        metadata.append(out[0])
        metadata.append(out[1])
        unaligned_outputs.append(out[2])
        aligned_outputs.append(out[3])
    
    metadata = pd.DataFrame(metadata)
    unaligned_outputs = np.concatenate(unaligned_outputs, axis=0)
    aligned_outputs = np.concatenate(aligned_outputs, axis=0)
    
    
    ### add pfam level info to metadata
    for key, val in pfam_level_metadata.items():
        metadata[key] = val
        
    unaligned_outputs = safe_int8(unaligned_outputs)
    aligned_outputs = safe_int16(aligned_outputs)
    
    return unaligned_outputs, aligned_outputs, metadata


##################################
### gather random pairs; combine #
##################################
def make_rand_samp(pfam, 
                   seed_folder,
                   trees_folder,
                   percent_of_pairs,
                   file_of_cherries,
                   dset_prefix,
                   max_len):
    out = featurize_one_pfam(pfam = pfam, 
                             seed_folder = seed_folder,
                             trees_folder = trees_folder, 
                             pairs_from = 'rand_samp',
                             percent_of_pairs = percent_of_pairs,
                             filename = file_of_cherries,
                             max_len = max_len)
    
    if out != None:
        unaligned_outputs = out[0]
        aligned_outputs = out[1]
        metadata = out[2]
        
        with open(f'{dset_prefix}/{pfam}_seqs_unaligned.npy', 'wb') as g:
            np.save(g, unaligned_outputs)
        
        with open(f'{dset_prefix}/{pfam}_aligned_mats.npy', 'wb') as g:
            np.save(g, aligned_outputs)
        
        metadata.to_csv(f'{dset_prefix}/{pfam}_metadata.tsv', sep='\t')
        


##########################################################
### generate pairs from an input file (usually cherries) #
##########################################################
def samples_from_file(pfam, 
                      seed_folder,
                      trees_folder,
                      filename,
                      dset_prefix,
                      max_len):
    out = featurize_one_pfam(pfam = pfam, 
                              seed_folder = seed_folder,
                              trees_folder = trees_folder, 
                              pairs_from = 'file',
                              filename = filename,
                              max_len = max_len)
    
    if out != None:
        unaligned_outputs = out[0]
        aligned_outputs = out[1]
        metadata = out[2]
        
        with open(f'{dset_prefix}_full_length/{pfam}_seqs_unaligned.npy', 'wb') as g:
            np.save(g, unaligned_outputs)
        
        with open(f'{dset_prefix}_full_length/{pfam}_aligned_mats.npy', 'wb') as g:
            np.save(g, aligned_outputs)
        
        metadata.to_csv(f'{dset_prefix}_all_metadata/{pfam}_metadata.tsv', sep='\t')
