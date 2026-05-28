#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 12:39:48 2023


About:
======
Custom pytorch dataset object for giving pfam data to counts-based models


outputs:
========
1. sample_subCounts: substitution counts
2. sample_insCounts: insert counts
3. sample_delCounts: deleted char counts
4. sample_transCounts: transition counts
5. sample_time: time to use per sample, or None
6. sample_idx: pair index, to retrieve info from metadata_df


Data to be read:
=================
1. subCounts.npy: (num_pairs, A, A)
    counts of emissions at match states across whole alignment length
    (i.e. true matches and substitutions)
    
2. insCounts.npy: (num_pairs, A)
    counts of emissions at insert states across whole alignment length

3. delCounts.npy: (num_pairs, A)
    counts of bases that get deleted

4. transCounts.npy: (num_pairs, 3, 3) OR (num_pairs, 5, 5)
    transition counts across whole alignment length
    3x3 if encoding start and end states as match sites; 5x5 otherwise


if protein:
    5. AAcounts.npy: (A, )
        equilibrium counts from whole dataset

if dna:
    5. NuclCounts.npy: (A,)
        equilibrium counts from whole dataset

6. metadata.tsv: [PANDAS DATAFRAME]
    metadata about each sample
    lengths do NOT include any sentinel tokens!!!

7. pair-times.tsv: (B,)
    if desired, branch length per sample
    plain .tsv file with two columns; no header and no index
    first column is pairID
    second column is time


"""
import torch
from torch.utils.data import Dataset, DataLoader,default_collate
import numpy as np
import jax
from jax import numpy as jnp
from jax.tree_util import tree_map
import pandas as pd



###############################################################################
### pytorch collator   ########################################################
###############################################################################    
def _default_collate_to_jax_array(mat):
    """
    kind of cumbersome, but conversion path is 
        tuple -> pytorch tensor -> numpy array -> jax array
    """
    pytorch_tensor = default_collate(mat)
    numpy_mat = pytorch_tensor.numpy()
    return jnp.array( numpy_mat )

def jax_collator(batch):
    """
    collator that can handle if time per sample is None
    
    B = number of samples in the batch
    A = alphabet size
    S = number of transitions; 4 here: M, I, D, START/END
    
    Returns
    -------
    collated_subCounts : ArrayLike, (B, A, A)
    collated_insCounts : ArrayLike, (B, A)
    collated_delCounts : ArrayLike, (B, A)
    collated_transCounts : ArrayLike, (B, S, S)
    collated_time : ArrayLike, (B,) OR None
    collated_idx : ArrayLike, (B,)
    """
    # unpack batch
    out = list( zip(*batch) )
    sample_subCounts = out[0]
    sample_insCounts = out[1]
    sample_delCounts = out[2]
    sample_transCounts = out[3]
    sample_time = out[4]
    sample_idx = out[5]
    del out
    
    # handle most with default_collate
    collated_sample_subCounts = _default_collate_to_jax_array( sample_subCounts )
    collated_sample_insCounts = _default_collate_to_jax_array( sample_insCounts )
    collated_sample_delCounts = _default_collate_to_jax_array( sample_delCounts )
    collated_sample_transCounts = _default_collate_to_jax_array( sample_transCounts )
    collated_idx = _default_collate_to_jax_array( sample_idx )
    
    # handle time, which could be none
    if (sample_time[0] is not None):
        collated_times = _default_collate_to_jax_array( sample_time )
    
    elif (sample_time[0] is None):
        collated_times = None
        
    return (collated_sample_subCounts,
            collated_sample_insCounts,
            collated_sample_delCounts,
            collated_sample_transCounts,
            collated_times,
            collated_idx)


###############################################################################
### some helpers   ############################################################
###############################################################################
def _safe_convert(mat):
    """
    pytorch doesn't support uint16 :(
    """
    # try int16 first
    int16_dtype_min = -32768
    int16_dtype_max = 32767
    
    cond1 = mat.max() <= int16_dtype_max
    cond2 = mat.min() >= int16_dtype_min
    
    if cond1 and cond2:
        return mat.astype('int16')
    
    # otherwise, return int32
    else:
        return mat.astype('int32')

"""
def _five_state_to_three_state_transCounts(five_by_five_mat):
    # turn start into M token; add to appropriate transition
    # to M: 0
    # to I: 1
    # to D: 2
    start_to_tok_trans = np.argwhere( five_by_five_mat[:,3,:]==1 )
    tok_to_end_trans = np.argwhere( five_by_five_mat[:,:,4]==1 )
    
    five_by_five_mat[start_to_tok_trans[:,0], 0, start_to_tok_trans[:,1]] += 1
    five_by_five_mat[tok_to_end_trans[:,0], tok_to_end_trans[:,1], 0] += 1
    
    three_by_three_mat = five_by_five_mat[:,:-2, :-2]
    return three_by_three_mat
"""


###############################################################################
### Main dataset object   #####################################################
###############################################################################
class CountsDset(Dataset):
    def __init__(self, 
                 data_dir: str, 
                 split_prefixes: list, 
                 t_per_sample: bool,
                 toss_alignments_longer_than = None,
                 emission_alphabet_size: int = 20,
                 subs_only: bool = False):
        """
        Load training data from precomputed counts of events
        

        Arguments
        ----------
        data_dir : str
            Where data is located
            
        split_prefixes : List[str]
            prefixes of the datasets to include
            
        emission_alphabet_size : int
            4 if DNA, 20 if proteins
            
        t_per_sample : bool
            True if you want to read a branch length per sample, False otherwise
            
        toss_alignments_longer_than : int, None
            Max alignment length to keep, if desired
            DEFAULT VALUE: None

        Attributes created
        -------------------
        self.emit_counts: used to store equilibrium counts of emissions
        self.num_transitions: 4 if using M,I,D,START/END, 3 if not using sentinel tokens
        self.subCounts: counts of emissions at match states
        self.insCounts: counts of emissions at insert states
        self.delCounts: counts of emissions at delete states
        self.transCounts: counts of transitions between states
        self.names_df: dataframe with metadata, for recording later
        self.times: branch length per sample

        """
        #######################################################################
        ### 1: ITERATE THROUGH SPLIT PREFIXES AND READ FILES   ################
        #######################################################################
        ### setup
        # always read
        subCounts_list = []
        insCounts_list = []
        delCounts_list = []
        transCounts_list = []
        metadata_list = []
        
        # equilibrium distribution counts file
        self.emit_counts = np.zeros(emission_alphabet_size, dtype=int)
        
        if (emission_alphabet_size == 20) and (not subs_only):
            counts_suffix = 'AAcounts'
        
        elif (emission_alphabet_size == 20) and (subs_only):
            counts_suffix = 'AAcounts_subsOnly'
        
        elif (emission_alphabet_size == 4) and (not subs_only):
            counts_suffix = 'NuclCounts'
        
        elif (emission_alphabet_size == 4) and (subs_only):
            counts_suffix = 'NuclCounts_subsOnly'
        
        # optionally read times
        if t_per_sample:
            times_lst = []
        
        
        ### start iter
        for split in split_prefixes:
            ##############
            ### metadata #
            ##############
            cols_to_keep = ['pairID',
                            'ancestor',
                            'descendant',
                            'pfam', 
                            'anc_seq_len', 
                            'desc_seq_len',
                            'alignment_len',
                            'num_matches',
                            'num_ins',
                            'num_del']
            meta_df =  pd.read_csv( f'./{data_dir}/{split}_metadata.tsv', 
                                    sep='\t', 
                                    index_col=0,
                                    usecols = cols_to_keep )
            meta_df = meta_df.reset_index(drop = True)
            
            
            #########################################################
            ### remove samples longer where                         #
            ###   align_len + 2 > toss_alignments_longer_than       #
            ###   (plus 2 to mimic the behavior in neural database, #
            ###   which has <bos> and <eos>)                        #
            #########################################################
            if (toss_alignments_longer_than is not None):
                cond = (meta_df['alignment_len'] + 2) <= toss_alignments_longer_than
                idxes_to_keep = list( meta_df[ cond ].index )
                
                if len(idxes_to_keep) == 0:
                    raise RuntimeError(f"no samples to keep from {split}!")
                
                meta_df = meta_df.iloc[idxes_to_keep]
            
            # otherwise, keep everything
            else:
                idxes_to_keep = list( meta_df.index )
            
            metadata_list.append(meta_df)
            
            
            ######################################
            ### counts of emissions, transitions #
            ######################################
            # subEncoded
            with open(f'./{data_dir}/{split}_subCounts.npy', 'rb') as f:
                mat = _safe_convert( np.load(f)[idxes_to_keep, ...] )
                subCounts_list.append( mat )
                del mat
            
            # insCounts
            with open(f'./{data_dir}/{split}_insCounts.npy', 'rb') as f:
                mat = _safe_convert( np.load(f)[idxes_to_keep, ...] )
                insCounts_list.append( mat )
                del mat
            
            # delCounts
            with open(f'./{data_dir}/{split}_delCounts.npy', 'rb') as f:
                mat = _safe_convert( np.load(f)[idxes_to_keep, ...] )
                delCounts_list.append( mat )
                del mat
            
            # transCounts
            with open(f'./{data_dir}/{split}_transCounts_five_by_five.npy', 'rb') as f:
                mat = _safe_convert( np.load(f)[idxes_to_keep, ...] )

                #if bos_eos_as_match:
                #    mat = _five_state_to_three_state_transCounts(mat)
                #    self.num_transitions = 3
                #elif not bos_eos_as_match:

                mat = mat[:, :-1, [0,1,2,4]]
                self.num_transitions = 4
                transCounts_list.append( mat )
                del mat       
            
            # counts (technically uses emissions from tossed samples... 
            #   fix this later)
            with open(f'./{data_dir}/{split}_{counts_suffix}.npy', 'rb') as f:
                mat = _safe_convert( np.load(f) )
                self.emit_counts += mat
                del mat
                   
            
            #####################
            ### (optional) time #
            #####################
            if t_per_sample:
                times = pd.read_csv(f'{data_dir}/{split}_pair-times.tsv', 
                                    sep='\t',
                                    header=None,
                                    names=['pairID','time'],
                                    index_col=None)
                times = times.iloc[idxes_to_keep]
                times_lst += times['time'].tolist()
                del times
            
            del split
                
        
        #######################################################################
        ### 2: CONCATENATE ALL DATA MATRICES   ################################
        #######################################################################
        
        ######################################
        ### counts of emissions, transitions #
        ######################################
        self.subCounts = np.concatenate(subCounts_list, axis=0)
        del subCounts_list
        
        self.insCounts = np.concatenate(insCounts_list, axis=0)
        del insCounts_list
        
        self.delCounts = np.concatenate(delCounts_list, axis=0)
        del delCounts_list
        
        self.transCounts = np.concatenate(transCounts_list, axis=0)
        del transCounts_list
        
        
        ##############
        ### metadata #
        ##############
        self.names_df = pd.concat(metadata_list, axis=0)
        self.names_df = self.names_df.reset_index(drop=True)
        del metadata_list
        
        
        #####################
        ### (optional) time #
        #####################
        if t_per_sample:
            self.times = np.array(times_lst) #(B,)
            del times_lst
        
        else:
            self.times = None
        
        
    def __len__(self):
        return self.subCounts.shape[0]

    def __getitem__(self, idx):
        sample_subCounts = self.subCounts[idx, ...]
        sample_insCounts = self.insCounts[idx, ...]
        sample_delCounts = self.delCounts[idx, ...]
        sample_transCounts = self.transCounts[idx, ...]
        
        if self.times is not None:
            sample_time = self.times[idx]
        else:
            sample_time = None
        
        sample_idx = idx
        
        return (sample_subCounts, 
                sample_insCounts, 
                sample_delCounts, 
                sample_transCounts, 
                sample_time,
                sample_idx)
    
    def retrieve_sample_names(self, idxes):
        # used the list of sample indices to query the original names_df
        return self.names_df.iloc[idxes]
    
    def retrieve_equil_dist(self):
        return self.emit_counts / ( self.emit_counts.sum() )
    