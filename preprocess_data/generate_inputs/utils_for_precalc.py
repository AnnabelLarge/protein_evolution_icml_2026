#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ABOUT:
======
precompute counts for HMM inputs

updated on Sept 26 2024 for (B, L, 2) input sizes

"""
import jax
from jax import numpy as jnp


######################
### HELPER FUNCTIONS #
######################
# these get called in summarize_alignment
def count_substitutions(one_sample, one_sample_bool):
    """
    vectorized way to count types of substitutions i.e. emissions at
    match states
    yields a (20, 20) matrix per site in max_len
     
    this will get vmapped over the batch dimension
    """
    # identify what the pair is, using an indicator matrix: 
    #    (rows = anc, cols = desc)
    # subtract index by 3, because there's three special tokens at beginning 
    # of alphabet: pad, bos, and eos
    # finally, multiply by sample boolean i.e. mask the whole matrix if 
    # the position is NOT a match position
    one_sample_bool = jnp.expand_dims(one_sample_bool, 1)
    
    def encode_pairs(vec_at_pos, bool_at_pos):
        indc_mat = jnp.zeros((20,20),dtype='uint8')
        anc_tok, desc_tok = vec_at_pos
        indc_mat = indc_mat.at[anc_tok-3, desc_tok-3].add(1)
        indc_mat = indc_mat * bool_at_pos
        return indc_mat
    vmapped_encode_pairs = jax.vmap(encode_pairs, in_axes=0)
    
    subCounts_persite_persamp = vmapped_encode_pairs(one_sample, one_sample_bool)
    subCounts = subCounts_persite_persamp.sum(axis=0)
    
    return subCounts


def count_insertions(batch, ins_pos_mask):
    """
    count different types of insertions i.e. emissions at insert states
    yields a (20,) vector for the whole batch
    
    unlike other counting methods, this can operate on the whole batch at once
    """
    ### use insertion positions as a mask on the batch
    # look at DESCENDANT SEQ to see what types of insertions happen
    all_inserts = batch[:, :, 1] * ins_pos_mask
    
    ### count the number of valid tokens
    # this gets vmapped over the valid alphabet
    valid_toks = jnp.arange(3, 23)
    def count_insertions(tok):
        return (all_inserts == tok).sum(axis=1)
    vmapped_count_insertions = jax.vmap(count_insertions, in_axes=0)
    
    insCounts = vmapped_count_insertions(valid_toks)
    return insCounts.T


def count_deletions(batch, del_pos_mask):
    """
    count different types of deletions 
    yields a (20,) vector for the whole batch
    
    unlike other counting methods, this can operate on the whole batch at once
    """
    ### use deletion positions as a mask on the batch
    # look at ANCESTOR SEQ to see what aas got deleted
    all_deletes = batch[:, :, 0] * del_pos_mask
    
    ### count the number of valid tokens
    # this gets vmapped over the valid alphabet
    valid_toks = jnp.arange(3, 23)
    def count_deletions(tok):
        return (all_deletes == tok).sum(axis=1)
    vmapped_count_deletions = jax.vmap(count_deletions, in_axes=0)
    
    delCounts = vmapped_count_deletions(valid_toks)
    return delCounts.T


def count_transitions(one_alignment_path, start_idxes):
    """
    vectorized way to count types of transitions (M, I, D)
    yields a (3, 3) matrix for the sample
     
    this will get vmapped over the batch dimension
    """
    # this is vmapped over the length of start_idxes, to get a sliding
    # window effect on one_alignment_path
    def identify_pair_type(start_idx):
        indicator_mat = jnp.zeros((4,4))
        from_tok, to_tok = one_alignment_path[(start_idx, start_idx+1),]
        indicator_mat = indicator_mat.at[from_tok, to_tok].add(1)
        return indicator_mat
    vmapped_subpairs = jax.vmap(identify_pair_type, in_axes=0)
    
    # indicator matrix is (4,4), but first row and column are transitions
    # to padding characters and don't really count; cut them off
    out = vmapped_subpairs(start_idxes)
    transition_counts = out[:, 1:, 1:]
    
    # sum over whole sequence length to get all transitions for this 
    # alignment path
    transition_counts = jnp.sum(transition_counts, axis=0)
    return transition_counts





###################
### MAIN FUNCTION #
###################
def summarize_alignment(batch, align_len, gap_tok=43):
    """
    batch should be a tensor of aligned sequences that are categorically 
    encoded, with the following non-alphabet tokens:
         0: <pad>
         1: <bos> (not included in pairHMM data, but still reserved token)
         2: <eos> (not included in pairHMM data, but still reserved token)
        43: default gap char (but can be changed)
    
    batch is of size (batch_size, max_seq_len, 2), where-
        dim2=0: ancestor sequence, aligned (i.e. with gaps)
        dim2=1: descendant sequence, aligned (i.e. with gaps)
    
    align_len is a vector of size (batch_size,) that has the length of the 
        alignments
    
    """
    #######################################
    ### COMPRESS ALIGNMENT REPRESENTATION #
    #######################################
    ### split into gaps vs not gaps
    non_gaps = jnp.where((batch != gap_tok) & (batch != 0), 1, 0)
    gaps = jnp.where((batch == gap_tok), batch, 0)
    
    ### find matches, inserts, and deletions
    # matches found using non_gaps vector
    match_pos = jnp.where(jnp.sum(non_gaps, axis=2) == 2, 1, 0)
    
    # inserts mean ancestor char == gap_tok
    ins_pos = jnp.where(gaps[:,:,0] == gap_tok, 1, 0)
    
    # deletions means descendant char == gap_tok
    del_pos = jnp.where(gaps[:,:,1] == gap_tok, 1, 0)
    
    # combine all into one vec for later
    # M = 1, I = 2, D = 3; padding is 0
    paths_compressed = (match_pos + (ins_pos*2) + (del_pos*3))
    
    
    ### ADD ADDITIONAL MATCH STATES AT BEGINNING AND END OF ALIGNMENT PATHS
    ### this is part of the LG05, RS07, and H20 assumptions
    # add match at the end of the paths
    extra_end_col = jnp.zeros((batch.shape[0], 1))
    to_adjust = jnp.concatenate([paths_compressed, extra_end_col], axis=1)
    x_idxes = (jnp.arange(0, batch.shape[0]))
    with_extra_end_match = to_adjust.at[x_idxes, align_len].add(1)
    
    # add extra start at the beginning of the paths
    extra_start_col = jnp.ones((batch.shape[0], 1))
    paths_with_extra_start_end_matches = jnp.concatenate([extra_start_col, 
                                                          with_extra_end_match], 
                                                          axis=1).astype(int)
    
    
    ### clean up variables
    del non_gaps, gaps, paths_compressed
    
    
    ######################################
    ### COUNT EMISSIONS FROM MATCH STATE #
    ######################################
    ### use count_substitutions function, defined above
    ### vmap it along the batch dimension (dim0)
    match_pos = match_pos.astype(bool)
    countsubs_vmapped = jax.vmap(count_substitutions, 
                                 in_axes = (0,0))
    subCounts_persamp = countsubs_vmapped(batch, 
                                          match_pos)
    
    
    #######################################
    ### COUNT EMISSIONS FROM INSERT STATE #
    #######################################
    insCounts_persamp = count_insertions(batch = batch, 
                                         ins_pos_mask = ins_pos)
    del ins_pos
    
    ################################################
    ### COUNT TYPES OF DELETIONS FROM DELETE STATE #
    ################################################
    delCounts_persamp = count_deletions(batch = batch, 
                                        del_pos_mask = del_pos)
    del del_pos
    
    
    #######################
    ### COUNT TRANSITIONS #
    #######################
    ### use count_transitions function, defined above
    ### vmap it along the batch dimension (dim0)
    counttrans_vmapped = jax.vmap(count_transitions, 
                                  in_axes=(0, None))
    start_idxes = jnp.arange(start = 0, 
                             stop = paths_with_extra_start_end_matches.shape[1] - 1).astype(int)
    transCounts_persamp = counttrans_vmapped(paths_with_extra_start_end_matches, 
                                             start_idxes)

    return (subCounts_persamp, insCounts_persamp,
            delCounts_persamp, transCounts_persamp)


######################################
### overal equilibrium distributions #
######################################
def get_aa_counts(seq_mat):
    valid_aas = jnp.arange(3, 23)
    
    # vmap this over valid_aas
    def count_aas(aa, dset):
        return (dset == aa).sum()
    
    vmapped_count_aas = jax.vmap(count_aas, in_axes=(0, None))
    counts = vmapped_count_aas(valid_aas, seq_mat)
    return counts

def safe_convert(mat):
    assert mat.max() <= 65535
    assert mat.max() >= 0
    return mat.astype('uint16')


##############################################
### functional wrappers that also save files #
##############################################
def get_equilib_counts(in_dir, split, gap_tok=43):
    print(f'Working on split: {split}')
    
    ### read file
    with open(f'{in_dir}/{split}_pair_alignments.npy', 'rb') as f:
        # B, L, 2
        hmm_paths = np.load(f)
        assert hmm_paths.shape[1] == 2324
        del f
    
    ### run get_aa_counts to get equilibrium distribution across all positions
    AA_counts = get_aa_counts(hmm_paths)
    
    out_file = f'{split}_AAcounts.npy'
    with open(out_file, 'wb') as g:
        np.save(out_file, AA_counts)
    del AA_counts, out_file
    
    
    ### mask to only select match positions
    # split into gaps vs not gaps
    non_gaps = np.where((hmm_paths != gap_tok) & (hmm_paths != 0), 1, 0)
        
    
    # find matches positions, use it to create a mask to only keep values at 
    # match sites
    match_pos = np.where(np.sum(non_gaps, axis=-1) == 2, True, False)[:,:,None]
    del non_gaps
    
    match_mask = np.repeat(match_pos, 2, axis=-1)
    del match_pos
    
    masked_hmm_paths = hmm_paths * match_mask
    del match_mask
    
    
    ### run get_aa_counts as normal on this masked matrix to get equilibrium
    ### distribution ONLY at match positions
    AA_counts_subsOnly = get_aa_counts(masked_hmm_paths)
    
    out_file = f'{split}_AAcounts_subsOnly.npy'
    with open(out_file, 'wb') as g:
        np.save(out_file, AA_counts_subsOnly)
        
        
def summarize_trans_emiss_counts_wrapper(in_dir, split):
    print(f'Processing {split}')
    ### initialize the pytorch dataset object
    dset = pairAlign_dloader(in_dir = in_dir,
                             split = split)
    
    ### create a dataloader that returns jax arrays
    dload = DataLoader(dset, 
                       batch_size = batch_size, 
                       shuffle = False,
                       collate_fn = jax_collator)
    
    subCounts_lst = []
    insCounts_lst = []
    delCounts_lst = []
    transCounts_lst = []
    for i, batch in enumerate(dload):
        # batch is (batchsize, max_len, 2)
        # lens will be (batchsize, max_len)
        lens = (batch != 0).sum(axis=1)[:,0]
        if i == 0:
            jitted_summarize_alignment = jax.jit(summarize_alignment)
            
            
        ### use jit-compiled summarize alignment
        out = jitted_summarize_alignment(batch, lens)
        batch_subCounts_persamp = out[0]
        batch_insCounts_persamp = out[1]
        batch_delCounts_persamp = out[2]
        batch_transCounts_persamp = out[3]
        del out
        
        # adjust types
        batch_subCounts_persamp = safe_convert(batch_subCounts_persamp)
        batch_insCounts_persamp = safe_convert(batch_insCounts_persamp)
        batch_delCounts_persamp = safe_convert(batch_delCounts_persamp)
        batch_transCounts_persamp = safe_convert(batch_transCounts_persamp)
        
        # append
        subCounts_lst.append(batch_subCounts_persamp)
        insCounts_lst.append(batch_insCounts_persamp)
        delCounts_lst.append(batch_delCounts_persamp)
        transCounts_lst.append(batch_transCounts_persamp)
    
    
    # concatenate
    subCounts_persamp = jnp.concatenate(subCounts_lst, axis=0)
    assert subCounts_persamp.shape == (len(dset), 20, 20), f'subCounts shape is wrong: {subCounts_persamp.shape}'
    del subCounts_lst
    
    insCounts_persamp = jnp.concatenate(insCounts_lst, axis=0)
    assert insCounts_persamp.shape == (len(dset), 20), f'insCounts shape is wrong: {insCounts_persamp.shape}'
    del insCounts_lst
    
    delCounts_persamp = jnp.concatenate(delCounts_lst, axis=0)
    assert delCounts_persamp.shape == (len(dset), 20), f'delCounts shape is wrong: {delCounts_persamp.shape}'
    del delCounts_lst
    
    transCounts_persamp = jnp.concatenate(transCounts_lst, axis=0)
    assert transCounts_persamp.shape == (len(dset), 3, 3), f'transCounts shape is wrong: {transCounts_persamp.shape}'
    del transCounts_lst
    
    
    # output
    with open(f'./precalculated_counts/{split}_subCounts.npy', 'wb') as g:
        jnp.save(g, subCounts_persamp)
    
    with open(f'./precalculated_counts/{split}_insCounts.npy', 'wb') as g:
        jnp.save(g, insCounts_persamp)
    
    with open(f'./precalculated_counts/{split}_delCounts.npy', 'wb') as g:
        jnp.save(g, delCounts_persamp)
    
    with open(f'./precalculated_counts/{split}_transCounts.npy', 'wb') as g:
        jnp.save(g, transCounts_persamp)


if __name__ == '__main__':
    import os
    import numpy as np
    from pairAlign_dloader import pairAlign_dloader, jax_collator
    from tqdm import tqdm
    
    ### make sure this still works as expected
    # (M->M) (added START)
    # (M->D)
    # (D->I)
    # (I->M) (added END)
    fake_align = jnp.array([[3,  4, 43, 0],
                            [3, 43,  5, 0]]).T[None,:,:]
    
    true_trans_mat = jnp.array([[1, 0, 1],
                                [1, 0, 0],
                                [0, 1, 0]])
    
    true_match_mat = jnp.zeros( (20, 20) )
    true_match_mat = true_match_mat.at[0,0].add(1)
    
    true_del_vec = jnp.zeros( (20,) )
    true_del_vec = true_del_vec.at[1].add(1)
    
    true_ins_vec = jnp.zeros( (20,) )
    true_ins_vec = true_ins_vec.at[2].add(1)
    
    out = summarize_alignment(fake_align, 
                              3, 
                              gap_tok=43)
    
    pred_match_mat = out[0][0]
    pred_ins_vec = out[1][0]
    pred_del_vec = out[2][0]
    pred_trans_mat = out[3][0]
    del out
    
    assert jnp.allclose(true_match_mat, pred_match_mat)
    assert jnp.allclose(true_del_vec, pred_del_vec)
    assert jnp.allclose(true_ins_vec, pred_ins_vec)
    assert jnp.allclose(true_trans_mat, pred_trans_mat)
    
    
    
    # ### do this for all regular spilts
    # splitnames = [f'TENPERC_split{i}' for i in range(5)]
    # batch_size = 1500
    
    # for split in splitnames:
    #     get_equilib_counts(in_dir = 'tenperc_data_pairAlignments',  
    #                        split = split, 
    #                        gap_tok = 43)
        
    #     summarize_trans_emiss_counts_wrapper(in_dir = 'tenperc_data_pairAlignments',  
    #                                          split = split)
        
    