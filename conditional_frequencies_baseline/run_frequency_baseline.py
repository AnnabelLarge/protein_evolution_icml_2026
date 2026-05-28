#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm


###############################################################################
### compile_joint_emission_counts                                             #
###############################################################################
def load_mat(file):
    with open(file, 'rb') as f:
        return np.load(f)


def count_pairs(prefix):
    mat = load_mat(f'{prefix}_aligned_mats.npy')

    all_emits = np.zeros((21, 21))

    anc_seqs = mat[..., 0]
    anc_seqs = anc_seqs[~np.isin(anc_seqs, np.array([0, 1, 2]))]
    anc_seqs = np.where(anc_seqs == 43, 23, anc_seqs) - 3

    desc_seqs = mat[..., 1]
    desc_seqs = desc_seqs[~np.isin(desc_seqs, np.array([0, 1, 2]))]
    desc_seqs = np.where(desc_seqs == 43, 23, desc_seqs) - 3

    np.add.at(all_emits, (anc_seqs, desc_seqs), 1)

    return all_emits


def compile_emission_counts(prefix_list):
    for prefix in tqdm(prefix_list):
        emits = count_pairs(prefix)
        with open(f'{prefix}_21x21_emission_counts.npy', 'wb') as g:
            np.save(g, emits)


###############################################################################
### score_sequences_with_frequencies                                          #
###############################################################################
def replace_zeros(mat):
    return np.where(mat != 0, mat, 1)


def load_params(train_set, data_dir, counts_dir, file_prefix):
    """Estimate geometric and emission parameters from training splits."""
    train_align_lengths = []
    train_anc_lengths = []
    joint_emission_counts = np.zeros((21, 21))

    for i in train_set:
        meta_df = pd.read_csv(
            f'{data_dir}/{file_prefix}_split{i}_metadata.tsv',
            sep='\t', usecols=['anc_seq_len', 'alignment_len']
        )
        train_align_lengths += meta_df['alignment_len'].to_list()
        train_anc_lengths += meta_df['anc_seq_len'].to_list()

        with open(f'{counts_dir}/{file_prefix}_split{i}_21x21_emission_counts.npy', 'rb') as f:
            joint_emission_counts += np.load(f)

    observed_marginal_counts = joint_emission_counts.sum(axis=1)
    observed_marginal_counts[-1] = 0  # gap char gets zero count

    align_geom_p_eos = (len(train_align_lengths)
                        / (len(train_align_lengths) + joint_emission_counts.sum()))
    anc_geom_p_eos = (len(train_anc_lengths)
                      / (len(train_align_lengths) + observed_marginal_counts.sum()))

    joint_probs = replace_zeros(joint_emission_counts / joint_emission_counts.sum())
    marg_probs  = replace_zeros(observed_marginal_counts / observed_marginal_counts.sum())

    return joint_probs, marg_probs, align_geom_p_eos, anc_geom_p_eos


def score_dset(split_idx, data_dir, file_prefix,
               observed_joint_emit_probs, observed_marg_emit_probs,
               align_geom_p_eos, anc_geom_p_eos):
    dset = f'split{split_idx}'
    metadata_file     = f'{data_dir}/{file_prefix}_{dset}_metadata.tsv'
    aligned_seqs_file = f'{data_dir}/{file_prefix}_{dset}_aligned_mats.npy'

    log_joint_probs = np.log(observed_joint_emit_probs)
    log_marg_probs  = np.log(observed_marg_emit_probs)

    with open(aligned_seqs_file, 'rb') as f:
        aligned_seqs_mat = (np.load(f)[:, :, [0, 1]]) - 3  # (B, L, 2)

    descendant_lengths = (~np.isin(aligned_seqs_mat[..., 1], [-3, -2, -1, 40])).sum(axis=1)
    ancestor_lengths   = (~np.isin(aligned_seqs_mat[..., 0], [-3, -2, -1, 40])).sum(axis=1)

    aligned_seqs_mat = np.where(aligned_seqs_mat == 40, 20, aligned_seqs_mat)

    rows, cols = aligned_seqs_mat[..., 0], aligned_seqs_mat[..., 1]

    # joint
    joint_emission_logprob = log_joint_probs[rows, cols] + np.log(1 - align_geom_p_eos)
    joint_emission_logprob = np.where(aligned_seqs_mat[..., 0] < 0, 0, joint_emission_logprob)
    joint_raw_logprob = joint_emission_logprob.sum(axis=1) + np.log(align_geom_p_eos)
    del joint_emission_logprob, cols, log_joint_probs

    # ancestral marginal
    marginal_emission_logprob = log_marg_probs[rows] + np.log(1 - anc_geom_p_eos)
    mask = ~np.isin(aligned_seqs_mat[..., 0], [-3, -2, -1, 20])
    marginal_emission_logprob = np.where(mask, marginal_emission_logprob, 0)
    marginal_raw_logprob = marginal_emission_logprob.sum(axis=1) + np.log(anc_geom_p_eos)
    del marginal_emission_logprob, mask

    cond_raw_logprob = joint_raw_logprob - marginal_raw_logprob

    out_df = pd.read_csv(metadata_file, sep='\t',
                         usecols=['pairID', 'ancestor', 'descendant', 'pfam'])
    out_df['desc_lens']                         = descendant_lengths
    out_df['joint_logprob']                     = joint_raw_logprob
    out_df['joint_logprob_normed_by_desc_lens'] = -(joint_raw_logprob / descendant_lengths)
    out_df['anc_logprob']                       = marginal_raw_logprob
    out_df['anc_logprob_normed_by_lens']        = -(marginal_raw_logprob / ancestor_lengths)
    out_df['cond_logprob']                      = cond_raw_logprob
    out_df['cond_logprob_normed_by_desc_lens']  = -(cond_raw_logprob / descendant_lengths)

    return out_df


def get_stats(df, outfile, dset_name):
    # joint
    sum_joint_loglikes            = -df['joint_logprob'].sum()
    joint_ave_loss                = -df['joint_logprob'].mean()
    joint_ave_loss_seqlen_normed  = df['joint_logprob_normed_by_desc_lens'].mean()
    joint_ece                     = np.exp(joint_ave_loss_seqlen_normed)
    joint_perplexity              = np.exp(df['joint_logprob_normed_by_desc_lens']).mean()

    # anc marginal
    sum_anc_loglikes              = -df['anc_logprob'].sum()
    anc_ave_loss                  = -df['anc_logprob'].mean()
    anc_ave_loss_seqlen_normed    = df['anc_logprob_normed_by_lens'].mean()
    anc_ece                       = np.exp(anc_ave_loss_seqlen_normed)
    anc_perplexity                = np.exp(df['anc_logprob_normed_by_lens']).mean()

    # conditional
    sum_cond_loglikes             = -df['cond_logprob'].sum()
    cond_ave_loss                 = -df['cond_logprob'].mean()
    cond_ave_loss_seqlen_normed   = df['cond_logprob_normed_by_desc_lens'].mean()
    cond_ece                      = np.exp(cond_ave_loss_seqlen_normed)
    cond_perplexity               = np.exp(df['cond_logprob_normed_by_desc_lens']).mean()

    out_dict = {
        'sum_joint_loglikes':           sum_joint_loglikes,
        'joint_ave_loss':               joint_ave_loss,
        'joint_ave_loss_seqlen_normed': joint_ave_loss_seqlen_normed,
        'joint_ece':                    joint_ece,
        'joint_perplexity':             joint_perplexity,
        'sum_anc_loglikes':             sum_anc_loglikes,
        'anc_ave_loss':                 anc_ave_loss,
        'anc_ave_loss_seqlen_normed':   anc_ave_loss_seqlen_normed,
        'anc_ece':                      anc_ece,
        'anc_perplexity':               anc_perplexity,
        'sum_cond_loglikes':            sum_cond_loglikes,
        'cond_ave_loss':                cond_ave_loss,
        'cond_ave_loss_seqlen_normed':  cond_ave_loss_seqlen_normed,
        'cond_ece':                     cond_ece,
        'cond_perplexity':              cond_perplexity,
    }

    with open(outfile, 'w') as g:
        g.write(f'RUN\t{dset_name}\n')
        for key, val in out_dict.items():
            g.write(f'{key}\t{val}\n')

    return out_dict


def run_analysis(train_set, test_set, prefix, data_dir, counts_dir, file_prefix):
    print(f'\n=== {prefix} ===')
    print('Preparing params')
    joint_probs, marg_probs, align_p_eos, anc_p_eos = load_params(
        train_set, data_dir, counts_dir, file_prefix
    )

    with open(f'{prefix}_geom_p_eos.tsv', 'w') as g:
        g.write(f'Parameter for geometric distribution over ALIGNMENTS: {align_p_eos}\n')
        g.write(f'Parameter for geometric distribution over ANCESTORS: {anc_p_eos}\n')

    print('Scoring train set')
    train_dfs = [score_dset(i, data_dir, file_prefix,
                            joint_probs, marg_probs, align_p_eos, anc_p_eos)
                 for i in tqdm(train_set)]

    print('Scoring test set')
    test_dfs = [score_dset(i, data_dir, file_prefix,
                           joint_probs, marg_probs, align_p_eos, anc_p_eos)
                for i in tqdm(test_set)]

    all_train = pd.concat(train_dfs, axis=0)
    all_test  = pd.concat(test_dfs,  axis=0)

    all_train.to_csv(f'{prefix}_train-set_loglikes-per-samp.tsv', sep='\t')
    all_test.to_csv( f'{prefix}_test-set_loglikes-per-samp.tsv',  sep='\t')

    get_stats(all_train, f'{prefix}_TRAIN-AVE-LOGLIKES.tsv', 'train_set')
    get_stats(all_test,  f'{prefix}_TEST-AVE-LOGLIKES.tsv',  'test_set')


###############################################################################
### entry point                                                               #
###############################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--example', action='store_true',
                        help='Run on example_data (split0=train, split1=test)')
    args = parser.parse_args()

    if args.example:
        print('=== Step 1: Compiling emission counts ===')
        compile_emission_counts(
            prefix_list=[f'example_data/CHERRIES_split{i}' for i in range(0, 2)]
        )

        print('\n=== Step 2: Scoring sequences ===')
        run_analysis(
            train_set   = [0],
            test_set    = [1],
            prefix      = 'example',
            data_dir    = 'example_data',
            counts_dir  = 'example_data',
            file_prefix = 'CHERRIES',
        )

    else:
        print('=== Step 1: Compiling emission counts ===')
        compile_emission_counts(
            prefix_list=[f'FAMCLAN-CHERRIES_split{i}' for i in range(0, 10)]
        )
        compile_emission_counts(prefix_list=['FAMCLAN-CHERRIES_OOD_Valid'])

        print('\n=== Step 2: Scoring sequences ===')
        run_analysis(
            train_set   = [0, 1, 2, 3, 4, 5, 6],
            test_set    = [8, 9],
            prefix      = 'data1',
            data_dir    = 'DATA_cherries',
            counts_dir  = 'pairhmm_counts',
            file_prefix = 'FAMCLAN-CHERRIES',
        )
        run_analysis(
            train_set   = [1, 3, 4, 5, 7, 8, 9],
            test_set    = [0, 2],
            prefix      = 'data2',
            data_dir    = 'DATA_cherries',
            counts_dir  = 'pairhmm_counts',
            file_prefix = 'FAMCLAN-CHERRIES',
        )
        run_analysis(
            train_set   = [1, 2, 4, 6, 7, 8, 9],
            test_set    = [3, 5],
            prefix      = 'data3',
            data_dir    = 'DATA_cherries',
            counts_dir  = 'pairhmm_counts',
            file_prefix = 'FAMCLAN-CHERRIES',
        )
