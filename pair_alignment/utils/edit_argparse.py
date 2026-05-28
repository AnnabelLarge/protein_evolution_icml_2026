#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 18:16:22 2025


"""
#########################################
### enforcing defaults in config file   #
#########################################
def enforce_valid_defaults(args):
    provided_args = list(vars(args).keys())
    
    # if skipping an embedder, make sure they don't write anything to tensorboard
    if ('anc_model_type' in provided_args) and (args.anc_model_type is None):
        args.interms_for_tboard['embeddings'] = False
        args.interms_for_tboard['sow_outputs'] = False

    if ('desc_model_type' in provided_args) and (args.desc_model_type is None):
        args.interms_for_tboard['embeddings'] = False
        args.interms_for_tboard['sow_outputs'] = False
    
    # if you're not updating gradients, don't run any training updates
    if not args.update_grads:
        args.num_epochs = 1


##########################################
### filling config with default values   #
##########################################
def general_fill_with_default_values(args):
    """
    alphabets:
    ===========
    emission alphabet size: normal tokens
      > 20 for amino acids
      > 4 for DNA

    base alphabet size: emission alphabet size + 3 (<bos>, <eos>, and <pad> tags); the input alphabet size
      > 23 for amino acids
      > 7 for DNA
    """
    args.seq_padding_idx = getattr( args, "seq_padding_idx", 0 )
    args.align_padding_idx = getattr( args, "align_padding_idx", -9 )
    args.update_grads = getattr( args, "update_grads", True )
    args.gap_idx = getattr( args, "gap_idx", 43 )
    args.emission_alphabet_size = getattr( args, "emission_alphabet_size", 20 )


def feedforward_fill_with_default_values(args):
    """
    alphabets:
    ===========
    full alphabet size: alignment-augment alphabet, where inserted 
        residues/nucleotides are different from matched residues/nucleotides; 
        the output alphabet size; includes special tokens
    """
    general_fill_with_default_values(args)

    args.in_alph_size = getattr( args, "in_alph_size", args.emission_alphabet_size + 3 )
    args.chunk_length = getattr( args, "chunk_length", 512 )
    args.out_alph_size = getattr( args, "out_alph_size", 44 )
    
    # remap option
    if args.pred_config['t_per_sample']:
        args.pred_config['times_from'] = 't_per_sample'
    
    elif not args.pred_config['t_per_sample']:
        args.pred_config['times_from'] = None


def neural_hmm_fill_with_default_values(args):
    """
    alphabets:
    ===========
    full alphabet size: alignment-augment alphabet, where inserted 
        residues/nucleotides are different from matched residues/nucleotides; 
        the output alphabet size; includes special tokens
    """
    general_fill_with_default_values(args)

    args.in_alph_size = getattr( args, "in_alph_size", args.emission_alphabet_size + 3 )
    args.chunk_length = getattr( args, "chunk_length", 512 )
    

def pairhmm_indp_sites_fill_with_default_values(args):
    """
    no indel mixtures, so num_domain_mixtures and num_fragment_mixtures is automatically 1
    """
    general_fill_with_default_values(args)
    args.pred_config['num_domain_mixtures'] = 1
    args.pred_config['num_fragment_mixtures'] = 1


def pairhmm_frag_and_site_classes_fill_with_default_values(args):
    """
    num_domain_mixtures is automatically 1
    """
    general_fill_with_default_values(args)
    if args.pred_model_type == 'pairhmm_frag_and_site_classes':
        args.pred_config['num_domain_mixtures'] = 1


#########################################################
### sharing top-level arguments with sub-dictionaries   #
#########################################################
def general_share_top_level_args(args):    
    args.pred_config['seq_padding_idx'] = args.seq_padding_idx
    args.pred_config['align_padding_idx'] = args.align_padding_idx
    args.pred_config['gap_idx'] = args.gap_idx


def feedforward_share_top_level_args(args):
    general_share_top_level_args(args)
    
    args.pred_config['in_alph_size'] = args.in_alph_size
    args.pred_config['out_alph_size'] = args.out_alph_size
    
    args.anc_enc_config['in_alph_size'] = args.in_alph_size
    args.anc_enc_config['seq_padding_idx'] = args.seq_padding_idx

    args.desc_dec_config['in_alph_size'] = args.in_alph_size
    args.desc_dec_config['seq_padding_idx'] = args.seq_padding_idx
    

def neural_hmm_share_top_level_args(args):
    general_share_top_level_args(args)
    
    args.pred_config['in_alph_size'] = args.in_alph_size
    
    args.anc_enc_config['in_alph_size'] = args.in_alph_size
    args.anc_enc_config['seq_padding_idx'] = args.seq_padding_idx

    args.desc_dec_config['in_alph_size'] = args.in_alph_size
    args.desc_dec_config['seq_padding_idx'] = args.seq_padding_idx
    
    args.pred_config['emission_alphabet_size'] = args.emission_alphabet_size
    args.pred_config['emissions_postproc_config']['emission_alphabet_size'] = args.emission_alphabet_size
    args.pred_config['transitions_postproc_config']['emission_alphabet_size'] = args.emission_alphabet_size
    
    
def pairhmms_share_top_level_args(args):
    general_share_top_level_args(args)
    args.pred_config['emission_alphabet_size'] = args.emission_alphabet_size
    