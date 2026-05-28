#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ABOUT:
======
Helpers to create train state objects; assumes all layers could have dropout

Also save the text outputs of nn.tabulate

Have option to initialize the final bias, but generally found this to be 
  unhelpful


TODO:
=====
- Incorporate batch stats (whenever you use BatchNorm)
"""
import importlib
from typing import Optional, Dict

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training.train_state import TrainState


##############################
### For sequence embedders   #
##############################
def create_seq_model_tstate(embedding_which,
                            seq_shape, 
                            tx, 
                            model_init_rngkey, 
                            tabulate_file_loc,
                            model_type: str = None, 
                            model_config: dict = dict() ):
    ### embedding_which option controls some naming/behavior
    if embedding_which == 'anc':
        model_name_suffix = 'ANCESTOR ENCODER'
        causal = False
        tabulate_prefix = 'ANC-ENCODER'
        
    elif embedding_which == 'desc':
        model_name_suffix = 'DESCENDANT DECODER'
        causal = True
        tabulate_prefix = 'DESC-DECODER'
    
    
    ### Import initial embedding module (some models won't need this)
    if 'initial_embed_module' in model_config:
        from neural_models.sequence_embedders import initial_embedding_blocks
        initial_embed_module = getattr(initial_embedding_blocks, 
                                       model_config['initial_embed_module'])
    
    
    ################################
    ### Sequence embedding imports #
    ################################
    ### CNN (only one block type: ConvnetBlock)
    if model_type == 'cnn':
        # import blocks to use 
        from neural_models.sequence_embedders.cnn import blocks_fns
        first_block_module = getattr(blocks_fns, 
                                     model_config.get("first_block_module", "ConvnetBlock") )
        subsequent_block_module = getattr(blocks_fns,
                                          model_config.get("subsequent_block_module", "ConvnetBlock") )
        
        # initialize
        from neural_models.sequence_embedders.cnn.embedders import CNNSeqEmb
        seq_model_instance = CNNSeqEmb(embedding_which = embedding_which.lower(),
                                        initial_embed_module = initial_embed_module,                          
                                        first_block_module = first_block_module,
                                        subsequent_block_module = subsequent_block_module,
                                        causal = causal,
                                        config = model_config,
                                        name =f'CNN {model_name_suffix}')
        
        # adjust dim3 size
        expected_dim3_size = model_config['hidden_dim']
    
    
    ### LSTM
    elif model_type == 'lstm':
        # import blocks to use 
        from neural_models.sequence_embedders.lstm import blocks_fns
        first_block_module = getattr(blocks_fns, 
                                     model_config["first_block_module"])
        subsequent_block_module = getattr(blocks_fns,
                                          model_config["subsequent_block_module"])

        # certain blocks are banned from being used on certain sequences
        if embedding_which == 'desc':
            banned_list = ['BidirecLSTMLayer', 
                           'BidirecLSTMLayerWithDropoutBefore',
                           'BidirecLSTMLayerWithDropoutAfter']

            err_msg = (f'Illegal block used for {embedding_which} sequence '+
                       f'embedding; check config again')
            assert first_block_module not in banned_list, err_msg
            assert subsequent_block_module not in banned_list, err_msg

        # initialize
        from neural_models.sequence_embedders.lstm.embedders import LSTMSeqEmb
        seq_model_instance = LSTMSeqEmb(embedding_which = embedding_which.lower(),
                                        initial_embed_module = initial_embed_module,                          
                                        first_block_module = first_block_module,
                                        subsequent_block_module = subsequent_block_module,
                                        causal = causal,
                                        config = model_config,
                                        name= f'LSTM {model_name_suffix}')
        
        # adjust dim3 size; might have different size for ancestor embeddings
        if embedding_which == 'anc':
            #assume merge_how=concat
            expected_dim3_size = model_config['hidden_dim']*2 
            
        elif embedding_which == 'desc':
            expected_dim3_size = model_config['hidden_dim']
    
    
    ### Transformer
    elif model_type == 'transformer':
        # import blocks to use 
        from neural_models.sequence_embedders.transformer import blocks_fns
        first_block_module = getattr(blocks_fns, 
                                     model_config["first_block_module"])
        subsequent_block_module = getattr(blocks_fns, 
                                          model_config["subsequent_block_module"])
        
        # initialize
        from neural_models.sequence_embedders.transformer.embedders import TransfSeqEmb
        seq_model_instance = TransfSeqEmb(embedding_which = embedding_which.lower(),
                                          initial_embed_module = initial_embed_module,                          
                                          first_block_module = first_block_module,
                                          subsequent_block_module = subsequent_block_module,
                                          causal = causal,
                                          config = model_config,
                                          name =f'TRANSFORMER {model_name_suffix}')
        
        # adjust dim3 size
        expected_dim3_size = model_config['hidden_dim']
        
    
    ### error if value not in list
    else:
        valid_types = ["cnn",
                       "lstm",
                       "transformer"]
        to_write = ", ".join(valid_types)
        raise RuntimeError(f'Pick valid model type for {model_type}: {to_write}')
    
    
    ##################
    ### initialize   #
    ##################
    dummy_in = jnp.empty( seq_shape, dtype=int)
    
    
    ### tabulate and save the model
    if (tabulate_file_loc is not None):
        tab_fn = nn.tabulate(seq_model_instance, 
                             rngs=model_init_rngkey,
                             console_kwargs = {'soft_wrap':True,
                                               'width':250})
        str_out = tab_fn(datamat = dummy_in, 
                         training = False,
                         sow_flax_intermeds = False,
                         mutable = ['params'])
        
        with open(f'{tabulate_file_loc}/{tabulate_prefix}_tabulate.txt','w') as g:
            g.write(str_out)
    
    
    ### turn into a train state
    init_params = seq_model_instance.init(rngs=model_init_rngkey,
                                         datamat = dummy_in,
                                         training = False,
                                         sow_flax_intermeds = False,
                                         mutable = ['params'])
    
    seq_model_trainstate = TrainState.create(apply_fn=seq_model_instance.apply, 
                                             params=init_params,
                                             tx=tx)
    
    return (seq_model_trainstate, seq_model_instance, expected_dim3_size)


def prediction_head_instance( pred_model_type: str,
                              datamat_lst_shapes: jnp.array, 
                              tx: Dict,
                              model_init_rngkey: jnp.array, 
                              tabulate_file_loc: str,
                              t_array: Optional[jnp.array],
                              model_config: Dict = dict() ):
    #############
    ### imports #
    #############
    if pred_model_type == 'neural_hmm':
        from neural_models.neural_hmm_predict.NeuralCondTKF import NeuralCondTKF as Model
        model_name = 'NEURAL-TKF PREDICT'
    
    elif pred_model_type == 'feedforward':
        from neural_models.feedforward_predict.FeedforwardPredict import FeedforwardPredict as Model
        model_name = 'FEEDFORWARD PREDICT'
    
    finalpred_instance = Model(config = model_config,
                               name = model_name)
    
    ##################
    ### initialize   #
    ##################
    dummy_mat_lst = [jnp.empty(s) for s in datamat_lst_shapes]
    dim0 = dummy_mat_lst[0].shape[0] #(B)
    dim1 = dummy_mat_lst[0].shape[1] #(L)
    dummy_masking_mat = jnp.empty( (dim0, dim1) ) #(B,L)
    
    
    ### tabulate and save the model
    if (tabulate_file_loc is not None):
        tab_fn = nn.tabulate( finalpred_instance, 
                             rngs=model_init_rngkey,
                             console_kwargs={'soft_wrap': True, 
                                             'width': 250} 
                             )
    
        # Build argument dictionary
        str_out = tab_fn( datamat_lst = dummy_mat_lst,
                          padding_mask = dummy_masking_mat,
                          t_array = t_array,
                          training = False,
                          sow_flax_intermeds = False,
                          mutable = ['params'] )
        with open(f'{tabulate_file_loc}/OUT-PROJ_tabulate.txt','w') as g:
            g.write(str_out)
        
    
    ### turn into a train state
    # Initialize with conditional arguments
    init_params = finalpred_instance.init( rngs = model_init_rngkey,
                                           datamat_lst = dummy_mat_lst,
                                           padding_mask = dummy_masking_mat,
                                           t_array = t_array,
                                           training = False,
                                           sow_flax_intermeds = False,
                                           mutable = ['params'] )
    finalpred_trainstate = TrainState.create(apply_fn=finalpred_instance.apply, 
                                              params=init_params,
                                              tx=tx)
    
    return (finalpred_trainstate, finalpred_instance)


def create_all_tstates(seq_shapes, 
                       tx, 
                       model_init_rngkey, 
                       tabulate_file_loc: str,
                       anc_model_type: str, 
                       desc_model_type: str, 
                       pred_model_type: str, 
                       anc_enc_config: dict, 
                       desc_dec_config: dict, 
                       pred_config: dict,
                       t_array_for_all_samples: Optional[jnp.array] ):
    # largest_seqs is (B, max_seq_len)
    # largest_aligns is (B, max_align_len)
    # max_seq_len != max_align_len
    largest_seqs, largest_aligns, t_per_sample = seq_shapes
    
    have_t_grid = t_array_for_all_samples is not None
    have_t_per_sample = t_per_sample is not None
    
    # for neural TKF and feedforward
    if have_t_grid and ~have_t_per_sample:
        t_array_for_init = t_array_for_all_samples #(T,)

    elif ~have_t_grid and have_t_per_sample:
        t_array_for_init = jnp.zeros( t_per_sample.shape ) #(B,)
    
    # could happen for feedforward
    elif ~have_t_grid and ~have_t_per_sample:
        t_array_for_init = None
        
    del have_t_grid, have_t_per_sample
    
    # keep track of dim3 size
    expected_dim3_size = 0
    
    # split input key
    keys = jax.random.split(model_init_rngkey, num=3)
    anc_rngkey, desc_rngkey, outproj_rngkey = keys
    del keys
    
    
    ### ancestor encoder
    out = create_seq_model_tstate( embedding_which = 'anc',
                                   seq_shape = largest_seqs, 
                                   tx = tx, 
                                   model_init_rngkey = anc_rngkey, 
                                   tabulate_file_loc = tabulate_file_loc,
                                   model_type = anc_model_type,
                                   model_config = anc_enc_config )
    ancestor_trainstate = out[0]
    ancestor_instance = out[1]
    ancestor_emb_size = (largest_seqs[0], largest_aligns[1], out[2])
    
    
    ### descendant decoder
    out = create_seq_model_tstate( embedding_which = 'desc',
                                   seq_shape = largest_seqs, 
                                   tx = tx, 
                                   model_init_rngkey = desc_rngkey, 
                                   tabulate_file_loc = tabulate_file_loc,
                                   model_type = desc_model_type,
                                   model_config = desc_dec_config )
    descendant_trainstate = out[0]
    descendant_instance = out[1]
    descendant_emb_size = (largest_seqs[0], largest_aligns[1], out[2])
    
    list_of_shapes = [ancestor_emb_size, descendant_emb_size]
    
    
    ### final prediction network
    # set output shape
    prev_state_size = (largest_seqs[0], largest_aligns[1], 5)
    list_of_shapes.append(prev_state_size)
    
    # init
    out = prediction_head_instance(pred_model_type = pred_model_type,
                                   datamat_lst_shapes = list_of_shapes, 
                                   tx = tx,
                                   model_init_rngkey = outproj_rngkey, 
                                   tabulate_file_loc = tabulate_file_loc,
                                   t_array = t_array_for_init,
                                   model_config = pred_config)
    
    finalpred_trainstate, finalpred_instance = out
    del out
    
    all_trainstates = (ancestor_trainstate, 
                       descendant_trainstate, 
                       finalpred_trainstate)
    
    all_instances = (ancestor_instance, 
                     descendant_instance, 
                     finalpred_instance)
    
    
    ### always use extract_embs concatenation function
    from neural_models.sequence_embedders.concatenation_fns import extract_embs as concat_fn
    
    return all_trainstates, all_instances, concat_fn
