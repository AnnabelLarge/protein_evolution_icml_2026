#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 19:23:51 2025


'record_sowed_outputs',
'record_stats_during_hmm_training',
'record_stats_during_neural_training',
'updated_adam_weights_collection',
'updated_gradient_collection',
'updated_model_weight_collection',
'updated_opt_updates_collection',
'updated_sowed_intermediates_collection',
'write_intermeds_to_np_array',
'write_stats_to_npz'
 
"""
from collections.abc import MutableMapping
import jax.numpy as jnp
import numpy as np
import pandas as pd

from typing import Optional
from numpy.typing import ArrayLike


###############################################################################
### HELPERS   #################################################################
###############################################################################
def flatten_convert(dictionary: dict, 
                    parent_key: Optional[str] = None,
                    separator: str = '/'):
    """
    flattens a nested dictionary
    
    arguments
    ---------
    dictionary : dict
        nested dictionary
    
    parent_key : str
        the previous key; accumulate as you go
    
    separator : str
        how to separate parent/child keys
    
    
    example
    -------
    d = { 'firstDict' : {'a': 1, 
                         'b': 2},
          'secondDict' : {'c': 3} }
    
    flat_dict = flatten_convert( dictionary = d,
                                 parent_key = '',
                                 separator = '/' )
    
    flat_dict
    >>> {'firstDict/a': 1,
         'firstDict/b': 2,
         'secondDict/c': 3}
    
    """
    items = []
    for key, value in dictionary.items():
        # format key to reduce repeated key
        if (parent_key is None) or (parent_key in key):
            new_key = key
        
        else:
            new_key = parent_key + separator + key
        
        # flatten and convert
        if isinstance(value, MutableMapping):
            items.extend(flatten_convert(value, 
                                          new_key, 
                                          separator=separator).items()
                          )
        else:
            items.append((new_key, np.array(value)))
    
    return dict(items)


def write_intermeds_to_np_array(step: int,
                                ave_loss: float,
                                mat: ArrayLike, 
                                include_min_max: bool = False,
                                include_perc_zeros: bool = False
                                ):
    """
    extract metrics from matrix
    
    NOTE: metrics could be skewed with many zeros
    
    
    arguments
    ---------
    step : int
        batch, epoch, or some unique identifier
    
    ave_loss : float
        average loss at this step
    
    mat : ArrayLike
        the matrix of interest
        
    include_min_max, include_perc_zeros : bool
        if true, also include max, min, and/or percent zeros
    
    
    returns
    -------
    out_arr : ArrayLike, (7,)
        values in a numpy array
    """
    # always include: L2 norm, mean, variance
    l2_norm = np.linalg.norm(mat.reshape(-1), ord=2).item()
    mean = mat.mean().item()
    variance = mat.var().item()
    
    # optional: record min and max, percent zeros
    max_val = mat.max().item() if include_min_max else -1
    min_val = mat.min().item() if include_min_max else -1
    perc_zeros = ( (mat==0).sum() / mat.size ) if include_perc_zeros else -1
    
    out_arr = np.array( [
        step,
        ave_loss,
        l2_norm,
        mean,
        variance,
        max_val,
        min_val,
        perc_zeros
        ] ) #(7,)
    
    return out_arr



###############################################################################
### Functions to extract dictionaries during training   #######################
###############################################################################
def updated_sowed_intermediates_collection(step: int,
                                           ave_loss: float,
                                           intermediates_dict: dict,
                                           which_model: str,
                                           dict_to_update: dict):
    """
    keys have naming convention:
        tag_prefix_INTERMS/layer_name
    
    
    arguments:
    ------------
    step: int
        > which batch/epoch
    
    ave_loss : float
        average loss at this step
    
    intermediates_dict : dict
        > intermediates for the specific model
        > NOTE: already aggregate stats when sowing
    
    which_model : str
        > which model this is
    
    dict_to_update : dict
        > flat dictionary of arrays 
    
    
    returns:
    ---------
    dict_to_update : dict
        > dictionary AFTER this function
    """
    for layer_name, stats_arr in intermediates_dict.items():
        keyname = f'{which_model.upper()}_INTERMS/'+layer_name
        dict_to_update.setdefault(keyname, []).append( stats_arr ) #(7)
        
    return dict_to_update


def updated_model_weight_collection(step: int,
                                    ave_loss: float,
                                    all_trainstates : list,
                                    dict_to_update : dict):
    """
    keys have naming convention:
        tag_prefix/WEIGHTS/layer_name
    
    
    arguments:
    ------------
    step: int
        > which batch/epoch
        
    ave_loss : float
        average loss at this step
        
    all_trainstates : list
        > list of flax trainstates
    
    dict_to_update : dict
        > flat dictionary of arrays 
    
    
    returns:
    ---------
    dict_to_update : dict
        > dictionary AFTER this function
    """
    for tstate in all_trainstates:
        param_dict = flatten_convert( tstate.params.get('params', dict()) )
        
        for layer_name, param_mat in param_dict.items():
            keyname = f'WEIGHTS/'+layer_name
            stats_arr = write_intermeds_to_np_array( step = step,
                                                     ave_loss = ave_loss,
                                                     mat = param_mat,
                                                     include_min_max = False,
                                                     include_perc_zeros = True ) #(7)
            dict_to_update.setdefault(keyname, []).append( stats_arr )
            
    return dict_to_update


def updated_gradient_collection(step: int,
                                ave_loss: float,
                                gradient_dictionary : dict,
                                dict_to_update : dict):
    """
    keys have naming convention:
        tag_prefix/GRADIENTS/layer_name
    
    
    arguments:
    ------------
    step: int
        > which batch/epoch
        
    ave_loss : float
        average loss at this step
        
    gradient_dictionary : dict
        > containians gradient info
    
    dict_to_update : dict
        > flat dictionary of arrays 
    
    
    returns:
    ---------
    dict_to_update : dict
        > dictionary AFTER this function
    """
    # extract gradients for one model; postproc
    gradient_dictionary = gradient_dictionary.get('params', dict() )
    gradient_dictionary = flatten_convert( gradient_dictionary )
    
    # loop through all items in the dictionary
    for layer_name, val in gradient_dictionary.items():
        keyname = f'GRADIENTS/'+layer_name
        
        # if this is one value, append as-is
        if val.size == 1:
            to_append = np.array( [step, ave_loss, val.item()] ) #(3,)
        
        # if this is a matrix, then calculate stats and append
        elif val.size > 1:
            to_append = write_intermeds_to_np_array( step = step,
                                                     ave_loss = ave_loss.item(),
                                                     mat = val,
                                                     include_min_max = False,
                                                     include_perc_zeros = True ) #(7)
        
        dict_to_update.setdefault(keyname, []).append( to_append )
    
    return dict_to_update


def updated_adam_weights_collection(step: int,
                                    ave_loss: float,
                                    all_trainstates: list,
                                    dict_to_update: dict):
    """
    keys have naming convention:
        tag_prefix/ADAM_OPTIMIZER_MU/layer_name and
        tag_prefix/ADAM_OPTIMIZER_NU/layer_name 
    
    
    arguments:
    ------------
    step: int
        > which batch/epoch
        
    ave_loss : float
        average loss at this step
        
    all_trainstates : list
        > list of flax trainstates
    
    dict_to_update : dict
        > flat dictionary of arrays 
    
    
    returns:
    ---------
    dict_to_update : dict
        > dictionary AFTER this function
    """
    for tstate in all_trainstates:
        ### mu
        mu = tstate.opt_state.inner_opt_state[0].mu.get( 'params', dict() )
        mu = flatten_convert( mu )
        
        for layer_name, param_mat in mu.items():
            keyname = f'ADAM_OPTIMIZER_MU/'+layer_name
            mu_stats = write_intermeds_to_np_array( step = step,
                                                    ave_loss = ave_loss,
                                                    mat = param_mat,
                                                    include_min_max = False,
                                                    include_perc_zeros = True ) #(7)
            dict_to_update.setdefault(keyname, []).append( mu_stats )
        
        del keyname, mu_stats
        
        
        ### nu
        nu = tstate.opt_state.inner_opt_state[0].nu.get( 'params', dict() )
        nu = flatten_convert( nu )

        for layer_name, param_mat in nu.items():
            keyname = f'ADAM_OPTIMIZER_NU/'+layer_name
            nu_stats = write_intermeds_to_np_array( step = step,
                                                    ave_loss = ave_loss,
                                                    mat = param_mat,
                                                    include_min_max = False,
                                                    include_perc_zeros = True ) #(7)
            dict_to_update.setdefault(keyname, []).append( nu_stats )
    
    return dict_to_update


def updated_opt_updates_collection(step: int, 
                                   ave_loss: float,
                                   model_updates_dict: dict,
                                   dict_to_update: dict):
    """
    keys have naming convention:
        PARAM_UPDATE/layer_name
    
    
    arguments:
    ------------
    step : int
        batch, epoch, or some unique identifier
    
    ave_loss : float
        average loss at this step
        
    model_updates_dict : dict
        > updates made to models
    
    dict_to_update : dict
        > flat dictionary of arrays 
    
    
    returns:
    ---------
    dict_to_update : dict
        > dictionary AFTER this function
    """
    model_updates_dict = model_updates_dict.get( 'params', dict() )
    model_updates_dict = flatten_convert( model_updates_dict )
    
    for layer_name, param_mat in model_updates_dict.items():
        keyname = f'PARAM_UPDATE/'+layer_name
        to_append = write_intermeds_to_np_array( step = step,
                                                 ave_loss = ave_loss,
                                                 mat = param_mat,
                                                 include_min_max = False,
                                                 include_perc_zeros = True ) #(7)
        dict_to_update.setdefault(keyname, []).append( to_append )
    
    return dict_to_update


###############################################################################
### Functions to record intermediates during training   #######################
###############################################################################
def record_sowed_outputs(step: int,
                         ave_loss: float,
                         all_trainstates: list,
                         all_intermediates_at_curr_step: dict, 
                         dict_to_update: dict,
                         neural_model: bool):
    """
    wrapper to record intermediate stats during training
    
    
    arguments:
    ------------
    step: int
        > which batch/epoch
        
    ave_loss : float
        average loss at this step
        
    all_trainstates : list
        > list of flax trainstates
    
    all_intermediates_at_curr_step : dict
        > all the intermediates from the training step    
    
    dict_to_update : dict
        > flat dictionary of arrays 
    
    neural_model : bool
        > if true, then also output intermediates from sequence embedders
    
    returns:
    ---------
    dict_to_update : dict
        > dictionary AFTER this function
    """    
    # final projection 
    flat_dict = flatten_convert( all_intermediates_at_curr_step['pred_layer_metrics'] )
    dict_to_update = updated_sowed_intermediates_collection(step = step,
                                                            ave_loss = ave_loss,
                                                            intermediates_dict=flat_dict, 
                                                            which_model = 'FINALPRED',
                                                            dict_to_update = dict_to_update)
    del flat_dict
    
    if neural_model:
        # ancestor embedder
        flat_dict = flatten_convert( all_intermediates_at_curr_step['anc_layer_metrics'] )
        dict_to_update = updated_sowed_intermediates_collection(step = step,
                                                                ave_loss = ave_loss,
                                                                intermediates_dict=flat_dict, 
                                                                which_model = 'ANC',
                                                                dict_to_update = dict_to_update)
        del flat_dict
        
        # descendant embedder
        flat_dict = flatten_convert( all_intermediates_at_curr_step['desc_layer_metrics'] )
        dict_to_update = updated_sowed_intermediates_collection(step = step,
                                                                ave_loss = ave_loss,
                                                                intermediates_dict=flat_dict, 
                                                                which_model = 'DESC',
                                                                dict_to_update = dict_to_update)
        del flat_dict
        
        
def record_stats_during_neural_training(step: int,
                                        ave_loss: float,
                                        all_trainstates: list,
                                        which_interms_to_record: dict,
                                        all_intermediates_at_curr_step: dict, 
                                        dict_to_update: dict):
    """
    wrapper to record intermediate stats during neural network training
    
    
    arguments:
    ------------
    step: int
        > which batch/epoch
        
    ave_loss : float
        average loss at this step
        
    all_trainstates : list
        > list of flax trainstates
    
    which_interms_to_record : dict
        > contains flags to trigger recording
    
    all_intermediates_at_curr_step : dict
        > all the intermediates from the training step    
    
    dict_to_update : dict
        > flat dictionary of arrays 
    
    returns:
    ---------
    dict_to_update : dict
        > dictionary AFTER this function
    """
    ### intermediates sowed by the models 
    if which_interms_to_record['sow_outputs']:  
        record_sowed_outputs(step = step,
                             ave_loss = ave_loss,
                             all_trainstates = all_trainstates,
                             all_intermediates_at_curr_step = all_intermediates_at_curr_step,
                             dict_to_update = dict_to_update,
                             neural_model = True)
    
    
    ### weights
    if which_interms_to_record['weights']:
        dict_to_update = updated_model_weight_collection(step = step,
                                                         ave_loss = ave_loss,
                                                         all_trainstates = all_trainstates,
                                                         dict_to_update = dict_to_update)
            
    
    ### gradients; also already flattened with top_layer_name
    if which_interms_to_record['gradients']:
        for key in ['enc_gradient', 
                    'dec_gradient',
                    'finalpred_gradient']:
            gradient_dict = all_intermediates_at_curr_step[key]
            dict_to_update = updated_gradient_collection(step = step,
                                                         ave_loss = ave_loss,
                                                         gradient_dictionary = gradient_dict,
                                                         dict_to_update = dict_to_update)
            del gradient_dict
        
        
    ### optimizer updates; functions defined above
    if which_interms_to_record['optimizer']:
        # mu, nu
        dict_to_update = updated_adam_weights_collection(step = step,
                                                         ave_loss = ave_loss,
                                                         all_trainstates = all_trainstates,
                                                         dict_to_update = dict_to_update)
        
        # updates
        for item in ['encoder_updates',
                     'decoder_updates',
                     'finalpred_updates']:
            model_updates_dict = all_intermediates_at_curr_step.get(item, dict())
            dict_to_update = updated_opt_updates_collection(step = step,
                                                            ave_loss = ave_loss,
                                                            model_updates_dict = model_updates_dict,
                                                            dict_to_update = dict_to_update)
    
    return dict_to_update
    

def record_stats_during_hmm_training( step: int,
                                      ave_loss: float,
                                      all_trainstates: list,
                                      which_interms_to_record: dict,
                                      all_intermediates_at_curr_step: dict, 
                                      dict_to_update: dict ):
    """
    wrapper to record intermediate stats during neural network training
    
    
    arguments:
    ------------
    step: int
        > which batch/epoch
        
    ave_loss : float
        average loss at this step
        
    all_trainstates : list
        > list of flax trainstates
    
    which_interms_to_record : dict
        > contains flags to trigger recording
    
    all_intermediates_at_curr_step : dict
        > all the intermediates from the training step    
    
    dict_to_update : dict
        > flat dictionary of arrays 
    
    returns:
    ---------
    dict_to_update : dict
        > dictionary AFTER this function
    """
    ### intermediates sowed by the models
    if which_interms_to_record['sow_outputs']:  
        record_sowed_outputs(step = step,
                             ave_loss = ave_loss,
                             all_trainstates = all_trainstates,
                             all_intermediates_at_curr_step = all_intermediates_at_curr_step,
                             dict_to_update = dict_to_update,
                             neural_model = False)
        
    
    ### gradients; also already flattened with top_layer_name
    if which_interms_to_record['gradients']:
        gradient_dict = all_intermediates_at_curr_step['finalpred_gradient']
        dict_to_update = updated_gradient_collection(step = step,
                                                     ave_loss = ave_loss,
                                                     gradient_dictionary = gradient_dict,
                                                     dict_to_update = dict_to_update)
        del gradient_dict
        
        
    ### optimizer updates; functions defined above
    if which_interms_to_record['optimizer']:
        # mu, nu
        dict_to_update = updated_adam_weights_collection(step = step,
                                                         ave_loss = ave_loss,
                                                         all_trainstates = all_trainstates,
                                                         dict_to_update = dict_to_update)
        
        # updates
        model_updates_dict = all_intermediates_at_curr_step.get('finalpred_updates', dict())
        dict_to_update = updated_opt_updates_collection(step = step,
                                                        ave_loss = ave_loss,
                                                        model_updates_dict = model_updates_dict,
                                                        dict_to_update = dict_to_update)
    
    return dict_to_update



###############################################################################
### Functions to use during final eval ########################################
###############################################################################
def write_stats_to_npz(dict_to_write: dict,
                       keyword_lst: list[str],
                       folder: str,
                       file_prefix: str):
    """
    at the very end, write all these intermediates to an npz file   

    possible keywords:
        'ANC_INTERMS',
        'DESC_INTERMS',
        'FINALPRED_INTERMS',
        'WEIGHTS',
        'GRADIENTS',
        'ADAM_OPTIMIZER',
        'PARAM_UPDATE'
    """
    for keyword in keyword_lst:
        subdict = {k: v for k,v in dict_to_write.items() if k.startswith(keyword)}
        if len(subdict) > 0:    
            filename = file_prefix + '_' + keyword.replace(' ','_') + '_STATS.npz'
            np.savez_compressed(f'{folder}/{filename}', **subdict)
    
    # save the column order
    with open(f'{folder}/stats_cols.txt','w') as g:
        g.write(f'columns in numpy stats arrays are:\n')
        g.write(f'step\n')
        g.write(f'ave_loss\n')
        g.write(f'l2_norm\n')
        g.write(f'mean\n')
        g.write(f'variance\n')
        g.write(f'max_val\n')
        g.write(f'min_val\n')
        g.write(f'perc_zeros\n')

                