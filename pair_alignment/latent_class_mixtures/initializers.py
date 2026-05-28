#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 05:47:08 2025

"""
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training.train_state import TrainState

def init_pairhmm_indp_sites( seq_shapes, 
                             dummy_t_array,
                             tx, 
                             model_init_rngkey,
                             pred_config,
                             tabulate_file_loc,
                             *args,
                             **kwargs
                             ):
    """
    for independent site classses over substitution models
    """
    if not pred_config['load_all']:
        from latent_class_mixtures.IndpSites import IndpSites as model
        
    elif pred_config['load_all']:
        from latent_class_mixtures.IndpSites import IndpSitesLoadAll as model
    
    pairhmm_instance = model(config = pred_config,
                                 name = 'IndpSites')
    
        
    ###################################
    ### tabulate and save the model   #
    ###################################
    if (tabulate_file_loc is not None):
        tab_fn = nn.tabulate(pairhmm_instance, 
                              rngs=model_init_rngkey,
                              console_kwargs = {'soft_wrap':True,
                                                'width':250})
        str_out = tab_fn(batch = seq_shapes,
                         t_array = dummy_t_array,
                         sow_flax_intermeds = False,
                         mutable = ['params'])
        with open(f'{tabulate_file_loc}/PAIRHMM_tabulate.txt','w') as g:
            g.write(str_out)
    
    init_params = pairhmm_instance.init(rngs = model_init_rngkey,
                                        batch = seq_shapes,
                                        t_array = dummy_t_array,
                                        sow_flax_intermeds = False,
                                        mutable=['params'])
        
    pairhmm_trainstate = TrainState.create( apply_fn=pairhmm_instance.apply, 
                                              params=init_params,
                                              tx=tx )
        
    return pairhmm_trainstate, pairhmm_instance


    
def init_pairhmm_transit_mixes( pred_model_type,
                                seq_shapes, 
                                dummy_t_array,
                                tx, 
                                model_init_rngkey,
                                pred_config,
                                tabulate_file_loc
                                ):
    """
    for pairHMM using mixtures of domains, mixtures of fragments
    """
    if (pred_model_type == 'pairhmm_frag_and_site_classes') and (not pred_config['load_all']):
        from latent_class_mixtures.FragAndSiteClasses import FragAndSiteClasses as model
        name = 'FragAndSiteClasses'
        
    elif (pred_model_type == 'pairhmm_frag_and_site_classes') and (pred_config['load_all']):
        from latent_class_mixtures.FragAndSiteClasses import FragAndSiteClassesLoadAll as model
        name = 'FragAndSiteClasses'
    
    elif (pred_model_type == 'pairhmm_nested_tkf') and (not pred_config['load_all']):
        from latent_class_mixtures.NestedTKF import NestedTKF as model
        name = 'NestedTKF'
        
    elif (pred_model_type == 'pairhmm_nested_tkf') and (pred_config['load_all']):
        from latent_class_mixtures.NestedTKF import NestedTKFLoadAll as model
        name = 'NestedTKF'
    
    pairhmm_instance = model(config = pred_config,
                             name = name)
    
    
    ###################################
    ### tabulate and save the model   #
    ###################################
    if (tabulate_file_loc is not None):
        tab_fn = nn.tabulate(pairhmm_instance, 
                              rngs=model_init_rngkey,
                              console_kwargs = {'soft_wrap':True,
                                                'width':250})
        str_out = tab_fn(batch = seq_shapes,
                         t_array = dummy_t_array,
                         sow_flax_intermeds = False,
                         mutable = ['params'])
        with open(f'{tabulate_file_loc}/PAIRHMM_tabulate.txt','w') as g:
            g.write(str_out)
    
    init_params = pairhmm_instance.init(rngs = model_init_rngkey,
                                        batch = seq_shapes,
                                        t_array = dummy_t_array,
                                        sow_flax_intermeds = False,
                                        mutable=['params'])
        
    pairhmm_trainstate = TrainState.create( apply_fn=pairhmm_instance.apply, 
                                            params=init_params,
                                            tx=tx )
        
    return pairhmm_trainstate, pairhmm_instance
