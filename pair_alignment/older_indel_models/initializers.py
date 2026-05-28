#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 29 19:40:31 2025

"""
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training.train_state import TrainState

def init_pairhmm( seq_shapes, 
                  dummy_t_array,
                  tx, 
                  model_init_rngkey,
                  pred_config,
                  tabulate_file_loc,
                  *args,
                  **kwargs
                  ):
    if not pred_config['load_all']:
        from older_indel_models.IndpSitesOldModels import IndpSitesOldModels as model
        
    elif pred_config['load_all']:
        from older_indel_models.IndpSitesOldModels import IndpSitesOldModelsLoadAll as model
    
    pairhmm_instance = model(config = pred_config,
                                 name = 'IndpSitesOldModels')
    
        
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
