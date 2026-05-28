#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 15:07:29 2025

"""
from jax import numpy as jnp
import numpy as np

def init_time_array(args):
    ### init from geometric grid, like in cherryML
    if args.pred_config['times_from'] == 'geometric':
        t_grid_center = args.pred_config['t_grid_center']
        t_grid_step = args.pred_config['t_grid_step']
        t_grid_num_steps = args.pred_config['t_grid_num_steps']
        
        quantization_grid = range( -(t_grid_num_steps-1), 
                                   t_grid_num_steps, 
                                   1
                                  )
        t_array = [ (t_grid_center * t_grid_step**q_i) for q_i in quantization_grid ]
        
        # make sure it's small times -> large times
        t_array.sort(reverse=False)
        
        return jnp.array(t_array)
    
    
    ### read grid of times from flat text file
    elif args.pred_config['times_from'] == 't_array_from_file':
        times_file = args.pred_config['filenames']['times']
        
        # read file
        t_array = []
        with open(f'{times_file}','r') as f:
            for line in f:
                t_array.append( float( line.strip() ) )
        
        # make sure it's small times -> large times
        t_array.sort(reverse=False)
        
        return jnp.array(t_array)
    
    
    ### figure out time quantization per sample... later
    elif args.pred_config['times_from'] == 't_quantized_per_sample':
        raise NotImplementedError