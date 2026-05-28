#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 19:27:33 2025

"""
import optax
import numpy as np

def build_optimizer(args):
    config = args.optimizer_config
    
    learning_rate = optax.warmup_cosine_decay_schedule(init_value= config['init_value'], 
                                                       peak_value= config['peak_value'], 
                                                       end_value = config['end_value'],
                                                       warmup_steps= config['warmup_steps'], 
                                                       decay_steps= args.num_epochs)
    
    base_optimizer = optax.adamw(learning_rate = learning_rate,
                                 weight_decay = config['weight_decay'])
    
    tx = optax.MultiSteps(opt = base_optimizer,
                          every_k_schedule = config['every_k_schedule'])
    
    return tx

