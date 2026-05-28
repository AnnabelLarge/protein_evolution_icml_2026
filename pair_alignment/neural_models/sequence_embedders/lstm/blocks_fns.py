#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 17:12:00 2023


Custom layers to throw into larger modules
"""
from flax import linen as nn
import jax
import jax.numpy as jnp
from typing import Callable



###############################################################################
### LSTM in one direction   ###################################################
###############################################################################
class UnidirecLSTMLayer(nn.Module):
    """
    An LSTM layer that operates in one direction
    
    """
    config: dict
    name: str
    
    def setup(self):
        ### unpack from config
        self.hidden_dim = self.config["hidden_dim"]
        
        ### LSTM layers
        # the cell
        self.lstm_cell = nn.OptimizedLSTMCell(features=self.hidden_dim, 
                                              name=self.name)
        
        # the wrapper around the cell
        self.rnn_wrapper = nn.RNN(cell = self.lstm_cell,
                                  return_carry = True)
        
    
    def __call__(self, 
                 datamat, 
                 datalens, 
                 carry=None, 
                 **kwargs):
        out_carry, datamat = self.rnn_wrapper(inputs=datamat, 
                                              initial_carry=carry,
                                              seq_lengths=datalens)
        return (out_carry, datamat)
    


class UnidirecLSTMLayerWithDropoutBefore(nn.Module):
    """
    Dropout THEN an LSTM layer 
    
    """
    config: dict
    name: str
    
    @nn.compact
    def __call__(self, 
                 datamat, 
                 datalens, 
                 training:bool, 
                 carry=None):
        # dropout
        do_rate = self.config["dropout"]
        datamat = nn.Dropout( rate = do_rate )(datamat,
                                               deterministic = not training)
        
        # single LSTM layer
        out_carry, datamat = UnidirecLSTMLayer( config = self.config,
                                           name = self.name )(datamat=datamat,
                                                              datalens=datalens, 
                                                              carry=carry)
        
        return (out_carry, datamat)
        
        
        
class UnidirecLSTMLayerWithDropoutAfter(nn.Module):
    """
    An LSTM layer followed by dropout
    
    """
    config: dict
    name: str
    
    @nn.compact
    def __call__(self, 
                 datamat, 
                 datalens, 
                 training:bool, 
                 carry=None):
        # single LSTM layer
        out_carry, datamat = UnidirecLSTMLayer( config = self.config,
                                           name = self.name )(datamat=datamat,
                                                              datalens=datalens, 
                                                              carry=carry)
        
        # dropout
        do_rate = self.config["dropout"]
        datamat = nn.Dropout( rate = do_rate )( datamat,
                                                deterministic = not training )
        
        return (out_carry, datamat)
        


###############################################################################
### LSTM in both directions   #################################################
###############################################################################
class BidirecLSTMLayer(nn.Module):
    """
    Bi-directional LSTM
    
    """
    config: dict
    name: str
    
    def setup(self):
        #!!! hard code this 
        self.merge_how = 'concat'
        self.merge_fn = lambda a, b: jnp.concatenate([a, b], axis=-1)
        
        # for add: self.merge_fn = lambda a, b: jnp.add(a, b)
        
        
        ### unpack from config
        self.hidden_dim = self.config["hidden_dim"]
        
        
        ### LSTM layers
        # forward cell + wrapper
        self.fw_lstm_cell = nn.OptimizedLSTMCell(features=self.hidden_dim, 
                                              name=f'FW_{self.name}')
        self.fw_rnn_wrapper = nn.RNN(cell = self.fw_lstm_cell,
                                  return_carry = True)
        
        # reverse cell + wrapper
        self.rv_lstm_cell = nn.OptimizedLSTMCell(features=self.hidden_dim, 
                                              name=f'RV_{self.name}')
        self.rv_rnn_wrapper = nn.RNN(cell = self.rv_lstm_cell,
                                  return_carry = True)
        
        # bidirectional wrapper
        self.bidirectional_wrapper = nn.Bidirectional(forward_rnn = self.fw_rnn_wrapper, 
                                                      backward_rnn = self.rv_rnn_wrapper,
                                                      return_carry = True,
                                                      merge_fn = self.merge_fn,
                                                      name = self.name)
    
    def __call__(self, 
                 datamat, 
                 datalens, 
                 carry=None, 
                 **kwargs):
        out_carry, datamat = self.bidirectional_wrapper(inputs=datamat, 
                                                        initial_carry=carry,
                                                        seq_lengths=datalens)
        return (out_carry, datamat)
              
        
        
class BidirecLSTMLayerWithDropoutBefore(nn.Module):
    """
    Dropout THEN an LSTM layer 
    
    """
    config: dict
    name: str
    
    @nn.compact
    def __call__(self, 
                 datamat, 
                 datalens, 
                 training:bool, 
                 carry=None):
        # dropout
        do_rate = self.config["dropout"]
        datamat = nn.Dropout( rate = do_rate )( datamat = datamat,
                                                deterministic = not training )
        
        # single bidirectional LSTM layer
        out_carry, datamat = BidirecLSTMLayer( config = self.config,
                                               name = self.name )(datamat=datamat,
                                                                  datalens=datalens, 
                                                                  carry=carry)
        
        
        return (out_carry, datamat)  



class BidirecLSTMLayerWithDropoutAfter(nn.Module):
    """
    An LSTM layer followed by dropout 
    
    """
    config: dict
    name: str
    
    @nn.compact
    def __call__(self, 
                 datamat, 
                 datalens, 
                 training:bool, 
                 carry=None):
        # single bidirectional LSTM layer
        out_carry, datamat = BidirecLSTMLayer( config = self.config,
                                               name = self.name )(datamat=datamat,
                                                                  datalens=datalens, 
                                                                  carry=carry)
        
        # dropout
        do_rate = self.config["dropout"]
        datamat = nn.Dropout( rate = do_rate )( datamat = datamat, 
                                                deterministic = not training)
        
        return (out_carry, datamat)  
