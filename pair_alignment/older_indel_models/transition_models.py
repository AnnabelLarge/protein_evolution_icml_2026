#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 12 18:09:27 2025


 'OtherTransitionLogprobs',
 'TKF91TransitionLogprobsOldStyle',
 'TKF92TransitionLogprobsOldStyle',

 'OtherTransitionLogprobsFromFile',
 'TKF91TransitionLogprobsOldStyleFromFile',
 'TKF92TransitionLogprobsOldStyleFromFile',
"""
from flax import linen as nn
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import pickle

from utils.BaseClasses import ModuleBase
from latent_class_mixtures.model_functions import (bound_sigmoid,
                                                   safe_log,
                                                   log_one_minus_x,
                                                   switch_tkf,
                                                   regular_tkf,
                                                   approx_tkf)

from older_indel_models.h20_funcs import vmappedLogTransitionMatrix as h20_cond_logprob
from older_indel_models.indel_functions import KM03LogTransitionMatrix as km03_cond_logprob
from older_indel_models.indel_functions import LG05LogTransitionMatrix as lg05_cond_logprob
from older_indel_models.indel_functions import RS07LogTransitionMatrix as rs07_cond_logprob


###############################################################################
### TKF91: Fragment level   ###################################################
###############################################################################
class TKF91TransitionLogprobsOldStyle(ModuleBase):
    """
    TKF91 model, but specifically the CONDITIONAL logprob P(desc, align | anc)
      and using older activation functions
    
    B = batch size; number of samples
    T = number of branch lengths; this could be: 
        > an array of times for all samples (T; marginalize over these later)
        > an array of time per sample (T=B)
        > a quantized array of times per sample (T = T', where T' <= T)
    S: number of transition states (4 here: M, I, D, start/end)
        
    Initialize with
    ----------------
    config : dict 

        config["tkf_function"] : {'regular_tkf','approx_tkf','switch_tkf'}
            which function to use to solve for tkf parameters
        
        config["mu_range"] : Tuple, (2,)
            range for bound sigmoid activation that determines lamdba
            DEFAULT: -1e-4, 2
        
        config["offset_range"] : Tuple, (2,)
            range for bound sigmoid activation that determines offset 
            (which determines mu)
            DEFAULT: -1e-4, 0.333 (but ignored if tie_params is true)
        
        config['tie_params'] : bool
            if true, offset = 1e-4 (not learned)
            
    name : str
        class name, for flax
    
    
    Methods here
    ------------
    setup
    
    __call__
    
    fill_cond_tkf91
        fills in conditional TKF91 transition matrix
    
    _logits_to_indel_rates
        converts mu/offset logits to mu/offset values
    
    """
    config: dict
    name: str
    
    def setup(self):
        """
        B = batch size; number of samples
        T = number of branch lengths; this could be: 
            > an array of times for all samples (T; marginalize over these later)
            > an array of time per sample (T=B)
            > a quantized array of times per sample (T = T', where T' <= T)
        S: number of transition states (4 here: M, I, D, start/end)
            
            
        Flax Module Parameters
        -----------------------
        tkf_mu_offset_logits: ArrayLike (2,)
            first value is logit for mu, second is for offset
        
        """
        ### unpack config
        # defaults from old code
        self.num_domain_mixtures = 1
        self.num_fragment_mixtures = 1
        self.tie_params = self.config['tie_params']
        tkf_function_name = self.config['tkf_function']
        
        
        ### initialize logits for mu, offset
        # optional range values
        self.mu_min_val, self.mu_max_val = self.config.get( 'mu_range', [1e-4, 2] )
        self.offs_min_val, self.offs_max_val = self.config.get( 'offset_range', [1e-4, 0.333] )
        
        # the logits
        self.tkf_mu_offset_logits = self.param('tkf_mu_offset_logits',
                                               nn.initializers.normal(),
                                               (1,2),
                                               jnp.float32) #(1, 2)
        
        
        ### decide tkf function
        if tkf_function_name == 'regular_tkf':
            self.tkf_function = regular_tkf
        elif tkf_function_name == 'approx_tkf':
            self.tkf_function = approx_tkf
        elif tkf_function_name == 'switch_tkf':
            self.tkf_function = switch_tkf
    
    
    def __call__(self,
                 t_array,
                 sow_flax_intermeds: bool):
        
        """
        B = batch size; number of samples
        T = number of branch lengths; this could be: 
            > an array of times for all samples (T; marginalize over these later)
            > an array of time per sample (T=B)
            > a quantized array of times per sample (T = T', where T' <= T)
        S: number of transition states (4 here: M, I, D, start/end)
           
        
        Arguments
        ----------
        t_array : ArrayLike
            branch lengths, times for marginalizing over
        
        return_all_matrices : bool
            more a placeholder than anything, but if true, return
            the correction factor for tkf92 (if applicable)
        
        sow_flax_intermeds : bool
            switch for tensorboard logging
          
        Returns
        -------
        cond_matrix: ArrayLike (T,S,S)
            score transitions in conditional probability calculation
        
        (placeholder value)
        
        tkf_param_dict : dict
            alpha, beta, gamma (and all associated values)
        """
        # logits -> params
        # mu, offset are each (1,)
        mu, offset = self._logits_to_indel_rates(mu_offset_logits = self.tkf_mu_offset_logits)
        
        lam = mu * (1-offset) #(1,)
        indel_params = {'mu': mu, #(1,)
                        'lam': lam, #(1,)
                        'offset': offset} #(1,)
        
        
        ### get alpha, beta, gamma
        # contents of tkf_param_dict ( all ArrayLike[float32], (T,1) ):
        #   tkf_param_dict['log_alpha']
        #   tkf_param_dict['log_one_minus_alpha']
        #   tkf_param_dict['log_beta']
        #   tkf_param_dict['log_one_minus_beta']
        #   tkf_param_dict['log_gamma']
        #   tkf_param_dict['log_one_minus_gamma']
        tkf_param_dict = self.tkf_function( mu = mu, 
                                            offset = offset,
                                            t_array = t_array)
        
        # add to these dictionaries before filling out matrix
        tkf_param_dict['log_offset'] = jnp.log(offset) #(1,)
        tkf_param_dict['log_one_minus_offset'] = jnp.log1p(-offset) #(1,)
        
        
        ### maybe sow outputs
        self.maybe_sow( sow_flax_intermeds = sow_flax_intermeds,
                        vals = jnp.exp(tkf_param_dict['log_alpha']),
                        label = f'{self.name}/tkf91_alpha',
                        include_min_max = True,
                        include_perc_zeros = False)
        
        self.maybe_sow( sow_flax_intermeds = sow_flax_intermeds,
                        vals = jnp.exp(tkf_param_dict['log_beta']),
                        label = f'{self.name}/tkf91_beta',
                        include_min_max = True,
                        include_perc_zeros = False)
        
        self.maybe_sow( sow_flax_intermeds = sow_flax_intermeds,
                        vals = jnp.exp(tkf_param_dict['log_gamma']),
                        label = f'{self.name}/tkf91_gamma',
                        include_min_max = True,
                        include_perc_zeros = False)
        
        self.maybe_sow( sow_flax_intermeds = sow_flax_intermeds,
                        vals = lam,
                        label = f'{self.name}/tkf91_lambda',
                        include_min_max = True,
                        include_perc_zeros = False)
        
        self.maybe_sow( sow_flax_intermeds = sow_flax_intermeds,
                        vals = mu,
                        label = f'{self.name}/tkf91_mu',
                        include_min_max = True,
                        include_perc_zeros = False)
        
        
        ### get conditional matrix
        cond_matrix =  self.fill_cond_tkf91(tkf_param_dict) #(T, S, S)
        
        return cond_matrix, jnp.zeros( () ), tkf_param_dict
    
    
    def fill_cond_tkf91(self, tkf_param_dict):
        """
        B = batch size; number of samples
        T = number of branch lengths; this could be: 
            > an array of times for all samples (T; marginalize over these later)
            > an array of time per sample (T=B)
            > a quantized array of times per sample (T = T', where T' <= T)
        S: number of transition states (4 here: M, I, D, start/end)
        
        
        Arguments
        ----------
        tkf_param_dict : dict
            contains values for calculating matrix terms: lambda, mu, 
            alpha, beta, gamma, 1 - alpha, 1 - beta, 1 - gamma
            (all in log space); all are (T,1)
                  
        Returns
        -------
        out : ArrayLike, (T,S,S)
            joint loglike of transitions
        
        """
        ### entries in the matrix
        # smi_to_m = (1-beta)*alpha;     log(smi_to_m) = log(1-beta) + log(alpha)
        # smi_to_i = beta;               log(smi_to_i) = log(beta)
        # smi_to_d = (1-beta)*(1-alpha); log(smi_to_d) = log(1-beta) + log(1-alpha)
        # smi_to_e = 1 - beta;           log(smi_to_e) = log(1-beta)
        smi_to_m = (tkf_param_dict['log_one_minus_beta'] + 
                    tkf_param_dict['log_alpha']) #(T, 1)
        smi_to_i = tkf_param_dict['log_beta'] #(T, 1)
        smi_to_d = (tkf_param_dict['log_one_minus_beta'] + 
                    tkf_param_dict['log_one_minus_alpha']) #(T, 1)
        smi_to_e = tkf_param_dict['log_one_minus_beta'] #(T, 1)
        
        
        # d_to_m = (1-gamma)*alpha;     log(d_to_m) = log(1-gamma) + log(alpha)
        # d_to_i = gamma;               log(d_to_i) = log(gamma)
        # d_to_d = (1-gamma)*(1-alpha); log(d_to_d) = log(1-gamma) + log(1-alpha)
        # d_to_e = 1-gamma;             log(d_to_e) = log(1-gamma)
        d_to_m = (tkf_param_dict['log_one_minus_gamma'] + 
                    tkf_param_dict['log_alpha']) #(T, 1)
        d_to_i = tkf_param_dict['log_gamma'] #(T, 1)
        d_to_d = (tkf_param_dict['log_one_minus_gamma'] + 
                    tkf_param_dict['log_one_minus_alpha']) #(T, 1)
        d_to_e = tkf_param_dict['log_one_minus_gamma'] #(T, 1)
        
        out = jnp.stack([ jnp.stack([smi_to_m, smi_to_i, smi_to_d, smi_to_e], axis=-1),
                          jnp.stack([smi_to_m, smi_to_i, smi_to_d, smi_to_e], axis=-1),
                          jnp.stack([  d_to_m,   d_to_i,   d_to_d,   d_to_e], axis=-1),
                          jnp.stack([smi_to_m, smi_to_i, smi_to_d, smi_to_e], axis=-1)
                          ], axis=-2) #(T, 1, S, S)
        return out[:, 0, :, :] #(T, S, S)
    
    def _logits_to_indel_rates(self, 
                              mu_offset_logits):
        """
        Arguments
        ---------
        mu_offset_logits : ArrayLike, (1,2)
            logits to transform with bound sigmoid activation
            
        Returns
        -------
        mu : ArrayLike, (1,)
            delete rate
        
        offset : ArrayLike, (1,)
            used to calculate lambda: lambda = mu * (1 - offset)
        
        """
        ### mu
        mu = bound_sigmoid(x = mu_offset_logits[:,0],
                           min_val = self.mu_min_val,
                           max_val = self.mu_max_val) #(1,)
        
        ### Offset
        if self.tie_params:
            offset = jnp.array( [1e-4] ) #(1,)
            
        elif not self.tie_params:
            offset = bound_sigmoid(x = mu_offset_logits[:,1],
                                   min_val = self.offs_min_val,
                                   max_val = self.offs_max_val) #(1,)
        
        return (mu, offset)


class TKF91TransitionLogprobsOldStyleFromFile(TKF91TransitionLogprobsOldStyle):
    """
    like TKF91TransitionLogprobsOldStyle, but load values from a file
    
    NOTE: lambda and mu are provided directly, no need for offset
    
    B = batch size; number of samples
    T = number of branch lengths; this could be: 
        > an array of times for all samples (T; marginalize over these later)
        > an array of time per sample (T=B)
        > a quantized array of times per sample (T = T', where T' <= T)
    S: number of transition states (4 here: M, I, D, start/end)
        
        
    Initialize with
    ----------------
    config : dict
    
        config["tkf_function"] : {'regular_tkf','approx_tkf','switch_tkf'}
            which function to use to solve for tkf parameters
            
        config["filenames"]["tkf_params_file"]
            contains values for lambda, mu
            
    name : str
        class name, for flax
    
    
    Methods here
    ------------
    setup
    __call__
    
    
    Inherited from TKF91TransitionLogprobsOldStyle
    -----------------------------------------------
    fill_cond_tkf91
        fills in conditional TKF91 transition matrix
    
    _logits_to_indel_rates
        converts mu/offset logits to mu/offset values
    
    """
    config: dict
    name: str
    
    def setup(self):
        """
        
        Flax Module Parameters
        -----------------------
        None
        
        """
        # unpack config
        self.num_domain_mixtures = 1
        self.num_fragment_mixtures = 1
        in_file = self.config['filenames']['tkf_params_file']
        tkf_function_name = self.config['tkf_function']

        
        ### read file
        # lam and mu should be (1, )
        with open(in_file,'rb') as f:
            param_dict = _expand_vals_in_dict(pickle.load(f), 1) 
                
        param_dict = {k: jnp.array(v) for k,v in param_dict.items()}
        
        err = f'KEYS SEEN: {param_dict.keys()}'
        assert 'lambda' in param_dict.keys(), err
        assert 'mu' in param_dict.keys(), err
        self.param_dict = param_dict
          
        
        ### pick tkf function
        if tkf_function_name == 'regular_tkf':
            self.tkf_function = regular_tkf
        elif tkf_function_name == 'approx_tkf':
            self.tkf_function = approx_tkf
        elif tkf_function_name == 'switch_tkf':
            self.tkf_function = switch_tkf
    
    def __call__(self,
                 t_array,
                 sow_flax_intermeds: bool):
        """
        B = batch size; number of samples
        T = number of branch lengths; this could be: 
            > an array of times for all samples (T; marginalize over these later)
            > an array of time per sample (T=B)
            > a quantized array of times per sample (T = T', where T' <= T)
        S: number of transition states (4 here: M, I, D, start/end)
        
        
        Arguments
        ----------
        t_array : ArrayLike
            branch lengths, times for marginalizing over
        
        sow_flax_intermeds : bool
            switch for tensorboard logging
          
        Returns
        -------
        cond_matrix: ArrayLike (T,S,S)
            score transitions in conditional probability calculation
        
        tkf_param_dict : dict
            alpha, beta, gamma (and all associated values)
        """
        lam = self.param_dict['lambda'] #(1,)
        mu = self.param_dict['mu'] #(1,)
        offset = 1 - (lam /mu) #(1,)
        
        # get alpha, beta, gamma
        tkf_param_dict = self.tkf_function(mu = mu, 
                                        offset = offset,
                                        t_array = t_array)
        tkf_param_dict['log_offset'] = jnp.log(offset)
        tkf_param_dict['log_one_minus_offset'] = jnp.log1p(-offset)
        
        ### get conditional matrix
        cond_matrix =  self.fill_cond_tkf91(tkf_param_dict) #(T, S, S)
        
        return cond_matrix, jnp.zeros( () ), tkf_param_dict
        

###############################################################################
### TKF92   ###################################################################
###############################################################################
class TKF92TransitionLogprobsOldStyle(TKF91TransitionLogprobsOldStyle):
    """
    TKF92 model; used for calculating transitions in model of
        P(anc, desc, align)
    
    B = batch size; number of samples
    T = number of branch lengths; this could be: 
        > an array of times for all samples (T; marginalize over these later)
        > an array of time per sample (T=B)
        > a quantized array of times per sample (T = T', where T' <= T)
        
        
    Initialize with
    ----------------
    config : dict
        config["tkf_function"] : {'regular_tkf','approx_tkf','switch_tkf'}
            which function to use to solve for tkf parameters
            
        config["mu_range"] : Tuple, (2,)
            range for bound sigmoid activation that determines mu
            DEFAULT: -1e-4, 2
        
        config["offset_range"] : Tuple, (2,)
            range for bound sigmoid activation that determines offset 
            (which determines mu)
            DEFAULT: -1e-4, 0.333 (ignored if tying parameters)
            
        config["r_range"]
            range for bound sigmoid activation that determines TKF r
            DEFAULT: -1e-4, 0.999
            
        config['tie_params'] : bool
            if true, offset = 1e-4 (not learned)
            
    name : str
        class name, for flax
    
    
    Methods here
    ------------
    setup
    
    __call__
    
    fill_cond_tkf92
        fills in conditional TKF92 transition matrix
        
        
    Inherited from TKF91TransitionLogprobsOldStyle
    ------------------------------------------------
    fill_cond_tkf91
        fills in conditional TKF91 transition matrix
    
    _logits_to_indel_rates
        converts mu/offset logits to mu/offset values
    
    """
    config: dict
    name: str
    
    def setup(self):
        """
        B = batch size; number of samples
        T = number of branch lengths; this could be: 
            > an array of times for all samples (T; marginalize over these later)
            > an array of time per sample (T=B)
            > a quantized array of times per sample (T = T', where T' <= T)
        S: number of transition states (4 here: M, I, D, start/end)
        
        
        Flax Module Parameters
        -----------------------
        tkf_mu_offset_logits : ArrayLike (1, 2)
            first value is logit for mu, second is for offset
        
        r_extend_logits : ArrayLike (1,)
            logits for TKF fragment extension probability, r
            this is EXCLUSIVELY for the fragment-level tkf92 indel process
        
        """
        ### unpack config
        # required
        self.num_domain_mixtures = 1
        self.num_fragment_mixtures = 1
        self.tie_params = self.config['tie_params']
        tkf_function_name = self.config['tkf_function']
        
        # optional inputs
        self.mu_min_val, self.mu_max_val = self.config.get( 'mu_range', [1e-4, 2] )
        self.offs_min_val, self.offs_max_val = self.config.get( 'offset_range', [1e-4, 0.333] )
        self.r_extend_min_val, self.r_extend_max_val = self.config.get( 'r_range', [1e-4, 0.999] )
        
        
        ### init flax parameters
        # initialize logits for mu, offset
        self.tkf_mu_offset_logits = self.param('tkf_mu_offset_logits',
                                               nn.initializers.normal(),
                                               (self.num_domain_mixtures, 2),
                                               jnp.float32) #(1, 2)
        
        # initializing r extension prob
        self.r_extend_logits = self.param('r_extend_logits',
                                          nn.initializers.normal(),
                                          (self.num_domain_mixtures, self.num_fragment_mixtures),
                                          jnp.float32) #(1, 1)
        
        
        ### decide tkf function
        if tkf_function_name == 'regular_tkf':
            self.tkf_function = regular_tkf
        elif tkf_function_name == 'approx_tkf':
            self.tkf_function = approx_tkf
        elif tkf_function_name == 'switch_tkf':
            self.tkf_function = switch_tkf
        
    
    def __call__(self,
                 t_array,
                 sow_flax_intermeds: bool):
        """
        B = batch size; number of samples
        T = number of branch lengths; this could be: 
            > an array of times for all samples (T; marginalize over these later)
            > an array of time per sample (T=B)
            > a quantized array of times per sample (T = T', where T' <= T)
        S: number of transition states (4 here: M, I, D, start/end)
           
        
        Arguments
        ----------
        t_array : ArrayLike, (T,)
            branch lengths, times for marginalizing over
        
        sow_flax_intermeds : bool
            switch for tensorboard logging
          
        Returns
        -------
        cond_matrix: ArrayLike (T,S,S)
            score transitions in conditional probability calculation
            
        tkf_param_dict : dict
            alpha, beta, gamma (and all associated values)
        """
        ### TKF92 model
        # mu, offset are each (1,)
        mu, offset = self._logits_to_indel_rates(mu_offset_logits = self.tkf_mu_offset_logits)
        
        lam = mu * (1-offset) #(1,)
        indel_params = {'mu': mu, #(1,)
                        'lam': lam, #(1,)
                        'offset': offset} #(1,)
        
        # r_extend
        r_extend = bound_sigmoid(x = self.r_extend_logits,
                                 min_val = self.r_extend_min_val,
                                 max_val = self.r_extend_max_val) # (1, 1)
        
        indel_params = {'mu': mu, #(1,)
                        'lam': lam, #(1,)
                        'offset': offset, #(1,)
                        'r_extend': r_extend} #(1,1)
        
        
        ### get alpha, beta, gamma
        # contents of tkf_param_dict ( all ArrayLike[float32], (T,1) ):
        #   tkf_param_dict['log_alpha']
        #   tkf_param_dict['log_one_minus_alpha']
        #   tkf_param_dict['log_beta']
        #   tkf_param_dict['log_one_minus_beta']
        #   tkf_param_dict['log_gamma']
        #   tkf_param_dict['log_one_minus_gamma']
        tkf_param_dict = self.tkf_function(mu = mu, 
                                            offset = offset,
                                            t_array = t_array)
        
        # add to these dictionaries before filling out matrix
        tkf_param_dict['log_offset'] = jnp.log(offset) #(1,)
        tkf_param_dict['log_one_minus_offset'] = jnp.log1p(-offset) #(1,)
        
        # maybe sow outputs
        self.maybe_sow( sow_flax_intermeds = sow_flax_intermeds,
                        vals = jnp.exp(tkf_param_dict['log_alpha']),
                        label = f'{self.name}/tkf92_alpha',
                        include_min_max = True,
                        include_perc_zeros = False)
        
        self.maybe_sow( sow_flax_intermeds = sow_flax_intermeds,
                        vals = jnp.exp(tkf_param_dict['log_beta']),
                        label = f'{self.name}/tkf92_beta',
                        include_min_max = True,
                        include_perc_zeros = False)
        
        self.maybe_sow( sow_flax_intermeds = sow_flax_intermeds,
                        vals = jnp.exp(tkf_param_dict['log_gamma']),
                        label = f'{self.name}/tkf92_gamma',
                        include_min_max = True,
                        include_perc_zeros = False)
        
        self.maybe_sow( sow_flax_intermeds = sow_flax_intermeds,
                        vals = lam,
                        label = f'{self.name}/tkf92_lambda',
                        include_min_max = True,
                        include_perc_zeros = False)
        
        self.maybe_sow( sow_flax_intermeds = sow_flax_intermeds,
                        vals = mu,
                        label = f'{self.name}/tkf92_mu',
                        include_min_max = True,
                        include_perc_zeros = False)
        
        self.maybe_sow( sow_flax_intermeds = sow_flax_intermeds,
                        vals = r_extend,
                        label = f'{self.name}/tkf92_r_extension_prob',
                        include_min_max = True,
                        include_perc_zeros = False)
        
        
        ### conditional prob matrix
        cond_matrix =  self.fill_cond_tkf92(tkf_param_dict=tkf_param_dict, 
                                              r_extend=r_extend) # (T, S_from=4, S_to=4)
        log_corr = jnp.log(lam/mu) - jnp.log( r_extend + (1-r_extend)*(lam/mu) ) #()
        log_corr = jnp.squeeze(log_corr) #()
        
        return cond_matrix, log_corr, tkf_param_dict
    
    
    def fill_cond_tkf92(self,
                        tkf_param_dict,
                        r_extend):
        """
        B = batch size; number of samples
        T = number of branch lengths; this could be: 
            > an array of times for all samples (T; marginalize over these later)
            > an array of time per sample (T=B)
            > a quantized array of times per sample (T = T', where T' <= T)
        S: number of transition states (4 here: M, I, D, start/end)
        
        
        Arguments
        ----------
        tkf_param_dict : dict
            contains values for calculating matrix terms: lambda, mu, 
            alpha, beta, gamma, 1 - alpha, 1 - beta, 1 - gamma
            (all in log space); all are (T,1)
        
        r_extend : ArrayLike, (1,1)
            fragment extension probabilities
        
        Returns
        -------
        out : ArrayLike, (T, 1, C_{frag_from}, C_{frag_to}, S_from=4, S_to=4)
            joint loglike of transitions
        
        """
        ### need conditional TKF91 for this 
        log_U = self.fill_cond_tkf91(tkf_param_dict) #(T, S_from, S_to)
        log_U = log_U[:, None, :, :] #(T, 1, S_from, S_to)
        
        # dims
        T = tkf_param_dict['log_alpha'].shape[0]
        C_dom = 1
        C_frag = 1
        S = log_U.shape[-1] #number of hidden states (like M, I, D, and start/end)
        
        # converted log values; expand
        log_r_extend = safe_log( r_extend ) #(1, 1)
        log_one_minus_r = log_one_minus_x(log_r_extend) #(1, 1)
        
        # kappa = (lambda / mu) = 1 - offset
        log_kappa = tkf_param_dict['log_one_minus_offset'] #(1, 1)
        
        # nu = r + (1-r) kappa
        log_nu = jnp.logaddexp( log_r_extend,
                                log_one_minus_r + log_kappa ) #(1, 1)
        log_one_div_nu = -log_nu #(1, 1)
        
        
        ### M -> any
        # m -> m: (1/nu) ( r + (1-r) kappa U_{m,m} )
        m_to_m = log_one_minus_r + log_kappa + log_U[:, :, 0, 0] #(T, 1)
        m_to_m = jnp.logaddexp( log_r_extend, m_to_m ) #(T, 1)
        m_to_m = log_one_div_nu + m_to_m #(T, 1)
        
        # m -> i: (1-r) U_{m,i}
        m_to_i = log_one_minus_r + log_U[:, :, 0, 1] #(T, 1)
        
        # m -> d: (1/nu) (1-r) kappa U_{m,d}
        m_to_d = ( log_one_div_nu + 
                   log_one_minus_r + 
                   log_kappa + 
                   log_U[:, :, 0, 2] ) #(T, 1)
        
        # m -> e: same as TKF91
        m_to_e = log_U[:, :, 0, 3] #(T, 1)
        
        match_to_any = jnp.stack( [m_to_m,
                                   m_to_i,
                                   m_to_d,
                                   m_to_e], axis=-1 ) #(T, 1, 4)
        
        
        ### I -> any
        # i -> m: (1/nu) (1-r) kappa U_{i,m}
        i_to_m = ( log_one_div_nu +
                   log_one_minus_r + 
                   log_kappa +
                   log_U[:, :, 1, 0] ) #(T, 1)
        
        # i -> i: r + (1-r) U_{i,i}
        i_to_i = log_one_minus_r + log_U[:, :, 1, 1] #(T, 1)
        i_to_i = jnp.logaddexp( log_r_extend, i_to_i ) #(T, 1)
        
        # i -> d: (1/nu) (1-r) kappa U_{i,d}
        i_to_d = ( log_one_div_nu + 
                   log_one_minus_r + 
                   log_kappa + 
                   log_U[:, :, 1, 2] ) #(T, 1)
        
        # i -> e: same as TKF91
        i_to_e = log_U[:, :, 1, 3] #(T, 1)
        
        ins_to_any = jnp.stack( [i_to_m,
                                 i_to_i,
                                 i_to_d,
                                 i_to_e], axis=-1 ) #(T, 1, 4)
      
        del i_to_m, i_to_i, i_to_d, i_to_e
        
        
        ### D -> any
        # d -> m: (1/nu) (1-r) kappa U_{d,m}
        d_to_m = ( log_one_div_nu + 
                   log_one_minus_r + 
                   log_kappa +
                   log_U[:, :, 2, 0] ) #(T, 1)
        
        # d -> i: (1-r) U_{d,i}
        d_to_i = log_one_minus_r + log_U[:, :, 2, 1] #(T, 1)
        
        # d -> d: (1/nu) ( r + (1-r) kappa U_{d,d} )
        d_to_d = log_one_minus_r + log_kappa + log_U[:, :, 2, 2] #(T, 1)
        d_to_d = jnp.logaddexp( log_r_extend, d_to_d ) #(T, 1)
        d_to_d = log_one_div_nu + d_to_d #(T, 1)
        
        # d -> e: same as TKF91
        d_to_e = log_U[:, :, 2, 3] #(T, 1)
        
        del_to_any = jnp.stack( [d_to_m,
                                 d_to_i,
                                 d_to_d,
                                 d_to_e], axis=-1 ) #(T, 1, 4)
        
        
        ### final matrix
        # start -> any same as TKF91
        start_to_any = log_U[:, :, 3, :] #(T, 1, 4)
        log_tkf92_rate_mat = jnp.concatenate( [match_to_any,
                                               ins_to_any,
                                               del_to_any,
                                               start_to_any], axis=1 ) #(T, 4, 4)
        
        return log_tkf92_rate_mat #(T, 4, 4)
    

class TKF92TransitionLogprobsOldStyleFromFile(TKF92TransitionLogprobsOldStyle):
    """
    like TKF92TransitionLogprobsOldStyle, but load values from a file
    
    NOTE: lambda and mu are provided directly, no need for offset
    
    B = batch size; number of samples
    T = number of branch lengths; this could be: 
        > an array of times for all samples (T; marginalize over these later)
        > an array of time per sample (T=B)
        > a quantized array of times per sample (T = T', where T' <= T)
    S: number of transition states (4 here: M, I, D, start/end)
        
        
    Initialize with
    ----------------
    config : dict
    
        config["tkf_function"] : {'regular_tkf','approx_tkf','switch_tkf'}
            which function to use to solve for tkf parameters
            
        config["filenames"]["tkf_params_file"]
            contains values for lambda, mu
            
        config["tie_params"] : bool
            whether or not to tie indel rates
            offset = 1e-4
            
    name : str
        class name, for flax
    
    
    Methods here
    ------------
    setup
    __call__
    
    
    Inherited from TKF92TransitionLogprobsOldStyle
    -----------------------------------------------
    fill_cond_tkf92
        fills in conditional TKF92 transition matrix
    
    _logits_to_indel_rates
        converts mu/offset logits to mu/offset values
    
    return_all_matrices
        return one intermediate; mostly kept for compatibility
    
    """
    config: dict
    name: str
    
    def setup(self):
        """
        
        Flax Module Parameters
        -----------------------
        None
        
        """
        # unpack config
        self.num_domain_mixtures = 1
        self.num_fragment_mixtures = 1
        in_file = self.config['filenames']['tkf_params_file']
        tkf_function_name = self.config['tkf_function']

        
        ### read file
        # lam and mu should be (1, )
        with open(in_file,'rb') as f:
            param_dict = pickle.load(f)
                
        param_dict = {k: jnp.array(v) for k,v in param_dict.items()}
        
        err = f'KEYS SEEN: {param_dict.keys()}'
        assert 'lambda' in param_dict.keys(), err
        assert 'mu' in param_dict.keys(), err
        assert 'r_extend' in param_dict.keys(), err
        self.param_dict = param_dict
          
        
        ### pick tkf function
        if tkf_function_name == 'regular_tkf':
            self.tkf_function = regular_tkf
        elif tkf_function_name == 'approx_tkf':
            self.tkf_function = approx_tkf
        elif tkf_function_name == 'switch_tkf':
            self.tkf_function = switch_tkf
    
    def __call__(self,
                 t_array,
                 sow_flax_intermeds: bool):
        """
        B = batch size; number of samples
        T = number of branch lengths; this could be: 
            > an array of times for all samples (T; marginalize over these later)
            > an array of time per sample (T=B)
            > a quantized array of times per sample (T = T', where T' <= T)
        S: number of transition states (4 here: M, I, D, start/end)
        
        
        Arguments
        ----------
        t_array : ArrayLike
            branch lengths, times for marginalizing over
        
        sow_flax_intermeds : bool
            switch for tensorboard logging
          
        Returns
        -------
        cond_matrix: ArrayLike (T,S,S)
            score transitions in conditional probability calculation
            
        tkf_param_dict : dict
            alpha, beta, gamma (and all associated values)
        """
        lam = self.param_dict['lambda'] #(1,)
        mu = self.param_dict['mu'] #(1,)
        offset = 1 - (lam /mu) #(1,)
        r_extend = self.param_dict['r_extend'] #(1, 1)
        
        indel_params = {'mu': mu, #(1,)
                        'lam': lam, #(1,)
                        'offset': offset, #(1,)
                        'r_extend': r_extend} #(1,1)
        
        # get alpha, beta, gamma
        tkf_param_dict = self.tkf_function(mu = mu, 
                                        offset = offset,
                                        t_array = t_array)
        tkf_param_dict['log_offset'] = jnp.log(offset)
        tkf_param_dict['log_one_minus_offset'] = jnp.log1p(-offset)
        
        
        ### get conditional matrix
        cond_matrix =  self.fill_cond_tkf92(tkf_param_dict = tkf_param_dict,
                                            r_extend = r_extend) # (T, S_from=4, S_to=4)
        log_corr = jnp.log(lam/mu) - jnp.log( r_extend + (1-r_extend)*(lam/mu) ) #()
        log_corr = jnp.squeeze(log_corr) #()
        
        return cond_matrix, log_corr, tkf_param_dict
            


###############################################################################
### All others: H20, RS07, LG05, KM03   #######################################
###############################################################################
class OtherTransitionLogprobs(ModuleBase):
    """
    used for calculating transitions in model of P(desc, align | anc)
    
    For all of these, start and end are treated as match states!
    
    
    B = batch size; number of samples
    T = number of branch lengths; this could be: 
        > an array of times for all samples (T; marginalize over these later)
        > an array of time per sample (T=B)
        > a quantized array of times per sample (T = T', where T' <= T)
    S: number of transition states (4 here: M, I, D, start/end)
        
    Initialize with
    ----------------
    config : dict 
        config["indel_model"] : {'h20', 'rs07', 'lg05', 'km03'}
            which indel model
        
        config["lambda_range"] : Tuple, (2,)
            range for bound sigmoid activation that determines lambda
            DEFAULT: -1e-4, 2
            
        config["mu_range"] : Tuple, (2,)
            range for bound sigmoid activation that determines mu
            DEFAULT: -1e-4, 2
        
        config["x_range"] : Tuple, (2,)
            range for bound sigmoid activation that determines x
            DEFAULT: -1e-4, 0.999
            
        config["y_range"] : Tuple, (2,)
            range for bound sigmoid activation that determines y
            DEFAULT: -1e-4, 0.999

        config["tie_params"] : bool
            if true, then lambda = mu, x = y
            
    name : str
        class name, for flax
    
    
    Methods here
    ------------
    setup
    
    __call__
    
    _logits_to_indel_rates
        converts logits to values
    
    """
    config: dict
    name: str
    
    def setup(self):
        ### unpack config
        self.tie_params = self.config['tie_params']
        self.indel_model = self.config['indel_model_type']
        
        # pick a transition function
        if self.indel_model == 'h20':
            self.get_cond_logprob = h20_cond_logprob
        elif self.indel_model == 'km03':
            self.get_cond_logprob = km03_cond_logprob
        elif self.indel_model == 'lg05':
            self.get_cond_logprob = lg05_cond_logprob
        elif self.indel_model == 'rs07':
            self.get_cond_logprob = rs07_cond_logprob
        
        # initialize logits for lambda, mu, x, and y
        self.indel_logits = self.param('indel_logits',
                                        nn.initializers.normal(),
                                        (4,),
                                        jnp.float32) #(4,)
        
        # define limits
        self.lambda_min_val, self.lambda_max_val = self.config.get( 'lambda_range', [1e-4, 2] )
        self.mu_min_val, self.mu_max_val = self.config.get( 'mu_range', [1e-4, 2] )
        self.x_min_val, self.x_max_val = self.config.get( 'x_range', [1e-4, 0.999] )
        self.y_min_val, self.y_max_val = self.config.get( 'y_range', [1e-4, 0.999] )
        
        
    def __call__(self,
                 t_array,
                 sow_flax_intermeds: bool,
                 *args,
                 **kwargs):
        """
        B = batch size; number of samples
        T = number of branch lengths; this could be: 
            > an array of times for all samples (T; marginalize over these later)
            > an array of time per sample (T=B)
            > a quantized array of times per sample (T = T', where T' <= T)
        S: number of transition states (4 here: M, I, D, start/end)
           
        
        Arguments
        ----------
        t_array : ArrayLike, (T,)
            branch lengths, times for marginalizing over
        
        sow_flax_intermeds : bool
            switch for tensorboard logging
          
        Returns
        -------
        cond_matrix: ArrayLike (T,S,S)
            score transitions in conditional probability calculation

        two placeholder objects
        """
        lam, mu, x, y = self._logits_to_indel_rates( indel_logits = self.indel_logits ) #(4,)
        
        indel_params = {'lam': lam, #(1,)
                        'mu': mu, #(1,)
                        'x': x, #(1,)
                        'y': y} #(1,1)
        
        # maybe sow outputs
        self.maybe_sow( sow_flax_intermeds = sow_flax_intermeds,
                        vals = lam,
                        label = f'{self.name}/{self.indel_model}_lambda',
                        include_min_max = True,
                        include_perc_zeros = False)
        
        self.maybe_sow( sow_flax_intermeds = sow_flax_intermeds,
                        vals = mu,
                        label = f'{self.name}/{self.indel_model}_mu',
                        include_min_max = True,
                        include_perc_zeros = False)
        
        self.maybe_sow( sow_flax_intermeds = sow_flax_intermeds,
                        vals = x,
                        label = f'{self.name}/{self.indel_model}_x',
                        include_min_max = True,
                        include_perc_zeros = False)
        
        self.maybe_sow( sow_flax_intermeds = sow_flax_intermeds,
                        vals = y,
                        label = f'{self.name}/{self.indel_model}_y',
                        include_min_max = True,
                        include_perc_zeros = False)
        
        
        ### conditional prob matrix
        # these are only 3x3
        cond_matrix =  self.get_cond_logprob( lam = lam,
                                              mu = mu,
                                              x = x,
                                              y = y,
                                              t_array = t_array ) # (T, 3, 3)
        
        # start -> any should be treated like match -> any
        start_to_mid = cond_matrix[:, 0, :] #(T, 3)
        start_to_end = start_to_mid[:, 0][:,None] #(T,1)
        start_to_any = jnp.concatenate( [start_to_mid, start_to_end], axis=-1 ) #(T, 4)
        
        # any -> end should be treated like any -> match
        match_to_end = cond_matrix[:, 0, 0] #(T,)
        ins_to_end = cond_matrix[:, 1, 0] #(T,)
        del_to_end = cond_matrix[:, 2, 0] #(T,)
        mid_to_end = jnp.stack( [match_to_end, ins_to_end, del_to_end], axis=-1 ) #(T, 3)
        mid_to_end = mid_to_end[:, :, None] #(T, 3, 1)
        
        # augment original matrix
        cond_matrix_aug = jnp.concatenate( [cond_matrix, mid_to_end], axis=2 ) #(T, 3, 4)
        cond_matrix_aug = jnp.concatenate( [cond_matrix_aug, start_to_any[:, None, :]], axis=1 ) #(T, 4, 4)
        
        return cond_matrix_aug, jnp.zeros( () ), None
    
    def _logits_to_indel_rates( self,
                                indel_logits ):
        lam_logits = indel_logits[0][None]
        mu_logits = indel_logits[1][None]
        x_logits = indel_logits[2][None]
        y_logits = indel_logits[3][None]
        
        mu = bound_sigmoid(x = mu_logits,
                           min_val = self.mu_min_val,
                           max_val = self.mu_max_val)
        
        x = bound_sigmoid( x = x_logits,
                           min_val = self.x_min_val,
                           max_val = self.x_max_val) 
        
        if self.tie_params:
            lam = mu #(1,)
            y = x #(1,)
        
        elif not self.tie_params: 
            lam = bound_sigmoid(x = lam_logits,
                               min_val = self.lambda_min_val,
                               max_val = self.lambda_max_val)
            
            y = bound_sigmoid( x = y_logits,
                               min_val = self.y_min_val,
                               max_val = self.y_max_val) 
            
        return lam, mu, x, y
    
    
class OtherTransitionLogprobsFromFile(OtherTransitionLogprobs):
    """
    like OtherTransitionLogprobs, but load values from a file
    
    B = batch size; number of samples
    T = number of branch lengths; this could be: 
        > an array of times for all samples (T; marginalize over these later)
        > an array of time per sample (T=B)
        > a quantized array of times per sample (T = T', where T' <= T)
    S: number of transition states (4 here: M, I, D, start/end)
        
        
    Initialize with
    ----------------
    config : dict
    
        config["tkf_function"] : {'regular_tkf','approx_tkf','switch_tkf'}
            which function to use to solve for tkf parameters
            
        config["filenames"]["tkf_params_file"]
            contains values for lambda, mu
        
        config["tie_params"] : bool
            whether or not to tie parameter values
            lambda = mu
            x = y
            
    name : str
        class name, for flax
    
    
    Methods here
    ------------
    setup
    __call__
    
    
    Inherited from OtherTransitionLogprobs
    -----------------------------------------------
    _logits_to_indel_rates
        converts mu/offset logits to mu/offset values
    
    """
    config: dict
    name: str
    
    def setup(self):
        ### unpack config
        self.tie_params = self.config['tie_params']
        self.indel_model = self.config['indel_model']
        
        # pick a transition function
        if self.indel_model == 'h20':
            self.get_cond_logprob = h20_cond_logprob
        elif self.indel_model == 'km03':
            self.get_cond_logprob = km03_cond_logprob
        elif self.indel_model == 'lg05':
            self.get_cond_logprob = lg05_cond_logprob
        elif self.indel_model == 'rs07':
            self.get_cond_logprob = rs07_cond_logprob
        
        
        ### read file
        in_file = self.config['filenames']['tkf_params_file']
        with open(in_file,'rb') as f:
            param_dict = pickle.load(f)

        param_dict = {k: jnp.array(v) for k,v in param_dict.items()}

        err = f'KEYS SEEN: {param_dict.keys()}'
        assert 'lam' in param_dict.keys(), err
        assert 'mu' in param_dict.keys(), err
        assert 'x' in param_dict.keys(), err
        assert 'y' in param_dict.keys(), err
        
        self.param_dict = param_dict
        
        
    def __call__(self,
                 t_array,
                 *args,
                 **kwargs):
        """
        B = batch size; number of samples
        T = number of branch lengths; this could be: 
            > an array of times for all samples (T; marginalize over these later)
            > an array of time per sample (T=B)
            > a quantized array of times per sample (T = T', where T' <= T)
        S: number of transition states (4 here: M, I, D, start/end)
           
        
        Arguments
        ----------
        t_array : ArrayLike, (T,)
            branch lengths, times for marginalizing over
        
        sow_flax_intermeds : bool
            switch for tensorboard logging
          
        Returns
        -------
        cond_matrix_aug: ArrayLike (T,S,S)
            score transitions in conditional probability calculation

        two placeholder values
        """
        lam = self.param_dict['lam']
        mu = self.param_dict['mu']
        x = self.param_dict['x']
        y = self.param_dict['y']
        
        indel_params = {'lam': lam, #(1,)
                        'mu': mu, #(1,)
                        'x': x, #(1,)
                        'y': y} #(1,)
        
        
        ### conditional prob matrix
        # these are only 3x3
        cond_matrix =  self.get_cond_logprob( lam = lam,
                                              mu = mu,
                                              x = x,
                                              y = y,
                                              t_array = t_array ) # (T, 3, 3)
        
        # start -> any should be treated like match -> any
        start_to_mid = cond_matrix[:, 0, :] #(T, 3)
        start_to_end = start_to_mid[:, 0][:,None] #(T,1)
        start_to_any = jnp.concatenate( [start_to_mid, start_to_end], axis=-1 ) #(T, 4)
        start_to_any = start_to_any[:, None, :] #(T, 1, 4)
        
        # any -> end should be treated like any -> match
        match_to_end = cond_matrix[:, 0, 0] #(T,)
        ins_to_end = cond_matrix[:, 1, 0] #(T,)
        del_to_end = cond_matrix[:, 2, 0] #(T,)
        mid_to_end = jnp.stack( [match_to_end, ins_to_end, del_to_end], axis=-1 ) #(T, 3)
        mid_to_end = mid_to_end[:, :, None] #(T, 3, 1)
        
        # augment original matrix
        cond_matrix_aug = jnp.concatenate( [cond_matrix, mid_to_end], axis=2 ) #(T, 3, 4)
        cond_matrix_aug = jnp.concatenate( [cond_matrix_aug, start_to_any], axis=1 ) #(T, 4, 4)
        
        return cond_matrix_aug, jnp.zeros( () ), None
    
    