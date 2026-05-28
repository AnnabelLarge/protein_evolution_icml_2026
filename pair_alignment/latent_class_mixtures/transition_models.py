#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 14:42:28 2024



modules:
=========

originals:
------------
'GeomLenTransitionLogprobs',
'TKF91TransitionLogprobs',
'TKF92TransitionLogprobs',
'TKF91DomainTransitionLogprobs',

loading from files:
--------------------
'GeomLenTransitionLogprobsFromFile',
'TKF91TransitionLogprobsFromFile',
'TKF92TransitionLogprobsFromFile',
'TKF91DomainTransitionLogprobsFromFile',

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
                                                   approx_tkf,                                                              
                                                   get_tkf91_single_seq_marginal_transition_logprobs,
                                                   get_tkf92_single_seq_marginal_transition_logprobs,
                                                   get_cond_transition_logprobs)

def _expand_vals_in_dict(d, expected_ndims):
    tkf_param_dict = {}
    for k, v in d.items():
        while len(v.shape) < expected_ndims:
            v = v[None,...]
        tkf_param_dict[k] = v
    return tkf_param_dict

def _expand_arr_in_dict(d, key, expected_ndims):
    """
    modifies the dictionary in-place
    """
    v = d[key]
    while len(v.shape) < expected_ndims:
        v = v[None,...]
    d[key] = v
    return d

 
###############################################################################
### TKF91: Fragment level   ###################################################
###############################################################################
class TKF91TransitionLogprobs(ModuleBase):
    """
    TKF91 model; used for calculating transitions in model of
        P(anc, desc, align)
    
    C_dom: number of mixtures associated with nested TKF92 models (domain-level)
        > for tkf91, there can NOT be mixtures over transitions (i.e. C_dom=1)
    C_frag: number of mixtures associated with TKF92 fragments (fragment-level)
        > for tkf91, there can NOT be mixtures over transitions (i.e. C_frag=1)
    B = batch size; number of samples
    T = number of branch lengths; this could be: 
        > an array of times for all samples (T; marginalize over these later)
        > an array of time per sample (T=B)
        > a quantized array of times per sample (T = T', where T' <= T)
    S: number of transition states (4 here: M, I, D, start/end)
        
    Initialize with
    ----------------
    config : dict 

        config["num_domain_mixtures"] : int
            number of domain mixtures (for nested TKF model)
            
        config["num_fragment_mixtures"] : int
            number of tkf92 fragments (none here)
            
        config["tkf_function"] : {'regular_tkf','approx_tkf','switch_tkf'}
            which function to use to solve for tkf parameters
    
        config["mu_range"] : Tuple, (2,)
            range for bound sigmoid activation that determines lamdba
            DEFAULT: -1e-4, 2
        
        config["offset_range"] : Tuple, (2,)
            range for bound sigmoid activation that determines offset 
            (which determines mu)
            DEFAULT: -1e-4, 0.333
            
    name : str
        class name, for flax
    
    
    Methods here
    ------------
    setup
    
    __call__
    
    fill_joint_tkf91
        fills in joint TKF91 transition matrix
    
    _logits_to_indel_rates
        converts mu/offset logits to mu/offset values
    
    return_all_matrices
        return transition matrices used for joint, marginal, and conditional 
        log-likelihood transitions
    
    """
    config: dict
    name: str
    
    def setup(self):
        """
        C_dom: number of mixtures associated with nested TKF92 models (domain-level)
        C_frag: number of mixtures associated with TKF92 fragments (fragment-level)
            > for tkf91, there can NOT be mixtures over transitions (i.e. C_frag=1)
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
        # required
        self.num_domain_mixtures = self.config['num_domain_mixtures']
        self.num_fragment_mixtures = self.config['num_fragment_mixtures'] 
        tkf_function_name = self.config['tkf_function']
        
        # optional
        self.mu_min_val, self.mu_max_val = self.config.get( 'mu_range', [1e-4, 2] )
        self.offs_min_val, self.offs_max_val = self.config.get( 'offset_range', [1e-4, 0.333] )
        
        # no fragment mixtures possible, so all P(c_frag | c_dom) = 1
        assert self.num_fragment_mixtures == 1
        self.log_frag_class_probs = jnp.zeros( (self.num_domain_mixtures, 1) ) #(C_dom, 1)
        
        
        ### initialize logits for mu, offset
        self.tkf_mu_offset_logits = self.param('tkf_mu_offset_logits',
                                               nn.initializers.normal(),
                                               (1,2) ) #(C_dom, 2)
        
        
        ### decide tkf function
        if tkf_function_name == 'regular_tkf':
            self.tkf_function = regular_tkf
        elif tkf_function_name == 'approx_tkf':
            self.tkf_function = approx_tkf
        elif tkf_function_name == 'switch_tkf':
            self.tkf_function = switch_tkf
    
    def __call__(self,
                 t_array,
                 return_all_matrices: bool,
                 sow_flax_intermeds: bool):
        
        """
        C_dom: number of mixtures associated with nested TKF92 models (domain-level)
        C_frag: number of mixtures associated with TKF92 fragments (fragment-level)
            > for tkf91, there can NOT be mixtures over transitions (i.e. C_frag=1)
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
            if false, only return the joint (used for model training)
            if true, return joint, conditional, and marginal matrices
        
        sow_flax_intermeds : bool
            switch for tensorboard logging
          
        Returns
        -------
        log_fragment_class_probs : (0, 0)
            placeholder tuple
        
        matrix_dict : dict
            matrix_dict["joint"]: (T,C_dom,S,S)
                score transitions in joint probability calculation
            
            matrix_dict["lam"]: (C_dom,)
                insert rates
                
            matrix_dict["mu"]: (C_dom,)
                delete rates
            
            matrix_dict["offset"]: (C_dom,)
                1 - (lam/mu)
            
            (if return_all_matrices) matrix_dict["marginal"]: (C_dom,2,2)
                score transitions in marginal probability calculation
            
            (if return_all_matrices) matrix_dict["conditional"]: (T,C_dom,S,S)
                score transitions in conditional probability calculation
        
            (if return_all_matrices) matrix_dict["corr"]: 0
                placeholder value
                
        tkf_param_dict : dict
            alpha, beta, gamma (and all associated values)
        """
        # logits -> params
        # mu, offset are each (C_dom,)
        mu, offset = self._logits_to_indel_rates(mu_offset_logits = self.tkf_mu_offset_logits,
                                         mu_min_val = self.mu_min_val,
                                         mu_max_val = self.mu_max_val,
                                         offs_min_val = self.offs_min_val,
                                         offs_max_val = self.offs_max_val)
        
        lam = mu * (1-offset) #(C_dom,)
        indel_params = {'mu': mu, #(C_dom,)
                        'lam': lam, #(C_dom,)
                        'offset': offset} #(C_dom,)
        
        
        ### get alpha, beta, gamma
        # contents of tkf_param_dict ( all ArrayLike[float32], (T,C_dom) ):
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
        tkf_param_dict['log_offset'] = jnp.log(offset) #(C_dom,)
        tkf_param_dict['log_one_minus_offset'] = jnp.log1p(-offset) #(C_dom,)
        
        
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
        
        
        ### get joint matrix
        joint_matrix =  self.fill_joint_tkf91(tkf_param_dict) #(T, C_dom, S, S)
        log_corr = 0
        
        if not return_all_matrices:
            # contents of final matrix_dict:
            # matrix_dict['joint'] (T, C_dom, S, S)
            matrix_dict = {'joint': joint_matrix}
        
        elif return_all_matrices:
            # contents of final matrix_dict:
            # matrix_dict['joint'] (T, C_dom, S, S)
            # matrix_dict['conditional'] (T, C_dom, S, S)
            # matrix_dict['marginal'] (C_dom, 2, 2)
            # matrix_dict['log_corr'] float
            matrix_dict = self.return_all_matrices(offset=offset,
                                                   joint_matrix=joint_matrix)
            matrix_dict['log_corr'] = 0
        
        # add tkf92 indel parameters
        # matrix_dict['lam'] (C_dom,)
        # matrix_dict['mu'] (C_dom,)
        # matrix_dict['offset'] (C_dom,)
        matrix_dict = {**matrix_dict, **indel_params}
        
        return (self.log_frag_class_probs, matrix_dict, tkf_param_dict)
        
    
    def fill_joint_tkf91(self, tkf_param_dict):
        """
        C_dom: number of mixtures associated with nested TKF92 models (domain-level)
        C_frag: number of mixtures associated with TKF92 fragments (fragment-level)
            > for tkf91, there can NOT be mixtures over transitions (i.e. C_frag=1)
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
            (all in log space); all are (T,C_dom)
                  
        Returns
        -------
        out : ArrayLike, (T,C_dom,S,S)
            joint loglike of transitions
        
        """
        ### entries in the matrix
        # lam / mu = 1 - offset
        log_lam_div_mu = tkf_param_dict['log_one_minus_offset'][None,:] #(1,C_dom)
        log_one_minus_lam_div_mu = tkf_param_dict['log_offset'][None,:] #(1,C_dom)
        
        
        # a_f = (1-beta)*alpha*(lam/mu);     log(a_f) = log(1-beta) + log(alpha) + log_lam_div_mu
        # b_g = beta;                        log(b_g) = log(beta)
        # c_h = (1-beta)*(1-alpha)*(lam/mu); log(c_h) = log(1-beta) + log(1-alpha) + log_lam_div_mu
        log_a_f = (tkf_param_dict['log_one_minus_beta'] + 
                   tkf_param_dict['log_alpha'] + 
                   log_lam_div_mu) #(T, C_dom)
        log_b_g = tkf_param_dict['log_beta']
        log_c_h = (tkf_param_dict['log_one_minus_beta'] + 
                   tkf_param_dict['log_one_minus_alpha'] + 
                   log_lam_div_mu) #(T, C_dom)
        log_mis_e = tkf_param_dict['log_one_minus_beta'] + log_one_minus_lam_div_mu #(T, C_dom)

        # p = (1-gamma)*alpha*(lam/mu);     log(p) = log(1-gamma) + log(alpha) + log_lam_div_mu
        # q = gamma;                        log(q) = log(gamma)
        # r = (1-gamma)*(1-alpha)*(lam/mu); log(r) = log(1-gamma) + log(1-alpha) + log_lam_div_mu
        log_p = (tkf_param_dict['log_one_minus_gamma'] + 
                 tkf_param_dict['log_alpha'] +
                 log_lam_div_mu) #(T, C_dom)
        log_q = tkf_param_dict['log_gamma']
        log_r = (tkf_param_dict['log_one_minus_gamma'] + 
                 tkf_param_dict['log_one_minus_alpha'] +
                 log_lam_div_mu) #(T, C_dom)
        log_d_e = tkf_param_dict['log_one_minus_gamma'] + log_one_minus_lam_div_mu #(T, C_dom)
        
        #(T, C_dom, S, S)
        out = jnp.stack([ jnp.stack([log_a_f, log_b_g, log_c_h, log_mis_e], axis=-1),
                          jnp.stack([log_a_f, log_b_g, log_c_h, log_mis_e], axis=-1),
                          jnp.stack([  log_p,   log_q,   log_r,   log_d_e], axis=-1),
                          jnp.stack([log_a_f, log_b_g, log_c_h, log_mis_e], axis=-1)
                          ], axis=-2)
        return out
    
    
    def return_all_matrices(self,
                            offset,
                            joint_matrix):
        """
        C_dom: number of domain mixtures
        T : number of branch lengths; this could be: 
            > an array of times for all samples (T; marginalize over these later)
            > an array of time per sample (T=B)
            > a quantized array of times per sample (T = T', where T' <= T)
        S: number of transition states (4 here: M, I, D, start/end)
         
        
        Arguments
        ---------
        offset : (C_dom,)
            1 - (lam/mu)
        
        joint_matrix : ArrayLike, (T, C_dom, S, S)
        
        
        Returns
        -------
        (returned_dictionary)["joint"]: (T, C_dom, S, S)
        (returned_dictionary)["marginal"]: (C_dom, 2, 2)
        (returned_dictionary)["conditional"]: (T, C_dom, S, S)
        
        """
        marginal_matrix = get_tkf91_single_seq_marginal_transition_logprobs(offset=offset) # (C_dom, 2, 2)
        
        # output is same as joint: (T, C_dom, S, S)
        cond_matrix = get_cond_transition_logprobs(marg_matrix=marginal_matrix, 
                                             joint_matrix=joint_matrix)
        
        return {'joint': joint_matrix,
                'marginal': marginal_matrix,
                'conditional': cond_matrix}
    
    def _logits_to_indel_rates(self, 
                              mu_offset_logits,
                              mu_min_val,
                              mu_max_val,
                              offs_min_val,
                              offs_max_val):
        """
        Arguments
        ---------
        mu_offset_logits : ArrayLike, (C_dom,2)
            logits to transform with bound sigmoid activation
        
        mu_min_val : float
            minimum value for bound sigmoid, to get mu
        
        mu_max_val : float
            maximum value for bound sigmoid, to get mu
        
        offs_min_val : float
            minimum value for bound sigmoid, to get offset
        
        offs_max_val : float
            maximum value for bound sigmoid, to get offset
        
        Returns
        -------
        mu : ArrayLike, (C_dom,)
            delete rate
        
        offset : ArrayLike, (C_dom,)
            used to calculate lambda: lambda = mu * (1 - offset)
        
        """
        ### mu
        mu = bound_sigmoid(x = mu_offset_logits[:,0],
                           min_val = mu_min_val,
                           max_val = mu_max_val) #(C_dom,)
        
        
        ### offset
        offset = bound_sigmoid(x = mu_offset_logits[:,1],
                               min_val = offs_min_val,
                               max_val = offs_max_val) #(C_dom,)
        
        return (mu, offset)
    
    
class TKF91TransitionLogprobsFromFile(TKF91TransitionLogprobs):
    """
    like TKF91TransitionLogprobs, but load values from a file
    
    NOTE: lambda and mu are provided directly, no need for offset
    
    C_dom: number of mixtures associated with nested TKF92 models (domain-level)
    C_frag: number of mixtures associated with TKF92 fragments (fragment-level)
        > for tkf91, there can NOT be mixtures over transitions (i.e. C_frag=1)
    B = batch size; number of samples
    T = number of branch lengths; this could be: 
        > an array of times for all samples (T; marginalize over these later)
        > an array of time per sample (T=B)
        > a quantized array of times per sample (T = T', where T' <= T)
    S: number of transition states (4 here: M, I, D, start/end)
        
        
    Initialize with
    ----------------
    config : dict
    
        config["num_domain_mixtures"] : int
            number of domain mixtures (for nested TKF model)
            
        config["num_fragment_mixtures"] : int
            number of tkf92 fragments (none here)
            
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
    
    
    Inherited from TKF91TransitionLogprobs
    ---------------------------------------
    fill_joint_tkf91
        fills in joint TKF91 transition matrix
    
    _logits_to_indel_rates
        converts mu/offset logits to mu/offset values
    
    return_all_matrices
        return transition matrices used for joint, marginal, and conditional 
        log-likelihood transitions
    
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
        self.num_domain_mixtures = self.config['num_domain_mixtures']
        self.num_fragment_mixtures = self.config['num_fragment_mixtures']
        in_file = self.config['filenames']['tkf_params_file']
        tkf_function_name = self.config['tkf_function']
        
        # no domain or fragment mixtures possible
        assert self.num_fragment_mixtures == 1
        self.log_frag_class_probs = jnp.zeros( (self.num_domain_mixtures, 1) ) #(C_dom, 1)
        
        
        ### read file
        # lam and mu should be (C_dom, )
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
                 return_all_matrices: bool,
                 sow_flax_intermeds: bool):
        """
        C_dom: number of mixtures associated with nested TKF92 models (domain-level)
        C_frag: number of mixtures associated with TKF92 fragments (fragment-level)
            > for tkf91, there can NOT be mixtures over transitions (i.e. C_frag=1)
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
            if false, only return the joint (used for model training)
            if true, return joint, conditional, and marginal matrices
            
        sow_flax_intermeds : bool
            switch for tensorboard logging
          
        Returns
        -------
        log_fragment_class_probs : (0, 0)
            placeholder tuple
            
        matrix_dict : dict
            matrix_dict["joint"]: (T,S,S)
                score transitions in joint probability calculation
            
            (if return_all_matrices) matrix_dict["marginal"]: (2,2)
                score transitions in marginal probability calculation
            
            (if return_all_matrices) matrix_dict["conditional"]: (T,S,S)
                score transitions in conditional probability calculation
        
            (if return_all_matrices) matrix_dict["corr"]: 0
                placeholder value
        
        None
            placeholder value
        
        tkf_param_dict : dict
            alpha, beta, gamma (and all associated values)
        """
        lam = self.param_dict['lambda'] #(C_dom,)
        mu = self.param_dict['mu'] #(C_dom,)
        offset = 1 - (lam /mu) #(C_dom,)
        
        # get alpha, beta, gamma
        tkf_param_dict, _ = self.tkf_function(mu = mu, 
                                        offset = offset,
                                        t_array = t_array)
        tkf_param_dict['log_offset'] = jnp.log(offset)
        tkf_param_dict['log_one_minus_offset'] = jnp.log1p(-offset)
        
        joint_matrix = self.fill_joint_tkf91(tkf_param_dict) #(T, C_dom, S, S)
        log_corr = 0
        
        if not return_all_matrices:
            matrix_dict = {'joint': joint_matrix}
        
        elif return_all_matrices:
            # contents of final matrix_dict:
            # matrix_dict['joint'] (T, C_dom, S, S)
            # matrix_dict['conditional'] (T, C_dom, S, S)
            # matrix_dict['marginal'] (C_dom, 2, 2)
            # matrix_dict['log_corr'] float
            matrix_dict = self.return_all_matrices(offset=offset,
                                                   joint_matrix=joint_matrix)
            matrix_dict['log_corr'] = 0
            
        return (self.log_frag_class_probs, matrix_dict, tkf_param_dict)
        
    
###############################################################################
### TKF91: DOMAIN level   #####################################################
###############################################################################
class TKF91DomainTransitionLogprobs(TKF91TransitionLogprobs):
    """
    TKF91 model; used for transitions between top-level domains
    
    B = batch size; number of samples
    T = number of branch lengths; this could be: 
        > an array of times for all samples (T; marginalize over these later)
        > an array of time per sample (T=B)
        > a quantized array of times per sample (T = T', where T' <= T)
    S: number of transition states (4 here: M, I, D, start/end)
        
    Initialize with
    ----------------
    config : dict 
        config["num_domain_mixtures"] : int
            number of domain mixtures (for nested TKF model)
        
        config["tkf_function"] : {'regular_tkf','approx_tkf','switch_tkf'}
            which function to use to solve for tkf parameters

        config["mu_range"] : Tuple, (2,)
            range for bound sigmoid activation that determines lamdba
            DEFAULT: -1e-4, 2
        
        config["offset_range"] : Tuple, (2,)
            range for bound sigmoid activation that determines offset 
            (which determines mu)
            DEFAULT: -1e-4, 0.333
            
    name : str
        class name, for flax
    
    
    Methods here
    ------------
    setup
    __call__
    
    
    inherited from TKF91TransitionLogprobs
    -----------------------------------------
    fill_joint_tkf91
        fills in joint TKF91 transition matrix
    
    _logits_to_indel_rates
        converts mu/offset logits to mu/offset values
    
    return_all_matrices
        return transition matrices used for joint, marginal, and conditional 
        log-likelihood transitions
    
    """
    config: dict
    name: str
    
    def setup(self):
        """
        C_dom: number of mixtures associated with nested TKF92 models (domain-level)
        B = batch size; number of samples
        T = number of branch lengths; this could be: 
            > an array of times for all samples (T; marginalize over these later)
            > an array of time per sample (T=B)
            > a quantized array of times per sample (T = T', where T' <= T)
        S: number of transition states (4 here: M, I, D, start/end)
            
            
        Flax Module Parameters
        -----------------------
        tkf_mu_offset_logits: ArrayLike (1,2)
            first value is logit for mu, second is for offset
            
        domain_class_prob_logits: ArrayLike (C_dom,)
            logits for P(domain)
        
        """
        ### unpack config
        # required
        self.num_domain_mixtures = self.config['num_domain_mixtures']
        tkf_function_name = self.config['tkf_function']
        
        # optional
        self.mu_min_val, self.mu_max_val = self.config.get( 'mu_range', [1e-4, 2] )
        self.offs_min_val, self.offs_max_val = self.config.get( 'offset_range', [1e-4, 0.333] )
        
        
        ### init flax parameters 
        # initialize logits for mu, offset
        self.tkf_mu_offset_logits = self.param('tkf_mu_offset_logits',
                                               nn.initializers.normal(),
                                               (1,2) ) #(1, 2)
        
        # initializing probability of domain classes
        if self.num_domain_mixtures > 1:
            self.domain_class_prob_logits = self.param('domain_class_prob_logits',
                                                       nn.initializers.normal(),
                                                       (self.num_domain_mixtures,) ) #(C_dom,)
        
        elif self.num_domain_mixtures == 1:
            self.domain_class_prob_logits = jnp.ones( (1,) )
        
        
        ### decide tkf function
        if tkf_function_name == 'regular_tkf':
            self.tkf_function = regular_tkf
        elif tkf_function_name == 'approx_tkf':
            self.tkf_function = approx_tkf
        elif tkf_function_name == 'switch_tkf':
            self.tkf_function = switch_tkf
    
    def __call__(self,
                 t_array,
                 return_all_matrices: bool,
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
            if false, only return the joint (used for model training)
            if true, return joint, conditional, and marginal matrices
        
        sow_flax_intermeds : bool
            switch for tensorboard logging
          
        Returns
        -------
        matrix_dict : dict
            matrix_dict["joint"]: (T,S,S)
                score transitions in joint probability calculation
            
            matrix_dict["lam"]: float
                insert rates
                
            matrix_dict["mu"]: float
                delete rates
            
            matrix_dict["offset"]: float
                1 - (lam/mu)
            
            (if return_all_matrices) matrix_dict["marginal"]: (2,2)
                score transitions in marginal probability calculation
            
            (if return_all_matrices) matrix_dict["conditional"]: (T,S,S)
                score transitions in conditional probability calculation
        
            (if return_all_matrices) matrix_dict["corr"]: 0
                placeholder value
                
        tkf_param_dict : dict
            alpha, beta, gamma (and all associated values)
        """
        ### P(C_domain) self.domain_class_prob_logits
        log_domain_class_probs = nn.log_softmax( self.domain_class_prob_logits, axis = -1 ) #(C_dom,)
        domain_class_probs = jnp.exp(log_domain_class_probs) #(C_dom,)
        
        if self.num_domain_mixtures > 1:
            self.maybe_sow( sow_flax_intermeds = sow_flax_intermeds,
                            vals = domain_class_probs,
                            label = f'{self.name}/domain_class_probabilities',
                            include_min_max = True,
                            include_perc_zeros = False)
            
            
        ### TKF91 top-level model
        # logits -> params
        # mu, offset are each (1,)
        mu, offset = self._logits_to_indel_rates(mu_offset_logits = self.tkf_mu_offset_logits,
                                         mu_min_val = self.mu_min_val,
                                         mu_max_val = self.mu_max_val,
                                         offs_min_val = self.offs_min_val,
                                         offs_max_val = self.offs_max_val)
        
        lam = mu * (1-offset) #(1,)
        
        # only store float values, not the size 1 array
        indel_params = {'mu': mu[0], #float
                        'lam': lam[0], #float
                        'offset': offset[0]} #float
        
        
        ### get alpha, beta, gamma
        # contents of tkf_param_dict ( all ArrayLike[float32], (1, T) ):
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
        
        
        ### maybe sow outputs
        self.maybe_sow( sow_flax_intermeds = sow_flax_intermeds,
                        vals = jnp.exp(tkf_param_dict['log_alpha']),
                        label = f'{self.name}/top_level_tkf91_alpha',
                        include_min_max = True,
                        include_perc_zeros = False)
        
        self.maybe_sow( sow_flax_intermeds = sow_flax_intermeds,
                        vals = jnp.exp(tkf_param_dict['log_beta']),
                        label = f'{self.name}/top_level_tkf91_beta',
                        include_min_max = True,
                        include_perc_zeros = False)
        
        self.maybe_sow( sow_flax_intermeds = sow_flax_intermeds,
                        vals = jnp.exp(tkf_param_dict['log_gamma']),
                        label = f'{self.name}/top_level_tkf91_gamma',
                        include_min_max = True,
                        include_perc_zeros = False)
        
        self.maybe_sow( sow_flax_intermeds = sow_flax_intermeds,
                        vals = lam,
                        label = f'{self.name}/top_level_tkf91_lambda',
                        include_min_max = False,
                        include_perc_zeros = False)
        
        self.maybe_sow( sow_flax_intermeds = sow_flax_intermeds,
                        vals = mu,
                        label = f'{self.name}/top_level_tkf91_mu',
                        include_min_max = False,
                        include_perc_zeros = False)
        
        
        ### calculate transition matrix
        joint_matrix =  self.fill_joint_tkf91(tkf_param_dict) #(T, 1, S, S)
        
        if not return_all_matrices:
            # contents of final matrix_dict (remove unused dim):
            # matrix_dict['joint'] (T, S, S)
            matrix_dict = {'joint': joint_matrix[:,0,...]}
        
        elif return_all_matrices:
            # contents of final matrix_dict (remove unused dim):
            # matrix_dict['joint'] (T, S, S)
            # matrix_dict['conditional'] (T, S, S)
            # matrix_dict['marginal'] (2, 2)
            # matrix_dict['log_corr'] float
            matrix_dict = self.return_all_matrices(offset=offset,
                                                   joint_matrix=joint_matrix)
            matrix_dict['joint'] = matrix_dict['joint'][:,0,...]
            matrix_dict['conditional'] = matrix_dict['conditional'][:,0,...]
            matrix_dict['marginal'] = matrix_dict['marginal'][0,...]
            matrix_dict['log_corr'] = 0
        
        # add tkf92 indel parameters
        # matrix_dict['lam'], float
        # matrix_dict['mu'], float
        # matrix_dict['offset'], float
        matrix_dict = {**matrix_dict, **indel_params}
        
        return (log_domain_class_probs, matrix_dict, tkf_param_dict)


class TKF91DomainTransitionLogprobsFromFile(TKF91DomainTransitionLogprobs):
    """
    Domain-level TKF91 transitions, but load parameters from file
    
    B = batch size; number of samples
    T = number of branch lengths; this could be: 
        > an array of times for all samples (T; marginalize over these later)
        > an array of time per sample (T=B)
        > a quantized array of times per sample (T = T', where T' <= T)
    S: number of transition states (4 here: M, I, D, start/end)
        
    Initialize with
    ----------------
    config : dict 
        config["num_domain_mixtures"] : int
            number of domain mixtures (for nested TKF model)
        
        config["tkf_function"] : {'regular_tkf','approx_tkf','switch_tkf'}
            which function to use to solve for tkf parameters
            
        config["filenames"]["top_level_tkf_params_file"]
            contains values for lambda, mu
            
        config["filenames"]["domain_class_probs"]: str
              file of domain class probabilities to load
            
    name : str
        class name, for flax
    
    
    Methods here
    ------------
    setup
    __call__
    
    
    inherited from TKF91DomainTransitionLogprobs
    ---------------------------------------------
    fill_joint_tkf91
        fills in joint TKF91 transition matrix
    
    return_all_matrices
        return transition matrices used for joint, marginal, and conditional 
        log-likelihood transitions
    
    """
    config: dict
    name: str
    
    def setup(self):
        """
        C_dom: number of mixtures associated with nested TKF92 models (domain-level)
        B = batch size; number of samples
        T = number of branch lengths; this could be: 
            > an array of times for all samples (T; marginalize over these later)
            > an array of time per sample (T=B)
            > a quantized array of times per sample (T = T', where T' <= T)
        S: number of transition states (4 here: M, I, D, start/end)
            
            
        Flax Module Parameters
        -----------------------
        None
        """
        ### unpack config
        # required
        self.num_domain_mixtures = self.config['num_domain_mixtures']
        tkf_function_name = self.config['tkf_function']
        tkf_params_file = self.config['filenames']['top_level_tkf_params_file']
        
        if self.num_domain_mixtures > 1:
            domain_class_probs_file = self.config['filenames']['domain_class_probs']
        
        
        ### read files
        # tkf parameters
        with open(tkf_params_file,'rb') as f:
            param_dict = pickle.load(f)
        
        param_dict = {k: jnp.array(v) for k,v in param_dict.items()}
        
        err = f'KEYS SEEN: {param_dict.keys()}'
        assert 'lambda' in param_dict.keys(), err
        assert 'mu' in param_dict.keys(), err
        assert 'r_extend' in param_dict.keys(), err
        
        param_dict = _expand_arr_in_dict(param_dict, 'lambda', 1) #(C_dom,)
        param_dict = _expand_arr_in_dict(param_dict, 'mu', 1) #(C_dom,)
        param_dict = _expand_arr_in_dict(param_dict, 'r_extend', 2) #(C_dom, C_frag)
        self.param_dict = param_dict
        
        # mixture probability of domain classes
        if self.num_domain_mixtures > 1:
            with open(domain_class_probs_file,'rb') as f:
                domain_class_probs = jnp.load(f) #(C_dom,) 
                
            self.log_domain_class_probs = safe_log(frag_class_probs) #(C_dom,)
        
        elif self.num_domain_mixtures == 1:
            self.log_domain_class_probs = jnp.zeros( (1,) ) #(C_dom,)
        
        assert self.log_domain_class_probs.shape[0] == self.num_domain_mixtures
        
        
        ### decide tkf function
        if tkf_function_name == 'regular_tkf':
            self.tkf_function = regular_tkf
        elif tkf_function_name == 'approx_tkf':
            self.tkf_function = approx_tkf
        elif tkf_function_name == 'switch_tkf':
            self.tkf_function = switch_tkf
    
    def __call__(self,
                 t_array,
                 return_all_matrices: bool,
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
            if false, only return the joint (used for model training)
            if true, return joint, conditional, and marginal matrices
        
        sow_flax_intermeds : bool
            switch for tensorboard logging
          
        Returns
        -------
        log_domain_class_probs: (C_dom,)
            
        matrix_dict : dict
            matrix_dict["joint"]: (T,S,S)
                score transitions in joint probability calculation
            
            matrix_dict["lam"]: float
                insert rates
                
            matrix_dict["mu"]: float
                delete rates
            
            matrix_dict["offset"]: float
                1 - (lam/mu)
            
            (if return_all_matrices) matrix_dict["marginal"]: (2,2)
                score transitions in marginal probability calculation
            
            (if return_all_matrices) matrix_dict["conditional"]: (T,S,S)
                score transitions in conditional probability calculation
        
            (if return_all_matrices) matrix_dict["corr"]: 0
                placeholder value
                
        tkf_param_dict : dict
            alpha, beta, gamma (and all associated values)
        """
        ### P(C_domain) self.domain_class_prob_logits
        log_domain_class_probs = self.log_domain_class_probs #(C_dom,)
        domain_class_probs = jnp.exp( log_domain_class_probs ) #(C_dom,)
        
        
        ### TKF91 top-level model
        lam = self.param_dict['lambda'] #(C_dom,)
        mu = self.param_dict['mu'] #(C_dom,)
        offset = 1 - (lam /mu) #(C_dom,)
        r_extend = self.param_dict['r_extend'] #(C_dom, C_frag)
        
        # only store float values, not the size 1 array
        indel_params = {'mu': mu[0], #float
                        'lam': lam[0], #float
                        'offset': offset[0]} #float
        
        
        ### get alpha, beta, gamma
        # contents of tkf_param_dict ( all ArrayLike[float32], (1, T) ):
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
        
        # joint transition matrix
        joint_matrix =  self.fill_joint_tkf91(tkf_param_dict) #(T, 1, S, S)
        
        if not return_all_matrices:
            # contents of final matrix_dict (remove unused dim):
            # matrix_dict['joint'] (T, S, S)
            matrix_dict = {'joint': joint_matrix[:,0,...]}
        
        elif return_all_matrices:
            # contents of final matrix_dict (remove unused dim):
            # matrix_dict['joint'] (T, S, S)
            # matrix_dict['conditional'] (T, S, S)
            # matrix_dict['marginal'] (2, 2)
            # matrix_dict['log_corr'] float
            matrix_dict = self.return_all_matrices(offset=offset,
                                                   joint_matrix=joint_matrix)
            matrix_dict['joint'] = matrix_dict['joint'][:,0,...]
            matrix_dict['conditional'] = matrix_dict['conditional'][:,0,...]
            matrix_dict['marginal'] = matrix_dict['marginal'][0,...]
            matrix_dict['log_corr'] = 0
        
        # add tkf92 indel parameters
        # matrix_dict['lam'], float
        # matrix_dict['mu'], float
        # matrix_dict['offset'], float
        matrix_dict = {**matrix_dict, **indel_params}
        
        return (log_domain_class_probs, matrix_dict, tkf_param_dict)
    
    
###############################################################################
### TKF92   ###################################################################
###############################################################################
class TKF92TransitionLogprobs(TKF91TransitionLogprobs):
    """
    TKF92 model; used for calculating transitions in model of
        P(anc, desc, align)
    
    C_dom: number of mixtures associated with nested TKF92 models (domain-level)
    C_frag: number of mixtures associated with TKF92 fragments (fragment-level)
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
        
        config["num_domain_mixtures"] : int
            number of domain mixtures (for nested TKF model)
            
        config["num_fragment_mixtures"] : int
            number of tkf92 fragments

        config["mu_range"] : Tuple, (2,)
            range for bound sigmoid activation that determines mu
            DEFAULT: -1e-4, 2
        
        config["offset_range"] : Tuple, (2,)
            range for bound sigmoid activation that determines offset 
            (which determines mu)
            DEFAULT: -1e-4, 0.333
            
        config["r_range"]
            range for bound sigmoid activation that determines TKF r
            DEFAULT: -1e-4, 0.999
            
    name : str
        class name, for flax
    
    
    Methods here
    ------------
    setup
    
    __call__
    
    fill_joint_tkf92
        fills in joint TKF92 transition matrix
        
    return_all_matrices
        return transition matrices used for joint, marginal, and conditional 
        log-likelihood transitions
        
    Inherited from TKF91TransitionLogprobs
    ---------------------------------------
    fill_joint_tkf91
        fills in joint TKF91 transition matrix
    
    _logits_to_indel_rates
        converts mu/offset logits to mu/offset values
    
    """
    config: dict
    name: str
    
    def setup(self):
        """
        C_dom: number of mixtures associated with nested TKF92 models (domain-level)
        C_frag: number of mixtures associated with TKF92 fragments (fragment-level)
        B = batch size; number of samples
        T = number of branch lengths; this could be: 
            > an array of times for all samples (T; marginalize over these later)
            > an array of time per sample (T=B)
            > a quantized array of times per sample (T = T', where T' <= T)
        S: number of transition states (4 here: M, I, D, start/end)
        
        
        Flax Module Parameters
        -----------------------
        tkf_mu_offset_logits : ArrayLike (C_dom, 2)
            first value is logit for mu, second is for offset
        
        r_extend_logits : ArrayLike (C_frag,)
            logits for TKF fragment extension probability, r
            this is EXCLUSIVELY for the fragment-level tkf92 indel process
        
        frag_class_prob_logits : ArrayLike (C_dom, C_frag)
        
        """
        ### unpack config
        # required
        self.num_domain_mixtures = self.config['num_domain_mixtures']
        self.num_fragment_mixtures = self.config['num_fragment_mixtures']
        tkf_function_name = self.config['tkf_function']
        
        # optional inputs
        self.mu_min_val, self.mu_max_val = self.config.get( 'mu_range', [1e-4, 2] )
        self.offs_min_val, self.offs_max_val = self.config.get( 'offset_range', [1e-4, 0.333] )
        self.r_extend_min_val, self.r_extend_max_val = self.config.get( 'r_range', [1e-4, 0.999] )
        
        
        ### init flax parameters
        # initialize logits for mu, offset
        self.tkf_mu_offset_logits = self.param('tkf_mu_offset_logits',
                                               nn.initializers.normal(),
                                               (self.num_domain_mixtures, 2) ) #(C_dom, 2)
        
        # initializing r extension prob
        self.r_extend_logits = self.param('r_extend_logits',
                                          nn.initializers.normal(),
                                          (self.num_domain_mixtures, 
                                           self.num_fragment_mixtures) ) # (C_dom, C_frag)
        
        # initializing probability of fragment classes
        if self.num_fragment_mixtures > 1:
            self.frag_class_prob_logits = self.param('frag_class_prob_logits',
                                              nn.initializers.normal(),
                                              (self.num_domain_mixtures, 
                                               self.num_fragment_mixtures) ) #(C_dom, C_frag)
        elif self.num_fragment_mixtures == 1:
            self.frag_class_prob_logits = jnp.ones( (self.num_domain_mixtures, 1) ) #(C_dom, C_frag)
        
        
        ### decide tkf function
        if tkf_function_name == 'regular_tkf':
            self.tkf_function = regular_tkf
        elif tkf_function_name == 'approx_tkf':
            self.tkf_function = approx_tkf
        elif tkf_function_name == 'switch_tkf':
            self.tkf_function = switch_tkf
        
    
    def __call__(self,
                 t_array,
                 return_all_matrices: bool,
                 sow_flax_intermeds: bool):
        """
        C_dom: number of mixtures associated with nested TKF92 models (domain-level)
        C_frag: number of mixtures associated with TKF92 fragments (fragment-level)
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
        
        return_all_matrices : bool
            if false, only return the joint (used for model training)
            if true, return joint, conditional, and marginal matrices
        
        sow_flax_intermeds : bool
            switch for tensorboard logging
          
        Returns
        -------
        log_frag_class_probs : ArrayLike, (C_dom, C_frag) 
            P(c_fragment | c_domain)
        
        matrix_dict : dict
            matrix_dict["joint"]: (T,C_dom,C_frag,C_frag,S,S)
                score transitions in joint probability calculation
                
            matrix_dict["lam"]: (C_dom,)
                insert rates
                
            matrix_dict["mu"]: (C_dom,)
                delete rates
            
            matrix_dict["offset"]: (C_dom,)
                1 - (lam/mu)
                
            matrix_dict["marginal"]: (C_dom,C_frag,C_frag,2,2)
                score transitions in marginal probability calculation
            
            matrix_dict["conditional"]: (T,C_dom,C_frag,C_frag,S,S)
                score transitions in conditional probability calculation
            
            matrix_dict["log_corr"]: (C_dom, C_frag)
                correction factor in case alignment starts with start -> ins
        
        tkf_param_dict : dict
            alpha, beta, gamma (and all associated values)
        """
        C_dom = self.num_domain_mixtures
        C_frag = self.num_fragment_mixtures
        
        
        ### P(C_fragment | C_domain)
        log_frag_class_probs = nn.log_softmax( self.frag_class_prob_logits, axis = -1 ) #(C_dom, C_fr)
        frag_class_probs = jnp.exp(log_frag_class_probs) #(C_dom, C_fr)
        
        if (sow_flax_intermeds) and (C_frag*C_dom > 1):
            self.maybe_sow( sow_flax_intermeds = sow_flax_intermeds,
                            vals = frag_class_probs,
                            label = f'{self.name}/fragment class probabilities',
                            include_min_max = True,
                            include_perc_zeros = False)
            
        ### TKF92 model
        out = self._logits_to_indel_rates(mu_offset_logits = self.tkf_mu_offset_logits,
                                         mu_min_val = self.mu_min_val,
                                         mu_max_val = self.mu_max_val,
                                         offs_min_val = self.offs_min_val,
                                         offs_max_val = self.offs_max_val) 
        
        # mu, offset are each (C_dom,)
        mu, offset = out 
        lam = mu * (1-offset) # (C_dom,)
        del out
        
        # r_extend
        r_extend = bound_sigmoid(x = self.r_extend_logits,
                                 min_val = self.r_extend_min_val,
                                 max_val = self.r_extend_max_val) # (C_dom, C_frag)
        
        indel_params = {'mu': mu, #(C_dom,)
                        'lam': lam, #(C_dom,)
                        'offset': offset, #(C_dom,)
                        'r_extend': r_extend} #(C_dom,C_frag)
        
        
        ### get alpha, beta, gamma
        # contents of tkf_param_dict ( all ArrayLike[float32], (T,C_dom) ):
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
        tkf_param_dict['log_offset'] = jnp.log(offset) #(C_dom,)
        tkf_param_dict['log_one_minus_offset'] = jnp.log1p(-offset) #(C_dom,)
        
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
        
        
        ### joint prob matrix
        # (T, C_dom, C_{frag_from}, C_{frag_to}, S_from=4, S_to=4)
        joint_matrix =  self.fill_joint_tkf92(tkf_param_dict=tkf_param_dict, 
                                              r_extend=r_extend,
                                              frag_class_probs=frag_class_probs)
        
        if not return_all_matrices:
            # contents of final matrix_dict:
            # matrix_dict['joint'] (T, C_dom, C_frag, C_frag, S, S)
            matrix_dict = {'joint': joint_matrix}
        
        elif return_all_matrices:
            # contents of final matrix_dict:
            # matrix_dict['joint'] (T, C_dom, C_frag, C_frag, S, S)
            # matrix_dict['conditional'] (T, C_dom, C_frag, C_frag, S, S)
            # matrix_dict['marginal'] (C_dom, 2, 2)
            # matrix_dict['log_corr'] float
            matrix_dict = self.return_all_matrices(offset=offset,
                                                   frag_class_probs=frag_class_probs,
                                                   r_extend = r_extend,
                                                   joint_matrix=joint_matrix)
            
            # correction factors for S->I transition
            matrix_dict['log_corr'] = jnp.log(lam[:,None]/mu[:,None]) - jnp.log( r_extend + (1-r_extend)*(lam[:,None]/mu[:,None]) ) #(C_dom, C_fr)
        
        # add tkf92 indel parameters
        # matrix_dict['lam'] (C_dom,)
        # matrix_dict['mu'] (C_dom,)
        # matrix_dict['offset'] (C_dom,)
        matrix_dict = {**matrix_dict, **indel_params}
        
        return (log_frag_class_probs, matrix_dict, tkf_param_dict)
        
    
    def fill_joint_tkf92(self,
                        tkf_param_dict,
                        r_extend,
                        frag_class_probs):
        """
        C_dom: number of mixtures associated with nested TKF92 models (domain-level)
        C_frag: number of mixtures associated with TKF92 fragments (fragment-level)
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
            (all in log space); all are (T,C_dom)
        
        r_extend : ArrayLike, (C_dom, C_frag)
            fragment extension probabilities
        
        frag_class_probs : ArrayLike, (C_dom, C_frag)
            support for the classes i.e. P(end at class c_frag)
          
        Returns
        -------
        out : ArrayLike, (T, C_dom, C_{frag_from}, C_{frag_to}, S_from=4, S_to=4)
            joint loglike of transitions
        
        """
        ### need joint TKF91 for this (which already contains lam/mu terms)
        log_U = self.fill_joint_tkf91(tkf_param_dict) #(T, C_dom, S_from, S_to)
        
        # dims
        T = tkf_param_dict['log_alpha'].shape[0]
        C_dom = self.num_domain_mixtures #domain-level classes
        C_frag = self.num_fragment_mixtures #fragment-level classes
        S = log_U.shape[-1] #number of hidden states (like M, I, D, and start/end)
        
        # converted log values; expand
        log_r_extend = safe_log( r_extend ) #(C_dom, C_{frag_from})
        log_one_minus_r = log_one_minus_x(log_r_extend) #(C_dom, C_{frag_from})
        log_one_minus_r = log_one_minus_r[None, ..., None, None] #(1, C_dom, C_{frag_from}, 1, 1)
        
        ### entries in the matrix
        # (1-r_c) U(i,j) for all (MID -> MIDE transitions), 
        #   U(i,j) for all start -> MIDE transitions
        log_U_exp = log_U[:, :, None, :, :]  # shape: (T, C_dom, 1, S, S)

        # Build a mask of shape (S_from,) where the last index is False
        s_mask = jnp.arange(S) != (S - 1)  # shape: (S_from,)
        s_mask_exp = s_mask[None, None, None, :, None]  # shape: (1, 1, 1, S_from, 1)
        
        # Apply log_one_minus_r only where S_from != S-1
        #                        log_U_exp: (T,     1,             1, S_from, S_to)
        # log_one_minus_r[..., None, None]: (1, C_dom, C_{frag_from},      1,    1)
        #            log_tkf92_rate_mat is: (T, C_dom, C_{frag_from}, S_from, S_to)
        log_tkf92_rate_mat = log_U_exp + jnp.where( s_mask_exp,
                                                    log_one_minus_r, 
                                                    0.0 ) 
        del s_mask_exp
        
        # expand again
        s_mask_exp = s_mask[None, None, None, None, None, :]  # shape: (1, 1, 1, 1, 1, S_to)
        log_tkf92_rate_mat = log_tkf92_rate_mat[:,:,:, None, ...] #(T, C_dom, C_{frag_from}, 1, S_from, S_to)
        log_frag_class_probs = safe_log(frag_class_probs) #(C_dom, C_{frag_to})
        log_frag_class_probs = log_frag_class_probs[None, :, None, :, None, None] #(1, C_dom, 1, C_{frag_to}, 1, 1)
        
        # multiply by P(c) across all C_to (not including transitions that 
        #   end with <end>
        # log_tkf92_rate_mat before: (T, C_dom, C_{frag_from},           1, S_from, S_to)
        #      log_frag_class_probs: (1, C_dom,             1, C_{frag_to},      1,    1)
        #  log_tkf92_rate_mat AFTER: (T, C_dom, C_{frag_from}, C_{frag_to}, S_from, S_to)        
        log_tkf92_rate_mat = log_tkf92_rate_mat + jnp.where( s_mask_exp,
                                                             log_frag_class_probs, 
                                                             0.0 ) 
        del s_mask_exp
        
        # at MID: where frag_class_from == frag_class_to and state_from == state_to, 
        #   add factor of r; S_from=3 means start, and S_to=3 means end
        i_idx, j_idx = jnp.meshgrid(jnp.arange(C_frag), jnp.arange(S-1), indexing="ij")
        i_idx = i_idx.flatten()
        j_idx = j_idx.flatten()

        # add r to specific locations
        # log_tkf92_rate_mat[:, :, i_idx, i_idx, j_idx, j_idx] (T, C_dom, S-1)
        prev_vals = log_tkf92_rate_mat[:, :, i_idx, i_idx, j_idx, j_idx].reshape( (T, C_dom, C_frag, S-1) ) #(T, C_dom, C_frag, S-1)
        # r_to_add = jnp.broadcast_to( log_r_extend[None,...,None], prev_vals.shape) #(T, C_dom, C_frag, S-1)
        # new_vals = logsumexp_with_arr_lst([r_to_add, prev_vals]).reshape(T, C_dom, -1) #(T, C_dom, C_frag * S-1)
        new_vals = jnp.logaddexp( log_r_extend[None,...,None], prev_vals ).reshape(T, C_dom, -1) #(T, C_dom, C_frag * S-1)
        del prev_vals #, r_to_add

        # scatter back with advanced indexing
        #  log_tkf92_rate_mat is: (T, C_dom, C_{frag_from}, C_{frag_to}, S_from, S_to)   
        log_tkf92_rate_mat = log_tkf92_rate_mat.at[:, :, i_idx, i_idx, j_idx, j_idx].set(new_vals) 
        
        return log_tkf92_rate_mat
    
    
    def return_all_matrices(self,
                            offset,
                            r_extend,
                            frag_class_probs,
                            joint_matrix):
        """
        C_dom: number of mixtures associated with nested TKF92 models (domain-level)
        C_frag: number of mixtures associated with TKF92 fragments (fragment-level)
        B = batch size; number of samples
        T = number of branch lengths; this could be: 
            > an array of times for all samples (T; marginalize over these later)
            > an array of time per sample (T=B)
            > a quantized array of times per sample (T = T', where T' <= T)
        S: number of transition states (4 here: M, I, D, start/end)
         
        
        Arguments
        ---------
        offset : ArrayLike, (C_dom,)
            1 - (lam/mu)
        
        r_extend : ArrayLike, (C_dom, C_frag)
            fragment extension probabilities
        
        frag_class_probs : ArrayLike, (C_dom, C_frag)
            support for the classes i.e. P(end at class c_frag)
         
        joint_matrix : ArrayLike, (T, C_dom, C_{frag_from}, C_{frag_to}, S_from=4, S_to=4)
        
        
        Returns
        -------
        (returned_dictionary)["joint"]: (T, C_dom, C_{frag_from}, C_{frag_to}, S_from=4, S_to=4)
        (returned_dictionary)["marginal"]: (C_dom, C_{frag_from}, C_{frag_to}, 2, 2)
        (returned_dictionary)["conditional"]: (T, C_dom, C_{frag_from}, C_{frag_to}, S_from=4, S_to=4)
        
        """
        # output is: (C_dom, C_{frag_from}, C_{frag_to}, 2, 2)
        marginal_matrix = get_tkf92_single_seq_marginal_transition_logprobs(offset=offset,
                                                      frag_class_probs=frag_class_probs,
                                                      r_extend=r_extend)
        
        # output is: (T, C_dom, C_{frag_from}, C_{frag_to}, S_from=4, S_to=4)
        cond_matrix = get_cond_transition_logprobs(marg_matrix=marginal_matrix, 
                                             joint_matrix=joint_matrix)
        
        return {'joint': joint_matrix,
                'marginal': marginal_matrix,
                'conditional': cond_matrix}
    

class TKF92TransitionLogprobsFromFile(TKF92TransitionLogprobs):
    """
    like TKF91TransitionLogprobs, but load values from a file
    
    NOTE: mu is provided directly, no need for offset
    
    C_dom: number of mixtures associated with nested TKF92 models (domain-level)
    C_frag: number of mixtures associated with TKF92 fragments (fragment-level)
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
    
        config["filenames"]["tkf_params_file"] : str
            contains values for lambda, mu, r-extension
            
        config["filenames"]["frag_class_probs"]: str
              file of fragment class probabilities to load
                    
    name : str
        class name, for flax
    
    
    Methods here
    ------------
    setup
    __call__
    
    
    Inherited from TKF91TransitionLogprobs
    ---------------------------------------
    fill_joint_tkf91
        fills in joint TKF91 transition matrix
        
    _logits_to_indel_rates
        converts mu/offset logits to mu/offset values
    
    
    Inherited from TKF92TransitionLogprobs
    ---------------------------------------
    fill_joint_tkf92
        fills in joint TKF92 transition matrix
        
    return_all_matrices
        return transition matrices used for joint, marginal, and conditional 
        log-likelihood transitions
    
    """
    config: dict
    name: str
    
    def setup(self):
        """
        
        Flax Module Parameters
        -----------------------
        None
        
        """
        ### unpack config
        self.num_domain_mixtures = self.config['num_domain_mixtures']
        self.num_fragment_mixtures = self.config['num_fragment_mixtures']
        tkf_params_file = self.config['filenames']['tkf_params_file']
        tkf_function_name = self.config['tkf_function']
        
        if (self.num_domain_mixtures * self.num_fragment_mixtures) > 1:
            frag_class_probs_file = self.config['filenames']['frag_class_probs']
        
        
        ### read files
        # tkf parameters
        with open(tkf_params_file,'rb') as f:
            param_dict = pickle.load(f)
        
        param_dict = {k: jnp.array(v) for k,v in param_dict.items()}
        
        err = f'KEYS SEEN: {param_dict.keys()}'
        assert 'lambda' in param_dict.keys(), err
        assert 'mu' in param_dict.keys(), err
        assert 'r_extend' in param_dict.keys(), err
        
        param_dict = _expand_arr_in_dict(param_dict, 'lambda', 1) #(C_dom,)
        param_dict = _expand_arr_in_dict(param_dict, 'mu', 1) #(C_dom,)
        param_dict = _expand_arr_in_dict(param_dict, 'r_extend', 2) #(C_dom, C_frag)
        
        self.param_dict = param_dict
        
        # mixture probability of fragment classes
        if (self.num_domain_mixtures * self.num_fragment_mixtures) > 1:
            with open(frag_class_probs_file,'rb') as f:
                frag_class_probs = jnp.load(f) #(C_dom, C_frag) or (C_frag,)
            
            if len(frag_class_probs.shape)==1:
                frag_class_probs = frag_class_probs[None, :] #(C_dom=1, C_frag)
            
            self.log_frag_class_probs = safe_log(frag_class_probs) #(C_dom, C_frag)
        
        elif (self.num_domain_mixtures * self.num_fragment_mixtures) == 1:
            self.log_frag_class_probs = jnp.zeros( (1,1) )
        
        assert self.log_frag_class_probs.shape[0] == self.num_domain_mixtures
        assert self.log_frag_class_probs.shape[1] == self.num_fragment_mixtures
        
        
        ### pick tkf function
        if tkf_function_name == 'regular_tkf':
            self.tkf_function = regular_tkf
        elif tkf_function_name == 'approx_tkf':
            self.tkf_function = approx_tkf
        elif tkf_function_name == 'switch_tkf':
            self.tkf_function = switch_tkf
                    
        
    def __call__(self,
                 t_array,
                 return_all_matrices: bool,
                 sow_flax_intermeds: bool):
        """
        C_dom: number of mixtures associated with nested TKF92 models (domain-level)
        C_frag: number of mixtures associated with TKF92 fragments (fragment-level)
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
        
        return_all_matrices : bool
            if false, only return the joint (used for model training)
            if true, return joint, conditional, and marginal matrices
        
        sow_flax_intermeds : bool
            switch for tensorboard logging
          
        Returns
        -------
        log_frag_class_probs : ArrayLike, (C_dom, C_frag) 
            P(c_fragment | c_domain)
        
        matrix_dict : dict
            matrix_dict["joint"]: (T,C_dom,C_frag,C_frag,S,S)
                score transitions in joint probability calculation
                
            matrix_dict["marginal"]: (C_dom,C_frag,C_frag,2,2)
                score transitions in marginal probability calculation
            
            matrix_dict["conditional"]: (T,C_dom,C_frag,C_frag,S,S)
                score transitions in conditional probability calculation
            
            matrix_dict["log_corr"]: (C_dom, C_frag)
                correction factor in case alignment starts with start -> ins
        
        tkf_param_dict : dict
            alpha, beta, gamma (and all associated values)
        """
        log_frag_class_probs = self.log_frag_class_probs #(C_dom, C_frag)
        frag_class_probs = jnp.exp(log_frag_class_probs) #(C_dom, C_frag)
        
        lam = self.param_dict['lambda'] #(C_dom,)
        mu = self.param_dict['mu'] #(C_dom,)
        offset = 1 - (lam /mu) #(C_dom,)
        r_extend = self.param_dict['r_extend'] #(C_dom, C_frag)
        
        indel_params = {'mu': mu, #(C_dom,)
                        'lam': lam, #(C_dom,)
                        'offset': offset, #(C_dom,)
                        'r_extend': r_extend} #(C_dom,C_frag)
        
        
        ### get alpha, beta, gamma
        # contents of tkf_param_dict ( all ArrayLike[float32], (T,C_dom) ):
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
        tkf_param_dict['log_offset'] = jnp.log(offset) #(C_dom,)
        tkf_param_dict['log_one_minus_offset'] = jnp.log1p(-offset) #(C_dom,)
        
        # (T, C_dom, C_{frag_from}, C_{frag_to}, S_from=4, S_to=4)
        joint_matrix =  self.fill_joint_tkf92(tkf_param_dict = tkf_param_dict, 
                                              r_extend = r_extend,
                                              frag_class_probs = frag_class_probs)
        
        if not return_all_matrices:
            # contents of final matrix_dict:
            # matrix_dict['joint'] (T, C_dom, C_frag, C_frag, S, S)
            matrix_dict = {'joint': joint_matrix}
        
        elif return_all_matrices:
            # contents of final matrix_dict:
            # matrix_dict['joint'] (T, C_dom, C_frag, C_frag, S, S)
            # matrix_dict['conditional'] (T, C_dom, C_frag, C_frag, S, S)
            # matrix_dict['marginal'] (C_dom, 2, 2)
            # matrix_dict['log_corr'] float
            matrix_dict = self.return_all_matrices(offset=offset,
                                                   frag_class_probs=frag_class_probs,
                                                   r_extend = r_extend,
                                                   joint_matrix=joint_matrix)
            
            # correction factors for S->I transition
            matrix_dict['log_corr'] = jnp.log(lam[:,None]/mu[:,None]) - jnp.log( r_extend + (1-r_extend)*(lam[:,None]/mu[:,None]) ) #(C_dom, C_fr)
        
        # add tkf92 indel parameters
        # matrix_dict['lam'] (C_dom,)
        # matrix_dict['mu'] (C_dom,)
        # matrix_dict['offset'] (C_dom,)
        # matrix_dict['r_extend'] (1,1)
        matrix_dict = {**matrix_dict, **indel_params}
        
        return (log_frag_class_probs, matrix_dict, tkf_param_dict)


###############################################################################
### Geometric sequence length (no indels)   ###################################
###############################################################################
class GeomLenTransitionLogprobs(ModuleBase):
    """
    Simply assume sequence has geometrically-distributed sequence length
    
    Used mostly for debugging; doesn't have mixtures
    
    
    Initialize with
    ----------------
    config : dict (but nothing used here)
            
    name : str
        class name, for flax
    
    Methods here
    ------------
    setup
    __call__
    
    Methods inherited from ModuleBase
    ----------------------------------
    sow_flax_intermeds
        for tensorboard logging
    
    """
    config: dict
    name: str
    
    def setup(self):
        """
        
        Flax Module Parameters
        -----------------------
        p_emit_logit: ArrayLike (1,)
            P(emit at alignment column) = 1 - P(end alignment)
        
        """
        # under standard sigmoid activation, initial value is: 0.95257413
        init_logit = jnp.array([3.0])
        self.p_emit_logit = self.param('p_emit_logit',
                                       lambda rng, shape, dtype: init_logit,
                                       init_logit.shape )
        
        # no domain or fragment mixtures possible
        self.log_frag_class_probs = jnp.zeros( (1,1) ) #(C_dom, C_frag)
        
    def __call__(self,
                 return_all_matrices: bool,
                 sow_flax_intermeds: bool,                 
                 *args,
                 **kwargs):
        """
        
        Arguments
        ----------
        return_all_matrices : bool
            if false, only return the joint (used for model training)
            if true, return joint, conditional, and marginal matrices
        
        sow_flax_intermeds : bool
            switch for tensorboard logging
          
        Returns
        -------
        log_fragment_class_probs : (0, 0)
            placeholder values
            
        matrix_dict : dict
            matrix_dict["joint"]: (2,)
                score transitions in joint probability calculation
                
            (if return_all_matrices) matrix_dict["marginal"]: (2,)
                score transitions in marginal probability calculation
            
            (if return_all_matrices) matrix_dict["conditional"]: (2,)
                score transitions in conditional probability calculation
            
            (if return_all_matrices) matrix_dict["log_corr"]: 0
                placeholder
        
        (output tuples) :  None
            placeholder values
        """
        p_emit = nn.sigmoid(self.p_emit_logit) #(1,)
        
        out_vec = safe_log( jnp.concatenate( [p_emit, 1-p_emit] ) ) #(2,)
        
        if not return_all_matrices:
            matrix_dict = {'joint': out_vec}
        
        elif return_all_matrices:
            matrix_dict = {'joint': out_vec,
                           'marginal': out_vec,
                           'conditional': out_vec,
                           'log_corr': 0}
            
        return (self.log_frag_class_probs, matrix_dict, None)
        

class GeomLenTransitionLogprobsFromFile(GeomLenTransitionLogprobs):
    """
    same as GeomLenTransitionLogprobs, but load parameter from file
    
    
    Initialize with
    ----------------
    config : dict 
        config["filenames"]["geom_length_params_file"] : str
            contains probability of emission; could either be a 1-element 
            numpy array or a flat text file
            
    name : str
        class name, for flax
    
    Methods here
    ------------
    __call__
    setup
    
    """
    config: dict
    name: str
    
    def setup(self):
        """
        
        Flax Module Parameters
        -----------------------
        None
        
        """
        file_with_transit_probs = self.config['filenames']['geom_length_params_file']
        
        if file_with_transit_probs.endswith('.npy'):
            with open(file_with_transit_probs,'rb') as f:
                vec = jnp.load(f) #(2,)
        p_emit = vec[0]
        one_minus_p_emit = vec[1]
        self.out_vec = safe_log( jnp.array( [p_emit, one_minus_p_emit] ) )
        
        # no domain or fragment mixtures possible
        self.log_frag_class_probs = jnp.zeros( (1,1) ) #(C_dom, C_frag)
        
    
    def __call__(self,
                 return_all_matrices: bool,
                 *args,
                 **kwargs):
        """
        
        Arguments
        ----------
        None
          
        Returns
        -------
        matrix_dict : dict
            matrix_dict["joint"]: (2,)
                score transitions in joint probability calculation
                
            matrix_dict["marginal"]: (2,)
                score transitions in marginal probability calculation
            
            matrix_dict["conditional"]: (2,)
                score transitions in conditional probability calculation
        
        (output tuples) : None
            placeholder values
        """
        if not return_all_matrices:
            matrix_dict = {'joint': self.out_vec}
        
        elif return_all_matrices:
            matrix_dict = {'joint': self.out_vec,
                           'marginal': self.out_vec,
                           'conditional': self.out_vec,
                           'log_corr': 0}
            
        return (self.log_frag_class_probs, matrix_dict, None)
    
   




