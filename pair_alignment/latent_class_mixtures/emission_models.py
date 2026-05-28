#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 02:03:13 2025



ABOUT:
======
Flax Modules needed for scoring emissions; may have their own parameters, and 
    could write to tensorboard


modules:
=========

originals:
------------
'EqulDistLogprobsFromCounts',
'EqulDistLogprobsPerClass',
'F81Logprobs',
'GTRLogprobs',
'IndpRateMultipliers',
'RateMultipliersPerClass',

loading from files:
--------------------
'EqulDistLogprobsFromFile',
'F81LogprobsFromFile',
'GTRLogprobsFromFile',
'LG08Logprobs',
'IndpRateMultipliersFromFile',
'RateMultipliersPerClassFromFile',
    

"""
from flax import linen as nn
import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm
from jax.scipy.special import logsumexp
from jax._src.typing import Array, ArrayLike

from functools import partial
from typing import Optional

from utils.BaseClasses import ModuleBase
from latent_class_mixtures.model_functions import (bound_sigmoid,
                                                    bound_sigmoid_inverse,
                                                    safe_log,
                                                    rate_matrix_from_exch_equl,
                                                    scale_rate_matrix,
                                                    upper_tri_vector_to_sym_matrix,
                                                    cond_logprob_emit_at_match_per_mixture,
                                                    joint_logprob_emit_at_match_per_mixture,
                                                    fill_f81_logprob_matrix)
                    
def _load_params(in_file, target_ndim: int):
    with open(in_file, 'rb') as f:
        mat = jnp.load(f)

    # Add leading singleton dims until desired ndim is reached
    while mat.ndim < target_ndim:
        mat = jnp.expand_dims(mat, axis=0)

    return mat


###############################################################################
### Rate multipliers   ########################################################
###############################################################################
class RateMultipliersPerClass(ModuleBase):
    """
    C_trans (C_frag + C_dom): number of mixtures associated with transitions (variable)
    C_sites: number of latent site classes
    K: number of rate multipliers
    
    
    Generate C_trans * C_sites * K rate multipliers, and 
      probabilty of rate multiplier k, given mixture classes: P(k|c_site, c_trans)
    
    
    Initialize with
    ----------------
    config : dict
        config['num_domain_mixtures'] : int
            number of larger TKF92 domain mixtures (>1 if nested TKF92)
    
        config['num_fragment_mixtures'] : int
            number of TKF92 fragment mixtures 
        
        config['num_site_mixtures'] : int
            number of mixtures associated with the EMISSIONS
        
        config['k_rate_mults'] : int
            number of rate multipliers
            
        config['rate_mult_range'] : (float, float)
            min and max rate multiplier
            DEFAULT: (0.01, 10)

        config['norm_rate_mults'] : bool
            if true, enforce constraint: 
            \sum_{c_transit} \sum_{c_sites} \sum_k 
                P(c_trans, c_sites, k) * \rho_{c_transit, c_sites, k} = 1
    
    name : str
        class name, for flax
    
    Methods here
    ------------
    setup
    __call__
    _set_model_simplification_flags
    _init_prob_logits
    _init_rate_logits
    _get_norm_factor
    
    Methods inherited from neural_models.model_utils.BaseClasses.ModuleBase
    ----------------------------------------------------------------
    sow_flax_intermeds
        for tensorboard logging
    
    """
    config: dict
    name: str
    
    def setup(self):
        """
        C_trans (C_frag + C_dom): number of mixtures associated with transitions (variable)
        C_sites: number of latent site classes
        K: numer of rate multipliers
        
        a shared method; shapes vary depending on how this module is used
        
        
        Flax Module Parameters
        -----------------------
        rate_mult_prob_logits : ArrayLike (C_trans, C_sites, K)
            logits for probability of selecting rate multiplier k;
            P(k|c_trans, c_sites)
        
        rate_mult_logits : ArrayLike (C_trans, C_sites, K)
            logits for rate multiplier k; \rho_{c_trans, c_sites, k}
        
        """
        ### read config file
        self.num_transit_mixtures = ( self.config['num_fragment_mixtures'] *
                                      self.config['num_domain_mixtures'] )# C_tr
        self.num_site_mixtures = self.config['num_site_mixtures'] # C_s
        self.k_rate_mults = self.config['k_rate_mults'] #K
        
        # optional
        self.rate_mult_min_val, self.rate_mult_max_val  = self.config.get( 'rate_mult_range', (0.01, 10) )
        
        # sometimes, might use model simplifications; also set 
        # norm_rate_mults flag here
        #
        # adds attributes:
        # > self.prob_rate_mult_is_one: P(k|...)=1, because no mixtures over 
        #   rates (but could site have mixtures over sites or transition classes;
        #   this just restricts the model to have one unique rate for ever one
        #   of these other classes)
        # > self.use_unit_rate_mult: \rho = 1, because no mixtures present at 
        #   all; sets norm_rate_mults to false
        self._set_model_simplification_flags()
        
            
        ### rate multipliers
        if not self.use_unit_rate_mult:
            self.rate_multiplier_activation = partial(bound_sigmoid,
                                               min_val = self.rate_mult_min_val,
                                               max_val = self.rate_mult_max_val)
            self._init_rate_logits() #(C_tr, C_s, K)
        
        
        ### probability of choosing a specific rate multiplier
        if not self.prob_rate_mult_is_one:
            self._init_prob_logits() #(C_tr, C_s, K)
    
        
    def __call__(self,
                 sow_flax_intermeds: bool,
                 log_site_class_probs: jnp.array,
                 log_transit_class_probs: jnp.array):
        """
        C_trans (C_frag + C_dom): number of mixtures associated with transitions (variable)
        C_sites: number of latent site classes
        K: number of rate multipliers
        
        Arguments
        ----------
        sow_flax_intermeds : bool
            switch for tensorboard logging
        
        log_site_class_probs : ArrayLike, (C_trans, C_sites)
            (from a different module); the log-probability of latent class 
            assignment for the emission site mixture
        
        log_transit_class_probs : ArrayLike, (C_trans,)
            (from a different module); the log-probability of latent class 
            assignment for the transition site mixture
        
        Returns
        -------
        log_rate_mult_probs : ArrayLike (C_trans, C_sites, K)
            the log-probability of having rate class k, given that the column 
            is assigned to latent site class c_st, in transit class c_trans
        
        rate_multipliers : ArrayLike, (C_trans, C_sites, K)
            the actual rate multiplier for rate class k, latent site class 
              c_sites, and latent transition class c_trans
          
        """
        # all outputs must be this size
        out_size = ( self.num_transit_mixtures, 
                      self.num_site_mixtures,
                      self.k_rate_mults ) #(C_tr, C_s, K)
            
        ### P(K|C_sites, C_trans)
        if not self.prob_rate_mult_is_one:
            log_rate_mult_probs = nn.log_softmax( self.rate_mult_prob_logits, axis=-1 ) #(C_tr, C_s, K) or (C_tr, K)
            
            # maybe sow outputs
            self.maybe_sow( sow_flax_intermeds = sow_flax_intermeds,
                            vals = log_rate_mult_probs,
                            label = f'{self.name}/logprob_rate_multiplier_class',
                            include_min_max = True,
                            include_perc_zeros = False)
            
        elif self.prob_rate_mult_is_one:
            log_rate_mult_probs = jnp.zeros( out_size ) #(C_tr, C_s, K) 
        
        
        ### \rho_{c_trans, c_sites, k}
        if not self.use_unit_rate_mult:
            rate_multipliers = self.rate_multiplier_activation( self.rate_mult_logits ) #(C_tr, C_s, K) or (C_tr, K)
    
            # maybe sow outputs
            self.maybe_sow( sow_flax_intermeds = sow_flax_intermeds,
                            vals = rate_multipliers,
                            label = f'{self.name}/rate_multipliers',
                            include_min_max = True,
                            include_perc_zeros = False)
        
            # possibly normalize to enforce average rate multiplier of one
            if self.norm_rate_mults:
                norm_factor = self._get_norm_factor(rate_multipliers = rate_multipliers,
                                                    log_transit_class_probs = log_transit_class_probs,
                                                    log_site_class_probs = log_site_class_probs,
                                                    log_rate_mult_probs = log_rate_mult_probs ) #float
                rate_multipliers = rate_multipliers / norm_factor #(C_tr, C_s, K) or (C_tr, K)
            
        elif self.use_unit_rate_mult:
            rate_multipliers = jnp.ones( out_size ) #(C_tr, C_s, K)
                
        return (log_rate_mult_probs, rate_multipliers)
    
    def _set_model_simplification_flags(self):
        """
        C_mix = C_sites + C_transits
        
        If C_mix = 1 and K = 1: no mixtures
            > prob_rate_mult_is_one = True
            > use_unit_rate_mult = True
            > don't ever have to normalize
        
        If C_mix > 1 and K = 1: then there's one unique rate per site class 
            (as was done in previous results); for each class, the probability
            of selecting the single possible rate multiplier is 1
            > prob_rate_mult_is_one = True
            > use_unit_rate_mult = False
            > may have to normalize rates by log_site_class_probs
        
        If C_mix = 1 and K > 1: no mixtures over site classes, but still
            have a mixture over rate multipliers (and potentially, transitions)
            > prob_rate_mult_is_one = False
            > use_unit_rate_mult = False
            > may have to normalize rates by log_rate_mult_probs
        
        If C_mix > 1 and K > 1: mixtures over both
            > prob_rate_mult_is_one = False
            > use_unit_rate_mult = False
            > may have to normalize rates by log_rate_mult_probs AND log_site_class_probs
        
        """
        # defaults
        prob_rate_mult_is_one = False
        use_unit_rate_mult = False
        norm_rate_mults = self.config.get('norm_rate_mults', True)
        
        # if k_rate_mults = 1, then P(k|c_trans, c_sites) = 1
        if self.k_rate_mults == 1:
            prob_rate_mult_is_one = True
            
            # if there are NO mixtures, then just use unit rate multiplier
            # also NEVER normalize the rate multiplier (since its just one)
            if (self.num_transit_mixtures * self.num_site_mixtures) == 1:
                use_unit_rate_mult = True
                norm_rate_mults = False
                
        self.prob_rate_mult_is_one = prob_rate_mult_is_one
        self.use_unit_rate_mult = use_unit_rate_mult
        self.norm_rate_mults = norm_rate_mults
        
    
    def _init_prob_logits(self):
        """
        initialize the (C_trans, C_sites, K) logits for P(K|C_trans, C_sites)
        
        self.rate_mult_prob_logits is the flax parameter
        """
        out_size = ( self.num_transit_mixtures, 
                     self.num_site_mixtures,
                     self.k_rate_mults ) #(C_tr, C_s, K)
        
        self.rate_mult_prob_logits = self.param('rate_mult_prob_logits',
                                        nn.initializers.normal(),
                                        out_size,
                                        jnp.float32)  #(C_tr, C_s, K)
    
    def _init_rate_logits(self):
        """
        initialize the (C_trans, C_sites, K) logits for \rho_{c_trans, c_sites, k}
        self.rate_mult_logits is the flax parameter
        """
        out_size = ( self.num_transit_mixtures, 
                     self.num_site_mixtures,
                     self.k_rate_mults ) #(C_tr, C_s, K)
        
        self.rate_mult_logits = self.param('rate_mult_logits',
                                           nn.initializers.normal(),
                                           out_size,
                                           jnp.float32)  #(C_tr, C_s, K)
    
    def _get_norm_factor(self,
                         rate_multipliers: jnp.array,
                         log_transit_class_probs: jnp.array,
                         log_site_class_probs: jnp.array,
                         log_rate_mult_probs: jnp.array,
                         *args,
                         **kwargs):
        """
        return the normalization factor needed for constraint:
            \sum_{c_transit} \sum_{c_sites} \sum_k 
                P(c_trans, c_sites, k) * \rho_{c_transit, c_sites, k} = 1
        
        Arguments:
        ----------
        rate_multipliers : ArrayLike, (C_trans, C_sites, K)
            \rho_{c_trans, c_sites, k}; the un-normalized rate multipliers
            
        log_transit_class_probs : ArrayLike, (C_trans)
            P(c_trans); marginal probability of transition latent class 
            assignment c_trans (for example, fragment type c_frag)
        
        log_site_class_probs : ArrayLike, (C_trans, C_sites)
            P(c_sites | c_trans); probability of site class, given transition
            latent class assignment
        
        log_rate_mult_probs : ArrayLike, (C_trans, C_sites, K)
            P(k | c_trans, c_sites); probability of assigning a specific rate
            multiplier, given transition and site class assignment
        
        
        Returns:
        --------
        norm_factor : float
        
        """
        # logP(C_trans) + logP(C_sites | C_trans) + logP(K|C_sites, C_trans) =
        #   logP(C_trans, C_sites, K)
        log_joint_mix_weight = ( log_transit_class_probs[...,None,None] +
                                 log_site_class_probs[...,None] +
                                 log_rate_mult_probs ) #(C_tr, C_s, K)
        
        # P(C_trans, C_sites, K) = exp( logP(C_trans, C_sites, K) )
        joint_mix_weight = jnp.exp( log_joint_mix_weight ) #(C_tr, C_s, K)
        
        # normalization factor is
        #   sum_{c_trans, c_sites, k} 
        #   P(C_trans, C_sites, K) * \rho(c_trans, c_sites, k)
        norm_factor = jnp.multiply(joint_mix_weight, rate_multipliers).sum() #float
        
        return norm_factor
        
    
class IndpRateMultipliers(RateMultipliersPerClass):
    """
    C_trans (C_frag + C_dom): number of mixtures associated with transitions (variable)
    C_sites: number of latent site classes
    K: numer of rate multipliers
    
    Generate C_trans * C_sites * K rate multipliers, and 
      probabilty of rate multiplier k, given mixture classes: P(k|c_site, c_trans)
    
    THIS ASSUMES K IS INDEPENDENT OF C_sites and C_trans (past models make
      this assumption)
    
    
    Initialize with
    ----------------
    config : dict
        config['num_domain_mixtures'] : int
            number of larger TKF92 domain mixtures (>1 if nested TKF92)
    
        config['num_fragment_mixtures'] : int
            number of TKF92 fragment mixtures 
            
        config['num_site_mixtures'] : int
            number of mixtures associated with the EMISSIONS
    
        config['k_rate_mults'] : int
            number of rate multipliers
            
        config['rate_mult_range'] : (float, float)
            min and max rate multiplier
            DEFAULT: (0.01, 10)

        config['norm_rate_mults'] : bool
            if true, enforce constraint: \sum_k P(k)*\rho_k = 1
    
    name : str
        class name, for flax
    
    Methods here
    ------------
    _set_model_simplification_flags
    _init_prob_logits
    _init_rate_logits
    _get_norm_factor
    
    Methods inherited from RateMultipliersPerClass
    ------------------------------------------------
    __call__
    setup
    
    Methods inherited from neural_models.model_utils.BaseClasses.ModuleBase
    ----------------------------------------------------------------
    sow_flax_intermeds
        for tensorboard logging
    
    """
    config: dict
    name: str
      
    def _set_model_simplification_flags(self):
        """
        C_mix = C_sites + C_transits
        
        If C_mix = 1 and K = 1: no mixtures
            > prob_rate_mult_is_one = True
            > use_unit_rate_mult = True
            > dont ever have to normalize
        
        If C_mix > 1 and K = 1: no mixtures over rate multipliers; all classes 
            use rate multiplier of 1
            > prob_rate_mult_is_one = True
            > use_unit_rate_mult = True
            > also dont ever have to normalize, since rate multipliers are
              independent of class label
        
        If C_mix = 1 and K > 1: no mixtures over site classes, but still
            have a mixture over rate multipliers
            > prob_rate_mult_is_one = False
            > use_unit_rate_mult = False
            > might normalize over log_rate_mult_probs
        
        If C_mix > 1 and K > 1: mixtures over both, but the same mixture over rate 
            multipliers is used for all possible latent site class labels
            > prob_rate_mult_is_one = False
            > use_unit_rate_mult = False
            > might normalize over log_rate_mult_probs (but not 
              log_site_class_probs or log_transit_class_probs,
              since this is independent of other mixture class labels)
        """
        if self.k_rate_mults == 1:
            self.prob_rate_mult_is_one = True
            self.use_unit_rate_mult = True
            self.norm_rate_mults = False
        
        elif self.k_rate_mults > 1:
            self.prob_rate_mult_is_one = False
            self.use_unit_rate_mult = False
            self.norm_rate_mults = self.config.get('norm_rate_mults', True)
        
        
    def _init_prob_logits(self):
        """
        initialize the (1, 1, K) logits (have dummy axes for other mixtures) 
          for P(k)
        
        self.rate_mult_prob_logits is a flax parameter
        """
        self.rate_mult_prob_logits = self.param( 'rate_mult_prob_logits',
                                            nn.initializers.normal(),
                                            (1, 1, self.k_rate_mults),
                                            jnp.float32 ) #(1, 1, K)
    
    def _init_rate_logits(self):
        """
        initialize the (1, 1, K) logits (have dummy axes for other mixtures) 
          for rate multiplier \rho_{k}
        
        self.rate_mult_logits is a flax parameter
        """
        self.rate_mult_logits = self.param( 'rate_mult_logits',
                                       nn.initializers.normal(),
                                       (1, 1, self.k_rate_mults),
                                       jnp.float32 ) #(1,1,K)
        
    def _get_norm_factor(self,
                         rate_multipliers: jnp.array,
                         log_rate_mult_probs: jnp.array,
                         *args,
                         **kwargs):
        """
        return the normalization factor needed for constraint:
            \sum_k P(k) * \rho_{k} = 1
        
        Arguments:
        ----------
        rate_multipliers : ArrayLike, (1,1,K)
            \rho_{k}; the un-normalized rate multipliers
            
        log_rate_mult_probs : ArrayLike, (1,1,K)
            P(k); probability of having rate class k
        
        
        Returns:
        --------
        norm_factor : float
        
        """
        # exp( logP(K) ) = P(K)
        mix_weights = jnp.exp(log_rate_mult_probs) #(1, 1, K)
        
        # \sum_k P(K) \rho_{k} 
        norm_factor = jnp.multiply( mix_weights, rate_multipliers ).sum() #float
        
        return norm_factor

class RateMultipliersPerClassFromFile(RateMultipliersPerClass):
    """
    C_trans (C_frag + C_dom): number of mixtures associated with transitions (variable)
    C_sites: number of latent site classes
    K: number of rate multipliers
    
    load probabilities and rate multipliers from files
    
    
    Initialize with
    ----------------
    config : dict
        config['num_domain_mixtures'] : int
            number of larger TKF92 domain mixtures (>1 if nested TKF92)
    
        config['num_fragment_mixtures'] : int
            number of TKF92 fragment mixtures 
    
        config['num_site_mixtures'] : int
            number of mixtures associated with the EMISSIONS
            
        config['k_rate_mults'] : int
            number of rate multipliers
    
        config['filenames']['rate_mults'] : str
            file of rate multipliers
            
        config['filenames']['rate_mult_probs'] : str
            file of probabilities
            
        config['norm_rate_mults'] : bool
            if true, enforce constraint: 
            \sum_{c_transit} \sum_{c_sites} \sum_k 
                P(c_trans, c_sites, k) * \rho_{c_transit, c_sites, k} = 1
    
    name : str
        class name, for flax
    
    Methods here
    ------------
    setup
    __call__
    
    Methods inherited from RateMultipliersPerClass
    ------------------------------------------------
    _set_model_simplification_flags
    _get_norm_factor
    
    """
    def setup(self):
        ### read config file
        self.num_transit_mixtures = ( self.config['num_fragment_mixtures'] *
                                      self.config['num_domain_mixtures'] )# C_tr
        self.num_site_mixtures = self.config['num_site_mixtures'] #C_s
        self.k_rate_mults = self.config['k_rate_mults'] #K
        out_size = ( self.num_transit_mixtures,
                     self.num_site_mixtures, 
                     self.k_rate_mults )
        
        # possibly simplify model setup
        self._set_model_simplification_flags()
        
        
        ### read files: rate multipliers
        if not self.use_unit_rate_mult:
            in_file = self.config['filenames']['rate_mults']
            self.rate_multipliers = _load_params(in_file, target_ndim=3) #(C_tr, C_s, K)
            del in_file
        
        elif self.use_unit_rate_mult:
            self.rate_multipliers = jnp.ones( out_size ) #(C_tr, C_s, K)
        
        assert self.rate_multipliers.shape[0] == self.num_transit_mixtures
        assert self.rate_multipliers.shape[1] == self.num_site_mixtures
        assert self.rate_multipliers.shape[2] == self.k_rate_mults
        
            
        ### read files: P(k|c_trans, c_sites)
        if not self.prob_rate_mult_is_one:
            in_file =  self.config['filenames']['rate_mult_probs']
            rate_mult_probs = _load_params(in_file, target_ndim=3) #(C_tr, C_s, K)
            self.log_rate_mult_probs = safe_log(rate_mult_probs) #(C_tr, C_s, K)
            del in_file
            
        elif self.prob_rate_mult_is_one:
            self.log_rate_mult_probs = jnp.zeros( out_size ) #(C_tr, C_s, K)
        
        assert self.log_rate_mult_probs.shape[0] == self.num_transit_mixtures
        assert self.log_rate_mult_probs.shape[1] == self.num_site_mixtures
        assert self.log_rate_mult_probs.shape[2] == self.k_rate_mults
            
        
    def __call__(self,
                 sow_flax_intermeds: bool,
                 log_site_class_probs: jnp.array,
                 log_transit_class_probs: jnp.array):
        """
        C_trans (C_frag * C_dom): number of mixtures associated with transitions (variable)
        C_sites: number of latent site classes
        K: number of rate multipliers
        
        
        Arguments
        ----------
        sow_flax_intermeds : bool
            switch for tensorboard logging
        
        log_site_class_probs : ArrayLike, (C_trans, C_cites)
            (from a different module); the log-probability of latent class 
            assignment for the emission site mixture
        
        log_transit_class_probs : ArrayLike, (C_trans,)
            (from a different module); the log-probability of latent class 
            assignment for the transition site mixture
        
        Returns
        -------
        log_rate_mult_probs : ArrayLike (C_trans, C_sites, K)
            the log-probability of having rate class k, given that the column 
            is assigned to latent class c
        
        rate_multipliers : ArrayLike, (C_trans, C_sites, K)
            the actual rate multiplier for rate class k, latent site class 
              c_sites, and latent transition class c_trans
          
        """
        # P(K|C_trans, C_sites), \rho_{c_trans, c_sites, k}
        log_rate_mult_probs = self.log_rate_mult_probs #(C_tr, C_s, K)
        rate_multipliers = self.rate_multipliers #(C_tr, C_s, K)
        
        # possibly normalize
        if self.norm_rate_mults:
            norm_factor = self._get_norm_factor(rate_multipliers = rate_multipliers,
                                                log_transit_class_probs = log_transit_class_probs,
                                                log_site_class_probs = log_site_class_probs,
                                                log_rate_mult_probs = log_rate_mult_probs ) #float
            rate_multipliers = rate_multipliers / norm_factor #(C_tr, C_s, K) or (C_tr, K)
        
        return (log_rate_mult_probs, rate_multipliers)


class IndpRateMultipliersFromFile(IndpRateMultipliers):
    """
    C_trans (C_frag + C_dom): number of mixtures associated with transitions (variable)
    C_sites: number of latent site classes
    K: number of rate multipliers
    
    load probabilities and rate multipliers from files
    
    Initialize with
    ----------------
    config : dict
        config['num_domain_mixtures'] : int
            number of larger TKF92 domain mixtures (>1 if nested TKF92)
    
        config['num_fragment_mixtures'] : int
            number of TKF92 fragment mixtures 
    
        config['num_site_mixtures'] : int
            number of mixtures associated with the EMISSIONS
            
        config['k_rate_mults'] : int
            number of rate multipliers
    
        config['filenames']['rate_mults'] : str
            file of rate multipliers
            
        config['filenames']['rate_mult_probs'] : str
            file of probabilities
            
        config['norm_rate_mults'] : bool
            if true, enforce constraint: \sum_c \sum_k P(c,k) * \rho_{c,k} = 1
    
    name : str
        class name, for flax
    
    Methods here
    ------------
    setup
    __call__
    
    Methods inherited from IndpRateMultipliers
    -------------------------------------------
    _set_model_simplification_flags
    _get_norm_factor
    
    """
    def setup(self):
        ### read config file
        self.num_transit_mixtures = ( self.config['num_fragment_mixtures'] *
                                      self.config['num_domain_mixtures'] )# C_tr
        self.num_site_mixtures = self.config['num_site_mixtures'] #C_s
        self.k_rate_mults = self.config['k_rate_mults'] #K
        
        # possibly simplify model setup; also set 
        # norm_rate_mults flag here
        #
        # adds attributes:
        # > self.prob_rate_mult_is_one: P(k|...)=1, because no mixtures over 
        #   rates (but could site have mixtures over sites or transition classes;
        #   this just restricts the model to have one unique rate for ever one
        #   of these other classes)
        # > self.use_unit_rate_mult: \rho = 1, because no mixtures present at 
        #   all; sets norm_rate_mults to false
        self._set_model_simplification_flags()
        
        
        ### read files: rate multipliers
        if not self.use_unit_rate_mult:
            in_file = self.config['filenames']['rate_mults']
            self.rate_multipliers = _load_params(in_file, target_ndim=3) #(1, 1, K)
            del in_file
        
        elif self.use_unit_rate_mult:
            self.rate_multipliers = jnp.ones( (1, 1, self.k_rate_mults,) ) #(1, 1, K)
            
            
        ### read files: P(k|c_trans, c_sites)
        if not self.prob_rate_mult_is_one:
            in_file =  self.config['filenames']['rate_mult_probs']
            self.log_rate_mult_probs = safe_log( _load_params(in_file, target_ndim=3) ) #(1, 1, K)
            del in_file
            
        elif self.prob_rate_mult_is_one:
            self.log_rate_mult_probs = jnp.zeros( (1, 1, self.k_rate_mults) ) #(1, 1, K)
    
    
    def __call__(self,
                 sow_flax_intermeds: bool,
                 *args,
                 **kwargs):
        """
        Arguments
        ----------
        sow_flax_intermeds : bool
            switch for tensorboard logging
        
        Returns
        -------
        log_rate_mult_probs : ArrayLike (1, 1, K)
            the log-probability of having rate class k
        
        rate_multipliers : ArrayLike, (1, 1, K)
            the actual rate multiplier for rate class k
          
        """
        # P(K|C_trans, C_sites), \rho_{c_trans, c_sites, k}
        log_rate_mult_probs = self.log_rate_mult_probs #(1, 1, K)
        rate_multipliers = self.rate_multipliers #(1, 1, K)
        
        # possibly normalize
        if self.norm_rate_mults:
            norm_factor = self._get_norm_factor(rate_multipliers = rate_multipliers,
                                                log_rate_mult_probs = log_rate_mult_probs ) #float
            rate_multipliers = rate_multipliers / norm_factor #(1,1,K)
            
        return (log_rate_mult_probs, rate_multipliers)
    

###############################################################################
### EQUILIBRIUM DISTRIBUTION MODELS   #########################################
###############################################################################
class EqulDistLogprobsPerClass(ModuleBase):
    """
    C_trans (C_frag + C_dom): number of mixtures associated with transitions (variable)
    C_sites: number of latent site classes
    A: alphabet size
    
    Equilibrium distribution of emissions, as well as probability of 
      site-level classes (i.e. latent site classes over EMISSIONS)
    
    
    Initialize with
    ----------------
    config : dict
        config['num_domain_mixtures'] : int
            number of larger TKF92 domain mixtures (>1 if nested TKF92)
    
        config['num_fragment_mixtures'] : int
            number of TKF92 fragment mixtures 
    
        config['num_site_mixtures'] : int
            number of mixtures associated with the EMISSIONS
            
        config['emission_alphabet_size'] : int
            size of emission alphabet; 20 for proteins, 4 for DNA
            
    name : str
        class name, for flax
    
    Methods here
    ------------
    setup
    __call__
    
    Methods inherited from neural_models.model_utils.BaseClasses.ModuleBase
    ----------------------------------------------------------------
    sow_flax_intermeds
        for tensorboard logging
    
    """
    config: dict
    name: str
    
    def setup(self):
        """
        C_trans (C_frag + C_dom): number of mixtures associated with transitions (variable)
        C_sites: number of latent site classes
        A: alphabet size
        
        
        Flax Module Parameters
        -----------------------
        equl_dist_logits : ArrayLike (C_trans, C_sites, A)
            logits for the equilibrium distribution over emitted characters,
            one distribution for every site class c_site a,d transit class 
            c_trans
        
        site_class_prob_logits : ArrayLike (C_trans, C_sites)
            logits for probability of being in site class c_site
            
        
        """
        ### read config file
        self.num_transit_mixtures = ( self.config['num_fragment_mixtures'] *
                                      self.config['num_domain_mixtures'] )# C_tr
        self.num_site_mixtures = self.config['num_site_mixtures'] #C_s
        emission_alphabet_size = self.config['emission_alphabet_size'] #A
        
        
        ### init flax parameters
        # equilibrium distributions
        out_size = ( self.num_transit_mixtures, 
                     self.num_site_mixtures,
                     emission_alphabet_size )
        self.equl_dist_logits = self.param('equl_dist_logits',
                                           nn.initializers.normal(),
                                           out_size,
                                           jnp.float32) #(C_tr, C_s, A)
        del out_size
        
        # probability of emission site classes
        out_size = ( self.num_transit_mixtures, 
                     self.num_site_mixtures )
        self.site_class_prob_logits = self.param('site_class_prob_logits',
                                           nn.initializers.normal(),
                                           out_size,
                                           jnp.float32) #(C_tr, C_s)
        
        
    def __call__(self,
                 sow_flax_intermeds: bool ):
        """
        C_trans (C_frag + C_dom): number of mixtures associated with transitions (variable)
        C_sites: number of latent site classes
        A: alphabet size
        
        Arguments
        ----------
        sow_flax_intermeds : bool
            switch for tensorboard logging
          
        Returns
        -------
        log_site_class_probs : ArrayLike (C_trans, C_sites)
            log-probability of site classes; P(C_sites | C_trans)

        log_equl_dist : ArrayLike, (C_trans, C_sites, A)
            log-transformed equilibrium distribution
        """
        ### equilibrium distribution
        log_equl_dist = nn.log_softmax( self.equl_dist_logits, axis = -1 ) #(C_tr, C_s, A)

        # maybe sow
        self.maybe_sow( sow_flax_intermeds = sow_flax_intermeds,
                        vals = log_equl_dist,
                        label = f'{self.name}/log_equl_dists',
                        include_min_max = True,
                        include_perc_zeros = False)
        
        
        ### P(C_sites | C_trans)
        log_site_class_probs = nn.log_softmax( self.site_class_prob_logits, axis = -1 ) #(C_tr, C_s)
        
        # maybe sow
        self.maybe_sow( sow_flax_intermeds = sow_flax_intermeds,
                        vals = log_site_class_probs,
                        label = f'{self.name}/logprob_site_classes',
                        include_min_max = True,
                        include_perc_zeros = False)
        
        return ( log_site_class_probs, log_equl_dist )


class EqulDistLogprobsFromFile(ModuleBase):
    """
    Load equilibrium distribution and log-probability of site classes from file
    
    
    Initialize with
    ----------------
    config : dict
        config['num_domain_mixtures'] : int
            number of larger TKF92 domain mixtures (>1 if nested TKF92)
    
        config['num_fragment_mixtures'] : int
            number of TKF92 fragment mixtures 
    
        config['num_site_mixtures'] : int
            number of mixtures associated with the EMISSIONS
            
        config["filenames"]["equl_dist"]: str
              file of equilibrium distributions to load
        
        config["filenames"]["site_class_probs"]: str
              file of site class probabilities to load
            
    name : str
        class name, for flax
    
    Methods here
    ------------
    setup
    __call__
    
    """
    config: dict
    name: str
    
    def setup(self):
        """
        
        Flax Module Parameters
        -----------------------
        none
        
        """
        ### read config file
        self.num_transit_mixtures = ( self.config['num_fragment_mixtures'] *
                                      self.config['num_domain_mixtures'] )# C_tr
        self.num_site_mixtures = self.config['num_site_mixtures'] # C_s
        self.emission_alphabet_size = self.config['emission_alphabet_size'] #A
        
        equl_file = self.config['filenames']['equl_dist']

        if (self.num_transit_mixtures * self.num_site_mixtures) > 1:
            site_class_probs_file = self.config['filenames']['site_class_probs']
        
        
        ### load params
        # equilibrium distribution
        equl_dist = _load_params(equl_file, target_ndim=3) #(C_tr, C_s, A)
        self.log_equl_dist = safe_log(equl_dist) #(C_tr, C_s, A)
        assert self.log_equl_dist.shape[0] == self.num_transit_mixtures
        assert self.log_equl_dist.shape[1] == self.num_site_mixtures
        assert self.log_equl_dist.shape[2] == self.emission_alphabet_size
        
        # probability of site classes
        if (self.num_transit_mixtures * self.num_site_mixtures) > 1:
            site_class_probs = _load_params(site_class_probs_file, target_ndim=2) #(C_tr, C_s)
            self.log_site_class_probs = safe_log(site_class_probs) #(C_tr, C_s)
            
        elif (self.num_transit_mixtures * self.num_site_mixtures) == 1:
            self.log_site_class_probs = jnp.zeros( (1,1) ) #(C_tr=1, C_s=1)


        assert self.log_site_class_probs.shape[0] == self.num_transit_mixtures
        assert self.log_site_class_probs.shape[1] == self.num_site_mixtures

        
    def __call__(self,
                 *args,
                 **kwargs):
        """
        Returns
        -------
        log_site_class_probs : ArrayLike
            log-probability of site classes; P(C_sites | C_trans)

        log_equl_dist : ArrayLike, (C_trans, C_sites, A)
            log-transformed equilibrium distribution
        """
        return ( self.log_site_class_probs, self.log_equl_dist )
    

class EqulDistLogprobsFromCounts(ModuleBase):
    """
    If there's only one site and transition class, construct an equilibrium 
      distribution from observed frequencies
    
    A = alphabet size
    
    
    Initialize with
    ----------------
    config : dict
        config["training_dset_emit_counts"] : ArrayLike, (A,)
            observed counts to turn into frequencies
            
    name : str
        class name, for flax
    
    Methods here
    ------------
    setup
    __call__
    """
    config: dict
    name: str
    
    def setup(self):
        """
        
        Flax Module Parameters
        -----------------------
        none
        
        """
        training_dset_emit_counts = self.config['training_dset_emit_counts'] #(A,)
        equl_dist = training_dset_emit_counts / ( training_dset_emit_counts.sum() ) #(A,)
        log_equl_dist = safe_log( equl_dist ) #(A,)
        
        # C_trans = 1, C_sites = 1
        self.log_equl_dist = log_equl_dist[None,None,...] #(1, 1, A)
        self.log_site_class_probs = jnp.zeros( (1,1) ) #(1, 1)
        
    def __call__(self,
                 *args,
                 **kwargs):
        """
        Returns
        -------
        log_site_class_probs : ArrayLike
            log-probability of site classes; P(C_sites | C_trans)

        log_equl_dist : ArrayLike, (C_trans, C_sites, A)
            log-transformed equilibrium distribution
        """
        return ( self.log_site_class_probs, self.log_equl_dist )


###############################################################################
### Substitution Models: Generate time reversible   ###########################
###############################################################################
class GTRLogprobs(ModuleBase):
    """
    Get the conditional and joint logprobs for a GTR model
    
    
    Initialize with
    ----------------
    config : dict
        config['random_init_exchanges'] : bool
            whether or not to initialize exchangeabilities from random; if 
            not random, need to provide filename of exchangeabilities
            
        config['norm_rate_matrix'] : bool
            flag to normalize rate matrix to t = one substitution
            
        config['exchange_range'] : List[float, float], optional
            exchangeabilities undergo bound_sigmoid transformation, this
            specifies the min and max
            Default is (1e-4, 12)
        
        config['filenames']['exch'] : str, optional
            name of the exchangeabilities to intiialize with, if desired
            Default is None
        
    name : str
        class name, for flax
    
    Methods here
    ------------
    setup
    __call__
    _get_square_exchangeabilities_matrix
    
    Methods inherited from neural_models.model_utils.BaseClasses.ModuleBase
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
        exchangeabilities_logits_vec : ArrayLike, (n,)
            upper triangular values for exchangeability matrix
            190 for proteins, 6 for DNA
            Usually initialize logits from LG08 exchangeabilities
        
        """
        ###################
        ### read config   #
        ###################
        # required
        emission_alphabet_size = self.config['emission_alphabet_size']
        
        # optional
        self.exchange_min_val, self.exchange_max_val  = self.config.get( 'exchange_range', (1e-4, 12) )
        self.random_init_exchanges = self.config.get('random_init_exchanges', True)
        self.norm_rate_matrix = self.config.get('norm_rate_matrix', True)
        
        # only needed if loading random_init_exchanges is False
        exchangeabilities_file = self.config.get('filenames', {}).get('exch', None)
        
        
        ##########################################################
        ### Parameter: exchangeabilities as a flattened vector   # 
        ### of upper triangular values                           #
        ##########################################################
        # N = 190 for proteins
        # N =  6 for DNA
        
        ### activation is bound sigmoid; setup the activation function with 
        ###   min/max values
        self.exchange_activation = partial(bound_sigmoid,
                                           min_val = self.exchange_min_val,
                                           max_val = self.exchange_max_val)
        
        ### initialization
        # init from file
        if not self.random_init_exchanges:
            with open(exchangeabilities_file,'rb') as f:
                vec = jnp.load(f)
        
            transformed_vec = bound_sigmoid_inverse(vec, 
                                                    min_val = self.exchange_min_val,
                                                    max_val = self.exchange_max_val)
        
            self.exchangeabilities_logits_vec = self.param("exchangeabilities_logits_vec", 
                                                           lambda rng, shape: transformed_vec,
                                                           transformed_vec.shape ) #(N,)
        
        # init from random
        elif self.random_init_exchanges:
            A = emission_alphabet_size
            num_exchange = int( (A * (A-1)) / 2 )
            self.exchangeabilities_logits_vec = self.param("exchangeabilities_logits_vec", 
                                                           nn.initializers.normal(),
                                                           (num_exchange,),
                                                           jnp.float32 ) #(N,)
        
        
    def __call__(self,
                 log_equl_dist: jnp.array,
                 rate_multipliers: jnp.array,
                 t_array: jnp.array, 
                 sow_flax_intermeds: bool,
                 return_cond: bool,
                 return_intermeds: bool = False,
                 *args,
                 **kwargs):
        """
        C_trans (C_frag + C_dom): number of mixtures associated with transitions (variable)
        C_sites: number of latent site classes
        K = number of site rates
        A: alphabet size
        
        
        Arguments
        ----------
        log_equl_dist : ArrayLike, (C_trans, C_sites, A)
            log-transformed equilibrium distribution
        
        rate_multipliers : ArrayLike, (C_trans, C_sites, K)
            the actual rate multiplier for rate class k, latent site class 
              c_sites, and latent transition class c_trans
        
        t_array : ArrayLike, (T,) or (B,)
            either one time grid for all samples (T,) or unique branch
            length for each sample (B,)
        
        sow_flax_intermeds : bool
            switch for tensorboard logging
        
        return_cond : bool
            whether or not to return conditional logprob
        
        return_intermeds : bool
            whether or not to return intermediate values:
                > exchangeabilities: (A, A)
                > rate_matrix: (C_trans, C_sites, A, A)
          
        Returns
        -------
        ArrayLike, (T, C_trans, C_sites, K, A, A)
            either joint or conditional logprob of emissions at match sites;
            NOT YET SCALED BY ANY CLASS/RATE PROBABILITIES!!!
        """
        # undo log transform on equilibrium
        equl = jnp.exp(log_equl_dist) #(C_tr, C_s, A)
        
        # 1.) fill in square matrix, \chi
        exchangeabilities_mat = self._get_square_exchangeabilities_matrix(sow_flax_intermeds) #(A, A)
        
        # 2.) prepare rate matrix Q_c = \chi * \diag(\pi_c); normalize if desired
        rate_matrix_Q = rate_matrix_from_exch_equl( exchangeabilities = exchangeabilities_mat,
                                                    equilibrium_distributions = equl,
                                                    norm=self.norm_rate_matrix ) #(C_tr, C_s, A, A)
        
        # 3.) scale by rate multipliers, \rho_{C_tr, C_s, k}
        rate_matrix_times_rho = scale_rate_matrix(subst_rate_mat = rate_matrix_Q,
                                                  rate_multipliers = rate_multipliers) #(C_tr, C_s, K, A, A)
        
        # 4.) apply matrix exponential to get conditional logprob
        # cond_logprobs is either (T, C_tr, C_s, K, A, A) or (B, C_tr, C_s, K, A, A)
        cond_logprobs = cond_logprob_emit_at_match_per_mixture( t_array = t_array,
                                                                scaled_rate_mat_per_mixture = rate_matrix_times_rho )
        
        if return_cond:
            logprobs = cond_logprobs
        
        # 5.) multiply by equilibrium distributions to get joint logprob
        elif not return_cond:
            # joint_logprobs is either (T, C_tr, C_s, K, A, A) or (B, C_tr, C_s, K, A, A)
            # NOTE: this uses original log_equl_dist (before exp() operation)
            logprobs = joint_logprob_emit_at_match_per_mixture( cond_logprob_emit_at_match_per_mixture = cond_logprobs,
                                                                log_equl_dist_per_mixture = log_equl_dist )
        
        
        ### optionally, return intermediates too; useful for final eval or debugging
        if return_intermeds:
            intermeds_dict = {'exchangeabilities': exchangeabilities_mat,
                              'rate_matrix': rate_matrix_Q}
        
        elif not return_intermeds:
            intermeds_dict = {}
        
        return logprobs, intermeds_dict
    
    
    def _get_square_exchangeabilities_matrix(self,
                                             sow_flax_intermeds: bool):
        ### apply activation of choice
        # number of upper triangular values for alphabet size A is (A * A-1)/2
        # upper_triag_values will have this many values
        # for proteins: upper_triag_values is (190,)
        # for DNA: upper_triag_values is (6,)
        upper_triag_values = self.exchange_activation( self.exchangeabilities_logits_vec )
        
        # maybe sow
        self.maybe_sow( sow_flax_intermeds = sow_flax_intermeds,
                        vals = upper_triag_values,
                        label = f'{self.name}/gtr_exchangeabilities',
                        include_min_max = True,
                        include_perc_zeros = False)
    
        
        ### fill in square matrix
        exchangeabilities_mat = upper_tri_vector_to_sym_matrix( upper_triag_values ) #(A, A)
        
        return exchangeabilities_mat
        

class GTRLogprobsFromFile(GTRLogprobs):
    """
    Like GTRLogprobs, but load parameters as-is
        
        
    Initialize with
    ----------------
    config : dict
        config['filenames']['exch'] : str
            name of the exchangeabilities to load
        
        config['norm_rate_matrix'] : bool
            flag to normalize rate matrix to t = one substitution
            
    name : str
        class name, for flax
    
    Methods here
    ------------
    setup
    _get_square_exchangeabilities_matrix
    
    Methods inheried from GTRLogprobs
    ----------------------------------
    __call__
    
    """
    config: dict
    name: str
    
    def setup(self):
        """
        Flax Module Parameters
        -----------------------
        None
        
        """
        ###################
        ### read config   #
        ###################
        self.norm_rate_matrix = self.config.get('norm_rate_matrix',True)
        exchangeabilities_file = self.config['filenames']['exch']
        
        
        ###################################
        ### Read exchangeabilities file   #
        ###################################
        with open(exchangeabilities_file,'rb') as f:
            exch_from_file = jnp.load(f)
        
        # if providing a vector, need to prepare a square exchangeabilities matrix
        if len(exch_from_file.shape) == 1:
            self.exchangeabilities_mat = upper_tri_vector_to_sym_matrix( exch_from_file ) #(A, A)
            
        # otherwise, use the matrix as-is
        else:
            self.exchangeabilities_mat = exch_from_file #(A, A)
        
        
    def _get_square_exchangeabilities_matrix(self,
                                             *args,
                                             **kwargs):
        return self.exchangeabilities_mat


class LG08Logprobs(GTRLogprobsFromFile):
    """
    Specifically load LG08 exchangeabilities
        
        
    Initialize with
    ----------------
    config : dict
        config['norm_rate_matrix'] : bool
            flag to normalize rate matrix to t = one substitution
            
    name : str
        class name, for flax
    
    Methods here
    ------------
    setup
    
    Methods inheried from GTRLogprobsFromFile
    ----------------------------------
    __call__
    _get_square_exchangeabilities_matrix
    
    """
    config: dict
    name: str
    
    def setup(self):
        """
        Flax Module Parameters
        -----------------------
        None
        
        """
        ###################
        ### read config   #
        ###################
        self.norm_rate_matrix = self.config.get('norm_rate_matrix',True)
        
        
        ###################################
        ### Read exchangeabilities file   #
        ###################################
        # read LG08 rate matrix
        with open(f'older_indel_models/LG08_exchangeability_r.npy', 'rb') as f:
            exch_from_file = jnp.load(f) #(20, 20)
        self.exchangeabilities_mat = exch_from_file #(A, A)
        


###############################################################################
### PROBABILITY MATRICES: F81   ###############################################
###############################################################################
class F81Logprobs(ModuleBase):
    """
    Get the conditional and joint logprobs for an F81 model; doesn't
        really need to be a flax module, but keep for consistency with
        GTR models
    
    
    Initialize with
    ----------------
    config['norm_rate_matrix'] : bool
        flag to normalize rate matrix to t = one substitution
            
    name : str
        class name, for flax
    
    Methods here
    ------------
    setup
    __call__
    
    """
    config: dict
    name: str
    
    def setup(self):
        """
        Flax Module Parameters
        -----------------------
        None
        
        """
        self.norm_rate_matrix = self.config.get('norm_rate_matrix', True)
        
        
    def __call__(self,
                 log_equl_dist: jnp.array,
                 rate_multipliers: jnp.array,
                 t_array: jnp.array,
                 sow_flax_intermeds: bool,
                 return_cond: bool,
                 *args,
                 **kwargs):
        """
        C_trans (C_frag + C_dom): number of mixtures associated with transitions (variable)
        C_sites: number of latent site classes
        K = number of site rates
        A: alphabet size
        
        
        Arguments
        ----------
        log_equl_dist : ArrayLike, (C_trans, C_sites, A)
            log-transformed equilibrium distribution
        
        rate_multipliers : ArrayLike, (C_trans, C_sites, K)
            the actual rate multiplier for rate class k, latent site class 
              c_sites, and latent transition class c_trans
        
        t_array : ArrayLike, (T,) or (B,)
            either one time grid for all samples (T,) or unique branch
            length for each sample (B,)
        
        sow_flax_intermeds : bool
            switch for tensorboard logging
        
        return_cond : bool
            whether or not to return conditional logprob
        
        
        Returns
        -------
        ArrayLike, (T, C_trans, C_sites, K, A, A)
            either joint or conditional logprob of emissions at match sites;
            NOT YET SCALED BY ANY CLASS/RATE PROBABILITIES!!!
        """
        # undo log transform on equilibrium
        equl = jnp.exp(log_equl_dist) #(C_tr, C_s, A)
        
        logprobs = fill_f81_logprob_matrix( equl = equl, 
                                        rate_multipliers = rate_multipliers, 
                                        norm_rate_matrix = self.norm_rate_matrix,
                                        t_array = t_array,
                                        return_cond = return_cond ) #(T, C_tr, C_s, K, A, A)
        
        intermeds_dict = {}
        return logprobs, intermeds_dict
        
# alias
F81LogprobsFromFile = F81Logprobs

