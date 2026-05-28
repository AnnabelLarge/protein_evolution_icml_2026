#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 25 15:33:52 2025

"""
import pickle
import numpy as np

from flax import linen as nn
import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm
from jax.scipy.special import logsumexp

from utils.BaseClasses import ModuleBase
from latent_class_mixtures.emission_models import (EqulDistLogprobsFromCounts,
                                                              EqulDistLogprobsPerClass,
                                                              EqulDistLogprobsFromFile,
                                                              GTRLogprobs,
                                                              GTRLogprobsFromFile,
                                                              RateMultipliersPerClass,
                                                              IndpRateMultipliers,
                                                              RateMultipliersPerClassFromFile,
                                                              IndpRateMultipliersFromFile,
                                                              F81Logprobs,
                                                              F81LogprobsFromFile)

from latent_class_mixtures.transition_models import (TKF92TransitionLogprobs,
                                                                TKF92TransitionLogprobsFromFile,
                                                                TKF91DomainTransitionLogprobs,
                                                                TKF91DomainTransitionLogprobsFromFile)
from latent_class_mixtures.model_functions import (bound_sigmoid,
                                                              write_matrix_to_npy,
                                                              maybe_write_matrix_to_ascii,
                                                              logsumexp_with_arr_lst,
                                                              log_one_minus_x,
                                                              logspace_marginalize_inf_transits,
                                                              log_matmul,
                                                              get_cond_transition_logprobs)
from latent_class_mixtures.FragAndSiteClasses import FragAndSiteClasses


class NestedTKF(FragAndSiteClasses):
    """
    A nesting of TKF fragment models
    
    
    C_dom: number of mixtures associated with nested TKF92 models (domain-level)
    C_frag: number of mixtures associated with TKF92 fragments (fragment-level)
    C_trans: C_dom * C_frag = C_frag
    B = batch size; number of samples
    T = number of branch lengths; this could be: 
        > an array of times for all samples (T; marginalize over these later)
        > an array of time per sample (T=B)
        > a quantized array of times per sample (T = T', where T' <= T)
    S: number of transition states (4 here: M, I, D, start/end)
    A: emission alphebet size (20 for proteins)
    
    
    Initialize with
    ----------------
    config : dict
        config['num_domain_mixtures'] : int
            number of larger TKF92 domain mixtures
            (one domain type here)
    
        config['num_fragment_mixtures'] : int
            number of TKF92 fragment mixtures 
        
        config['num_site_mixtures'] : int
            number of mixtures associated with the EMISSIONS
        
        config['subst_model_type'] : {gtr, f81}
            which substitution model
        
        config['exponential_dist_param'] : int, optional
            There is an exponential prior over time; this provides the
            parameter for this during marginalization over times
            Default is 1
        
        config['times_from'] : {geometric, t_array_from_file, t_per_sample}
        
    name : str
        class name, for flax
    
    Main methods here
    -----------------
    setup
    
    write_params
        write parameters to files
    
    
    Other methods
    --------------
    _retrieve_both_indel_matrices
    
    _get_joint_domain_transit_matrix_without_null_cycles
    
    _get_marginal_domain_transit_matrix_without_null_cycles
    
    _retrieve_joint_transition_entries
    
    _build_joint_nested_tkf_matrix
    
    _build_marginal_nested_tkf_matrix
    
    _get_transition_scoring_matrices
    
    _get_scoring_matrices
        
        
    Inherited from FragAndSiteClasses
    -----------------------------------
    __call__
        unpack batch and calculate logP(anc, desc, align)
    
    calculate_all_loglikes
        calculate logP(anc, desc, align), logP(anc), logP(desc), and
        logP(desc, align | anc)
    
    _get_emission_scoring_matrices
    
    
    Methods inherited from neural_models.model_utils.BaseClasses.ModuleBase
    -----------------------------------------------------------------
    sow_flax_intermeds
        for tensorboard logging
    """
    config: dict
    name: str
    
    def setup(self):
        ###################
        ### read config   #
        ###################
        # required
        self.num_domain_mixtures = self.config['num_domain_mixtures']
        self.num_fragment_mixtures = self.config['num_fragment_mixtures']
        self.num_site_mixtures = self.config['num_site_mixtures']
        self.num_rate_mults = self.config['k_rate_mults']
        self.num_transit_mixtures = ( self.num_domain_mixtures * 
                                      self.num_fragment_mixtures ) # C_tr

        self.subst_model_type = self.config['subst_model_type']
        self.times_from = self.config['times_from']
        
        # optional or hard coded
        self.indp_rate_mults = False
        self.norm_reported_loss_by = 'desc_len'
        self.exponential_dist_param = self.config.get('exponential_dist_param', 1)
        self.gap_idx = self.config.get('gap_idx', 43)
        
        
        ###################################################################
        ### module for transition probabilities and class probabilities   #
        ###################################################################
        self.domain_transitions_module = TKF91DomainTransitionLogprobs(config = self.config,
                                                                       name = f'tkf91 domain indel model')
        
        self.fragment_transitions_module = TKF92TransitionLogprobs(config = self.config,
                                                                   name = f'tkf92 frag indel model')
        
        
        ###############################################################
        ### probability of having a particular subsitution rate       #
        ### rate multiplier, and the rate multipliers themselves      #
        ###############################################################
        if not self.indp_rate_mults:
            self.rate_mult_module = RateMultipliersPerClass(config = self.config,
                                                      name = f'get rate multipliers')
        
        elif self.indp_rate_mults:
            self.rate_mult_module = IndpRateMultipliers(config = self.config,
                                                      name = f'get rate multipliers')
        
        
        ###############################################################
        ### module for equilibrium distribution, and the site-level   # 
        ### mixture weights P(c_sites | c_frag)                       #
        ###############################################################
        if (self.num_transit_mixtures * self.num_site_mixtures) == 1:
            self.equl_dist_module = EqulDistLogprobsFromCounts(config = self.config,
                                                       name = f'get equilibrium')
        elif (self.num_transit_mixtures * self.num_site_mixtures) > 1:
            self.equl_dist_module = EqulDistLogprobsPerClass(config = self.config,
                                                     name = f'get equilibrium')
        
        
        ###########################################
        ### module for substitution rate matrix   #
        ###########################################
        if self.subst_model_type == 'gtr':
            self.logprob_subst_module = GTRLogprobs( config = self.config,
                                                  name = f'gtr subst. model' )
            
        elif self.subst_model_type == 'f81':
            self.logprob_subst_module = F81Logprobs( config = self.config,
                                                     name = f'f81 subst. model' )

    def _retrieve_both_indel_matrices(self,
                                      t_array,
                                      sow_flax_intermeds: bool):
        """
        T: time
        C_dom: domains
        C_frag: fragments
        S: normal states (4: Match, Ins, Del, Start/End)
        
        get both the fragment-level and domain-level transition matrices/params
        
        
        Returns: dict
        --------------
        out['log_frag_class_probs'] : ArrayLike, (C_dom, C_frag)
        out['frag_tkf_params_dict'] :  dict
        out['frag_joint_transit_mat'] :  ArrayLike, (T, C_dom, C_frag_to, C_frag_from, S_from, S_to)
        
        out['lam_frag'] :  ArrayLike, (C_dom,)
        out['mu_frag'] :  ArrayLike, (C_dom,)
        out['offset_frag'] :  ArrayLike, (C_dom,)
        out['r_frag'] :  ArrayLike, (C_dom, C_frag)
        
        out['log_domain_class_probs'] :  ArrayLike, (C_dom,)
        out['dom_joint_transit_mat'] :  ArrayLike, (T, S_from, S_to)
        out['frag_marginal_transit_mat'] :  ArrayLike, (C_dom, C_frag_to, C_frag_from, 2, 2)
        
        out['lam_dom'] :  float 
        out['mu_dom'] :  float 
        out['offset_dom'] :  float
            
        """
        ### fragment
        out = self.fragment_transitions_module( t_array = t_array,
                                       return_all_matrices = True,
                                       sow_flax_intermeds = sow_flax_intermeds )
        log_frag_class_probs = out[0]
        matrix_dict = out[1]
        frag_tkf_params_dict = out[2]
        del out

        # unpack
        frag_joint_transit_mat = matrix_dict['joint'] #(T, C_dom, C_frag_from, C_frag_to, S_from, S_to)
        frag_marginal_transit_mat = matrix_dict['marginal'] #(C_dom, C_frag_from, C_frag_to, 2, 2)
        lam_frag = matrix_dict['lam'] #(C_dom,)
        mu_frag = matrix_dict['mu'] #(C_dom,)
        offset_frag = matrix_dict['offset'] #(C_dom,)
        r_frag = matrix_dict['r_extend'] #(C_dom, C_frag)
        del matrix_dict
        
        
        ### domain
        out = self.domain_transitions_module( t_array = t_array,
                                       return_all_matrices = True,
                                       sow_flax_intermeds = sow_flax_intermeds )
        
        log_domain_class_probs = out[0]
        matrix_dict = out[1]
        del out
        
        # unpack
        dom_joint_transit_mat = matrix_dict['joint'] #(T, S_from, S_to)
        dom_marginal_transit_mat = matrix_dict['marginal'] #(2, 2)
        lam_dom = matrix_dict['lam'] #float
        mu_dom = matrix_dict['mu'] #float
        offset_dom = matrix_dict['offset'] #float
        
        # return all of these
        out = {'log_frag_class_probs': log_frag_class_probs,
               'frag_tkf_params_dict': frag_tkf_params_dict,
               'frag_joint_transit_mat': frag_joint_transit_mat,
               'frag_marginal_transit_mat': frag_marginal_transit_mat,
               'lam_frag': lam_frag,
               'mu_frag': mu_frag,
               'offset_frag': offset_frag,
               'r_frag': r_frag,
               'log_domain_class_probs': log_domain_class_probs,
               'dom_joint_transit_mat': dom_joint_transit_mat,
               'dom_marginal_transit_mat': dom_marginal_transit_mat,
               'lam_dom': lam_dom,
               'mu_dom': mu_dom,
               'offset_dom': offset_dom}
        return out
    
    
    def _get_joint_domain_transit_matrix_without_null_cycles(self,
                                                             log_domain_class_probs,
                                                             frag_tkf_params_dict,
                                                             dom_joint_transit_mat ):
        """
        T: time
        C_dom: domains
        S: normal states (4: Match, Ins, Del, Start/End)
        
        with the top-level domain model, eliminate null cycles to yield final
          T_{MIDS,MIDE} transition matrix used in final joint transition matrix
         
        
        Arguments:
        ----------
        log_domain_class_probs: ArrayLike, (C_dom,)
        frag_tkf_params_dict: dict
        dom_joint_transit_mat: ArrayLike, (T, S_from=4, S_to=4)
        
        Returns:
        ---------
        log_T_mat: ArrayLike, (T, S_from=4, S_to=4)
        
        """
        T = dom_joint_transit_mat.shape[0]
        S = dom_joint_transit_mat.shape[-1]
        
        ### helper values
        log_z_t = logsumexp( (log_domain_class_probs[None,:] +
                              frag_tkf_params_dict['log_offset'][None,:] +
                              frag_tkf_params_dict['log_one_minus_beta']), axis=-1) #(T,)

        log_z_0 = logsumexp( log_domain_class_probs + frag_tkf_params_dict['log_offset'], axis=-1) #float

        log_one_minus_z_t = log_one_minus_x( log_z_t ) #(T,)
        log_one_minus_z_0 = log_one_minus_x( log_z_0 ) #float
 
    
        ### create T_mat_{MIDS, MIDE} to modify later
        # multiply any -> M by (1 - z_t)
        mask = jnp.concatenate( [jnp.ones(  (T, S, 1), dtype = bool),
                                 jnp.zeros( (T, S, 3), dtype=bool )], axis=2 )
        log_T_mat = jnp.where(mask, 
                              dom_joint_transit_mat + log_one_minus_z_t[:,None,None], 
                              dom_joint_transit_mat) #(T, S_from, S_to)
        del mask
        
        # multiply any ->ID by (1 - z_0)
        mask = jnp.concatenate( [jnp.zeros( (T, S, 1), dtype = bool),
                                 jnp.ones(  (T, S, 2), dtype = bool),
                                 jnp.zeros( (T, S, 1), dtype = bool)], axis=2 )
        log_T_mat = jnp.where(mask, 
                              log_T_mat + log_one_minus_z_0, 
                              log_T_mat) #(T, S_from, S_to)
        del mask
        

        ### get U_{MIDS, AB}
        #   M: 0
        #   I: 1
        #   D: 2
        # S/E: 3
        
        # U_{any,A} = z_t \tau_{any, M} + z_0 \tau_{any, I}
        log_u_mids_a_pt1 = dom_joint_transit_mat[..., 0] + log_z_t[:, None] #(T, 4)
        log_u_mids_a_pt2 = dom_joint_transit_mat[..., 1] + log_z_0 #(T, 4)
        log_u_mids_a = jnp.logaddexp( log_u_mids_a_pt1, log_u_mids_a_pt2 ) #(T, 4)
        del log_u_mids_a_pt1, log_u_mids_a_pt2
        
        # U_{MIDS, D} = z_0 \tau_{MIDS,D}
        log_u_mids_b = dom_joint_transit_mat[..., 2] + log_z_0 #(T, 4)

        # final mat
        log_u_mids_ab = jnp.stack([log_u_mids_a, log_u_mids_b], axis=2) #(T, 4, 2)
        del log_u_mids_a, log_u_mids_b


        ### get U_{AB, MIDS}, U_{AB, AB} from already-created log_T_mat
        log_u_ab_mide = log_T_mat[:, [0, 2], :] #(T, 2, 4)
        log_u_ab_ab = log_u_mids_ab[:, [0, 2], :] #(T, 2, 2)
        
        
        ### T_{MIDS, MIDE} = U_{MIDS, MIDE} + U_{MIDS,AB} * (I-U_{AB,AB})^-1 * U_{AB,MIDE}
        # modifying matrix: U_{MIDS,AB} * (I-U_{AB,AB})^-1 * U_{AB,MIDE}
        log_inv_arg = logspace_marginalize_inf_transits( log_u_ab_ab ) #(T, 2, 2)
        mod_first_half = log_matmul( log_A = log_u_mids_ab,
                                     log_B = log_inv_arg ) #(T, 4, 2)
        modifier = log_matmul( log_A = mod_first_half, 
                               log_B = log_u_ab_mide ) #(T, 4, 4)
        log_T_mat = jnp.logaddexp( log_T_mat, modifier ) #(T, S_from, S_to)
        
        return log_T_mat #(T, S_from, S_to)
    
    
    def _get_marginal_domain_transit_matrix_without_null_cycles(self,
                                                                log_domain_class_probs,
                                                                frag_tkf_params_dict,
                                                                dom_marginal_transit_mat):
        """
        C_dom: domains
        S: normal states (4: Match, Ins, Del, Start/End)
        
        with the top-level domain model, eliminate null cycles to yield final
          T_{MIDS,MIDE} transition matrix used in final joint transition matrix
         
        
        Arguments:
        ----------
        log_domain_class_probs: ArrayLike, (C_dom,)
        frag_tkf_params_dict: dict
        dom_marginal_transit_mat: ArrayLike, (2, 2)
        
        Returns:
        ---------
        log_T_mat: ArrayLike, (2, 2)
        
        """
        S = 2
        
        ### create T_mat_{MS, ME} to modify later
        # helpers
        log_z_0 = logsumexp( log_domain_class_probs + frag_tkf_params_dict['log_offset'], axis=-1) #float
        log_one_minus_z_0 = log_one_minus_x( log_z_0 ) #float

        # multiply any -> M by (1 - z_t)
        log_T_mat = dom_marginal_transit_mat.at[:, 0].add( log_one_minus_z_0 ) #(2, 2)
        
        
        ### get modifying matrix
        # log( 1 / 1 - z_0 * \tau_{M,M} ) = -log( 1 - z_0 * \tau_{M,M} )
        inv_arg = - log_one_minus_x( log_z_0 + dom_marginal_transit_mat[0,0] ) #float
        
        # U_{MS,C} = [U_{M,C}, U_{S,C}]^T
        #   = [z_0*\tau_{M,M}, z_0*\tau_{S,M}]^T
        log_u_ms_c = jnp.array( [log_z_0 + dom_marginal_transit_mat[0,0],
                                 log_z_0 + dom_marginal_transit_mat[1,0]] )[:,None] #(2, 1)
        
        # U_{C,ME} = [U_{C,M}, U_{C,E}]
        #   = [(1-z_0)*\tau_{M,M}, \tau_{M,E}]
        log_u_c_me =  jnp.array( [log_one_minus_z_0 + dom_marginal_transit_mat[0,0],
                                  dom_marginal_transit_mat[0,1]] )[None,:] #(1,2)
        
        modifier = log_u_ms_c + log_u_c_me + inv_arg #(2,2)
        
        
        ### add and return
        log_T_mat = jnp.logaddexp( log_T_mat, modifier ) #(2,2)
        return log_T_mat
        
        
    def _retrieve_joint_transition_entries(self,
                                           log_domain_class_probs,
                                           log_frag_class_probs,
                                           frag_tkf_params_dict,
                                           frag_joint_transit_mat,
                                           frag_marginal_transit_mat,
                                           r_frag,
                                           log_T_mat):
        """
        all the tansitions needed for the final joint transition matrix
        
        T: time
        C_dom: domains
        C_frag: fragments
        S_from: 4 for match, ins, del, start/end
        
        Arguments:
        ----------
        log_domain_class_probs : ArrayLike, (C_dom, )
        log_frag_class_probs : ArrayLike, (C_dom, C_frag)
        frag_tkf_params_dict : dict
        frag_joint_transit_mat : ArrayLike, (T, C_dom, C_frag_from, C_frag_to, S_from, S_to)
        frag_marginal_transit_mat : ArrayLike, (C_dom, C_frag_from, C_frag_to, 2, 2)
        log_T_mat : ArrayLike (T, S_from, S_to)
        
        r_frag : ArrayLike, (C_dom, C_frag)
            THIS VALUE IN PROB SPACE!!!
        
        
        Returns: dict
        --------------
        out['mx_to_my'] : ArrayLike, (T, C_dom_from, C_frag_from, C_dom_to, C_frag_to, (S_from \in MID), (S_to \in MID) )
        out['mx_to_ii'] : ArrayLike, (T, C_dom_from, C_frag_from, C_dom_to, C_frag_to, (S_from \in MID) )
        out['mx_to_dd'] : ArrayLike, (T, C_dom_from, C_frag_from, C_dom_to, C_frag_to, (S_from \in MID) )
        out['mx_to_ee'] : ArrayLike, (T, C_dom_from, C_frag_from, (S_from \in MID) )
               
        out['ii_to_my'] : ArrayLike, (T, C_dom_from, C_frag_from, C_dom_to, C_frag_to, (S_to \in MID) )
        out['ii_to_ii'] : ArrayLike, (T, C_dom_from, C_frag_from, C_dom_to, C_frag_to)
        out['ii_to_dd'] : ArrayLike, (T, C_dom_from, C_frag_from, C_dom_to, C_frag_to)
        out['ii_to_ee'] : ArrayLike, (T, C_dom_from, C_frag_from)
               
        out['dd_to_my'] : ArrayLike, (T, C_dom_from, C_frag_from, C_dom_to, C_frag_to, (S_to \in MID) )
        out['dd_to_ii'] : ArrayLike, (T, C_dom_from, C_frag_from, C_dom_to, C_frag_to)
        out['dd_to_dd'] : ArrayLike, (T, C_dom_from, C_frag_from, C_dom_to, C_frag_to)
        out['dd_to_ee'] : ArrayLike, (T, C_dom_from, C_frag_from)
               
        out['ss_to_my'] : ArrayLike, (T, C_dom_to, C_frag_to, (S_to \in MID) )
        out['ss_to_ii'] : ArrayLike, (T, C_dom_to, C_frag_to)
        out['ss_to_dd'] : ArrayLike, (T, C_dom_to, C_frag_to)
        """
        mask_indels = (self.num_domain_mixtures == 1)
        
        
        ##############################################
        ### Precompute some values that get reused   #
        ##############################################
        # frag_joint_transit_mat: (T, C_dom, C_frag_from, C_frag_to, S_from, S_to)
        # log_domain_class_probs: (C_dom,)
        # log_frag_class_probs: (C_dom, C_frag)
        # log_T_mat = (T, S_from, S_to)

        ### v_m * (lam_m / mu_m) * w_{mg}; used to open single-sequence fragments
        start_single_seq_frag_g = ( log_domain_class_probs[:,None] +
                                    frag_tkf_params_dict['log_one_minus_offset'][:,None] + 
                                    log_frag_class_probs ) #(C_dom_to, C_frag_to)
        

        ### v_m * \tau_{SY}^(m) * w_{mg}; used to open pair-aligned fragments
        log_domain_class_probs_reshaped = log_domain_class_probs[None,:,None,None] #(1, C_dom_to, 1, 1)

        # for every C_frag_from -> C_frag_to, the S -> any transition row is the same
        # (since "start" has no class label); so just index the first instance here
        # w_{mg} is already included in frag_joint_transit_mat
        #
        # frag_joint_transit_mat indexing:
        # dim0: T; take all
        # dim1: C_dom_to; take all
        # dim2: C_frag_from; take the 0th element
        # dim3: C_frag_to; take all
        # dim4: S_from; take the last element (corresponding to START)
        # dim5: S_to; take element 0, 1, 2 (corresponding to MATCH, INS, DEL)
        frag_joint_transit_mat_reshaped = frag_joint_transit_mat[:, :, 0, :, 3, 0:3] #(T, C_dom_to, C_frag_to, (S_to \in MID) )

        # final mat
        start_pair_frag_g = log_domain_class_probs_reshaped + frag_joint_transit_mat_reshaped #(T, C_dom_to, C_frag_to, (S_to \in MID) )
        del log_domain_class_probs_reshaped, frag_joint_transit_mat_reshaped


        ### (1 - r_f) (1 - (lam_l / mu_l)); used to close single-sequence fragments
        end_single_seq_frag_f = jnp.log1p( -r_frag ) + frag_tkf_params_dict['log_offset'][:,None] #(C_dom_from, C_frag_from)
        

        ### (1 - r_f) \tau_{XE}^{l}; used to close pair-aligned fragments
        # for every C_frag_from -> C_frag_to, the any -> E transition column is the same
        # (since "end" has no class label); so just index the last instance here
        #
        # frag_joint_transit_mat indexing:
        # dim0: T; take all
        # dim1: C_dom_from; take all
        # dim2: C_frag_from; take all
        # dim3: C_frag_to; take the last element
        # dim4: S_from; take element 0, 1, 2 (corresponding to MATCH, INS, DEL) 
        # dim5: S_to; take the last element (corresponding to END)
        end_pair_frag_f = frag_joint_transit_mat[:, :, :, -1, 0:3, 3] #(T, C_dom_from, C_frag_from, (S_from \in MID) )
        

        ############################################
        ### Calculate all the transitions needed   #
        ############################################
        ### MX -> MY
        #   end_pair_frag_f: (T, C_dom_from, C_frag_from,        1,         1, (S_from \in MID),              1 )
        #  log_T_mat[:,0,0]: (T,          1,           1,        1,         1,                1,              1 )
        # start_pair_frag_g: (T,          1,           1, C_dom_to, C_frag_to,                1, (S_to \in MID) )
        #          mx_to_my: (T, C_dom_from, C_frag_from, C_dom_to, C_frag_to, (S_from \in MID), (S_to \in MID) )
        mx_to_my = ( end_pair_frag_f[:, :, :, None, None, :, None] +
                     log_T_mat[:,0,0][:, None, None, None, None, None, None] +
                     start_pair_frag_g[:, None, None, :, :, None, :] ) #(T, C_dom_from, C_frag_from, C_dom_to, C_frag_to, (S_from \in MID), (S_to \in MID) )
        
        # if extending the domain and/or fragment, add probabilities from JOINT transitions of fragment-level mixture model
        idx = jnp.arange(self.num_domain_mixtures)
        
        # note: advanced indexing moves C_dom to first axis position
        prev_values = mx_to_my[:, idx, :, idx, :, :, :] #(C_dom, T, C_frag_from, C_frag_to, (S_from \in MID), (S_to \in MID) )
        to_add = jnp.transpose( frag_joint_transit_mat[:, idx, :, :, 0:3, 0:3], (1, 0, 2, 3, 4, 5) ) #(C_dom, T, C_frag_from, C_frag_to, (S_from \in MID), (S_to \in MID) )
        new_values = jnp.logaddexp( prev_values, to_add )  #(C_dom, T, C_frag_from, C_frag_to, (S_from \in MID), (S_to \in MID) )
        del prev_values, to_add
        
        mx_to_my = mx_to_my.at[:, idx, :, idx, :, :, :].set(new_values) #(T, C_dom_from, C_frag_from, C_dom_to, C_frag_to, (S_from \in MID), (S_to \in MID) )
        del new_values


        ### MX -> II, DD, EE
        #         end_pair_frag_f: (T, C_dom_from, C_frag_from,        1,         1, (S_from \in MID) )
        #        log_T_mat[:,0,1]: (T,          1,           1,        1,         1,                1 )
        # start_single_seq_frag_g: (1,          1,           1, C_dom_to, C_frag_to,                1 )
        #                mx_to_ii: (T, C_dom_from, C_frag_from, C_dom_to, C_frag_to, (S_from \in MID) )
        mx_to_ii = ( end_pair_frag_f[:, :, :, None, None, :] +
                     log_T_mat[:,0,1][:, None, None, None, None, None] +
                     start_single_seq_frag_g[None, None, None, :, :, None] ) #(T, C_dom_from, C_frag_from, C_dom_to, C_frag_to, (S_from \in MID) )

        #         end_pair_frag_f: (T, C_dom_from, C_frag_from,        1,         1, (S_from \in MID) )
        #        log_T_mat[:,0,2]: (T,          1,           1,        1,         1,                1 )
        # start_single_seq_frag_g: (1,          1,           1, C_dom_to, C_frag_to,                1 )
        #                mx_to_dd: (T, C_dom_from, C_frag_from, C_dom_to, C_frag_to, (S_from \in MID) )
        mx_to_dd = ( end_pair_frag_f[:, :, :, None, None, :] +
                     log_T_mat[:,0,2][:, None, None, None, None, None] +
                     start_single_seq_frag_g[None, None, None, :, :, None] ) #(T, C_dom_from, C_frag_from, C_dom_to, C_frag_to, (S_from \in MID) )

        #  end_pair_frag_f: (T, C_dom_from, C_frag_from, (S_from \in MID) )
        # log_T_mat[:,0,3]: (T,          1,           1,                1 )
        #         mx_to_ee: (T, C_dom_from, C_frag_from, (S_from \in MID) )
        mx_to_ee = ( end_pair_frag_f + 
                     log_T_mat[:,0,3][:, None, None, None] ) #(T, C_dom_from, C_frag_from, (S_from \in MID) )


        ### II -> II
        #   end_single_seq_frag_f: (1, C_dom_from, C_frag_from,        1,         1)
        #        log_T_mat[:,1,1]: (T,          1,           1,        1,         1)
        # start_single_seq_frag_g: (1,          1,           1, C_dom_to, C_frag_to)
        #                ii_to_ii: (T, C_dom_from, C_frag_from, C_dom_to, C_frag_to)
        ii_to_ii = ( end_single_seq_frag_f[None, :, :, None, None] +
                     log_T_mat[:,1,1][:, None, None, None, None] +
                     start_single_seq_frag_g[None, None, None, :, :] ) # (T, C_dom_from, C_frag_from, C_dom_to, C_frag_to)

        if not mask_indels:
            # if extending the domain and/or fragment, add probabilities from 
            #   MARGINAL transitions of fragment-level mixture model
            idx = jnp.arange(self.num_domain_mixtures)

            # note: advanced indexing moves C_dom to first axis position
            prev_values = ii_to_ii[:, idx, :, idx, :] #(C_dom, T, C_frag_from, C_frag_to)
            to_add = frag_marginal_transit_mat[idx, :, :, 0, 0][:, None, :, :] #(C_dom, 1, C_frag_from, C_frag_to )
            new_values = jnp.logaddexp( prev_values, to_add )  #(C_dom, T, C_frag_from, C_frag_to )
            del prev_values, to_add
            
            ii_to_ii = ii_to_ii.at[:, idx, :, idx, :].set(new_values) # (T, C_dom_from, C_frag_from, C_dom_to, C_frag_to)
            del new_values


        ### II -> MY, DD, EE
        # end_single_seq_frag_f: (1, C_dom_from, C_frag_from,        1,         1               1 )
        #      log_T_mat[:,1,0]: (T,          1,           1,        1,         1,              1 )
        #     start_pair_frag_g: (T,          1,           1, C_dom_to, C_frag_to, (S_to \in MID) )
        #              ii_to_my: (T, C_dom_from, C_frag_from, C_dom_to, C_frag_to, (S_to \in MID) )
        ii_to_my = ( end_single_seq_frag_f[None, :, :, None, None, None] +
                     log_T_mat[:,1,0][:, None, None, None, None, None] +
                     start_pair_frag_g[:, None, None, :, :, :] )  # (T, C_dom_from, C_frag_from, C_dom_to, C_frag_to, (S_to \in MID) )
        
        #   end_single_seq_frag_f: (1, C_dom_from, C_frag_from,        1,         1)
        #        log_T_mat[:,1,2]: (T,          1,           1,        1,         1)
        # start_single_seq_frag_g: (1,          1,           1, C_dom_to, C_frag_to)
        #                ii_to_dd: (T, C_dom_from, C_frag_from, C_dom_to, C_frag_to)
        ii_to_dd = ( end_single_seq_frag_f[None, :, :, None, None] +
                     log_T_mat[:,1,2][:, None, None, None, None] +
                     start_single_seq_frag_g[None, None, None, :, :] ) #(T, C_dom_from, C_frag_from, C_dom_to, C_frag_to )

        # end_single_seq_frag_f: (1, C_dom_from, C_frag_from)
        #      log_T_mat[:,1,3]: (T,          1,           1)
        #              ii_to_ee: (T, C_dom_from, C_frag_from)
        ii_to_ee = ( end_single_seq_frag_f[None, :, :] +
                     log_T_mat[:,1,3][:,None,None] ) # (T, C_dom_from, C_frag_from)


        ### DD -> DD
        #   end_single_seq_frag_f: (1, C_dom_from, C_frag_from,        1,         1)
        #        log_T_mat[:,2,2]: (T,          1,           1,        1,         1)
        # start_single_seq_frag_g: (1,          1,           1, C_dom_to, C_frag_to)
        #                dd_to_dd: (T, C_dom_from, C_frag_from, C_dom_to, C_frag_to)
        dd_to_dd = ( end_single_seq_frag_f[None, :, :, None, None] +
                     log_T_mat[:,2,2][:, None, None, None, None] +
                     start_single_seq_frag_g[None, None, None, :, :] ) # (T, C_dom_from, C_frag_from, C_dom_to, C_frag_to)

        if not mask_indels:
            # if extending the domain and/or fragment, add probabilities from 
            #   MARGINAL transitions of fragment-level mixture model
            idx = jnp.arange(self.num_domain_mixtures)

            # note: advanced indexing moves C_dom to first axis position
            prev_values = dd_to_dd[:, idx, :, idx, :] #(C_dom, T, C_frag_from, C_frag_to)
            to_add = frag_marginal_transit_mat[idx, :, :, 0, 0][:, None, :, :] #(C_dom, 1, C_frag_from, C_frag_to )
            new_values = jnp.logaddexp( prev_values, to_add )  #(C_dom, T, C_frag_from, C_frag_to )
            del prev_values, to_add
            
            dd_to_dd = dd_to_dd.at[:, idx, :, idx, :].set(new_values) # (T, C_dom_from, C_frag_from, C_dom_to, C_frag_to)
            del new_values


        ### DD -> MY, II, EE
        # end_single_seq_frag_f: (1, C_dom_from, C_frag_from,        1,         1               1 )
        #      log_T_mat[:,2,0]: (T,          1,           1,        1,         1,              1 )
        #     start_pair_frag_g: (T,          1,           1, C_dom_to, C_frag_to, (S_to \in MID) )
        #              dd_to_my: (T, C_dom_from, C_frag_from, C_dom_to, C_frag_to, (S_to \in MID) )
        dd_to_my = ( end_single_seq_frag_f[None, :, :, None, None, None] +
                     log_T_mat[:,2,0][:, None, None, None, None, None] +
                     start_pair_frag_g[:, None, None, :, :, :] )  # (T, C_dom_from, C_frag_from, C_dom_to, C_frag_to, (S_to \in MID) )

        #   end_single_seq_frag_f: (1, C_dom_from, C_frag_from,        1,         1)
        #        log_T_mat[:,2,1]: (T,          1,           1,        1,         1)
        # start_single_seq_frag_g: (1,          1,           1, C_dom_to, C_frag_to)
        #                dd_to_ii: (T, C_dom_from, C_frag_from, C_dom_to, C_frag_to)
        dd_to_ii = ( end_single_seq_frag_f[None, :, :, None, None] +
                     log_T_mat[:,2,1][:, None, None, None, None] +
                     start_single_seq_frag_g[None, None, None, :, :] ) #(T, C_dom_from, C_frag_from, C_dom_to, C_frag_to )

        # end_single_seq_frag_f: (1, C_dom_from, C_frag_from)
        #      log_T_mat[:,2,3]: (T,          1,           1)
        #              dd_to_ee: (T, C_dom_from, C_frag_from)
        dd_to_ee = ( end_single_seq_frag_f[None, ...] +
                     log_T_mat[:,2,3][:,None,None] ) # (T, C_dom_from, C_frag_from)


        ### SS -> MY,II,DD
        # ss -> ee is just log_T_mat[:,3,3]; no other modifications needed

        #  log_T_mat[:,3,0]: (T,        1,         1,              1 )
        # start_pair_frag_g: (T, C_dom_to, C_frag_to, (S_to \in MID) )
        #          ss_to_my: (T, C_dom_to, C_frag_to, (S_to \in MID) )
        ss_to_my = log_T_mat[:,3,0][:,None, None, None] + start_pair_frag_g #(T, C_dom_to, C_frag_to, (S_to \in MID) )
        
        #  log_T_mat[:,3,1]: (T,        1,         1)
        # start_pair_frag_g: (T, C_dom_to, C_frag_to)
        #          ss_to_ii: (T, C_dom_to, C_frag_to)
        ss_to_ii = log_T_mat[:,3,1][:,None, None] + start_single_seq_frag_g[None, :, :] #(T, C_dom_to, C_frag_to)

        #  log_T_mat[:,3,2]: (T,        1,         1)
        # start_pair_frag_g: (T, C_dom_to, C_frag_to)
        #          ss_to_dd: (T, C_dom_to, C_frag_to)
        ss_to_dd = log_T_mat[:,3,2][:,None, None] + start_single_seq_frag_g[None, :, :] #(T, C_dom_to, C_frag_to)

        
        ### pack all of these transitions up
        out = {'mx_to_my': mx_to_my, # (T, C_dom_from, C_frag_from, C_dom_to, C_frag_to, (S_from \in MID), (S_to \in MID) )
               'mx_to_ii': mx_to_ii, # (T, C_dom_from, C_frag_from, C_dom_to, C_frag_to, (S_from \in MID) )
               'mx_to_dd': mx_to_dd, # (T, C_dom_from, C_frag_from, C_dom_to, C_frag_to, (S_from \in MID) )
               'mx_to_ee': mx_to_ee, # (T, C_dom_from, C_frag_from, (S_from \in MID) )
               
               'ii_to_my': ii_to_my, # (T, C_dom_from, C_frag_from, C_dom_to, C_frag_to, (S_to \in MID) )
               'ii_to_ii': ii_to_ii, # (T, C_dom_from, C_frag_from, C_dom_to, C_frag_to)
               'ii_to_dd': ii_to_dd, # (T, C_dom_from, C_frag_from, C_dom_to, C_frag_to)
               'ii_to_ee': ii_to_ee, # (T, C_dom_from, C_frag_from)
               
               'dd_to_my': dd_to_my, # (T, C_dom_from, C_frag_from, C_dom_to, C_frag_to, (S_to \in MID) )
               'dd_to_ii': dd_to_ii, # (T, C_dom_from, C_frag_from, C_dom_to, C_frag_to)
               'dd_to_dd': dd_to_dd, # (T, C_dom_from, C_frag_from, C_dom_to, C_frag_to)
               'dd_to_ee': dd_to_ee, # (T, C_dom_from, C_frag_from)
               
               'ss_to_my': ss_to_my, # (T, C_dom_to, C_frag_to, (S_to \in MID) )
               'ss_to_ii': ss_to_ii, # (T, C_dom_to, C_frag_to)
               'ss_to_dd': ss_to_dd, # (T, C_dom_to, C_frag_to)
               'ss_to_ee': log_T_mat[:,3,3] #(T,)
               }
        
        return out
    
    def _build_joint_nested_tkf_matrix(self,
                                       transitions_dict):
        """
        assemble final matrix
        
        T: time
        C_dom: domains
        C_frag: fragments
        S: valid states- match, ins, del, start/end
        
        Arguments:
        -----------
        transitions_dict : dict
            contain all the elements for the transtion matrix
        
        Returns:
        --------
        transit mat : ArrayLike (T, C_dom*C_frag, C_dom*C_frag, S_from, S_to)
        """
        ### unpack
        mx_to_my = transitions_dict['mx_to_my'] # (T, C_dom_from, C_frag_from, C_dom_to, C_frag_to, (S_from \in MID), (S_to \in MID) )
        mx_to_ii = transitions_dict['mx_to_ii'] # (T, C_dom_from, C_frag_from, C_dom_to, C_frag_to, (S_from \in MID) )
        mx_to_dd = transitions_dict['mx_to_dd'] # (T, C_dom_from, C_frag_from, C_dom_to, C_frag_to, (S_from \in MID) )
        mx_to_ee = transitions_dict['mx_to_ee'] # (T, C_dom_from, C_frag_from, (S_from \in MID) )
        
        ii_to_my = transitions_dict['ii_to_my'] # (T, C_dom_from, C_frag_from, C_dom_to, C_frag_to, (S_to \in MID) )
        ii_to_ii = transitions_dict['ii_to_ii'] # (T, C_dom_from, C_frag_from, C_dom_to, C_frag_to)
        ii_to_dd = transitions_dict['ii_to_dd'] # (T, C_dom_from, C_frag_from, C_dom_to, C_frag_to)
        ii_to_ee = transitions_dict['ii_to_ee'] # (T, C_dom_from, C_frag_from)
        
        dd_to_my = transitions_dict['dd_to_my'] # (T, C_dom_from, C_frag_from, C_dom_to, C_frag_to, (S_to \in MID) )
        dd_to_ii = transitions_dict['dd_to_ii'] # (T, C_dom_from, C_frag_from, C_dom_to, C_frag_to)
        dd_to_dd = transitions_dict['dd_to_dd'] # (T, C_dom_from, C_frag_from, C_dom_to, C_frag_to)
        dd_to_ee = transitions_dict['dd_to_ee'] # (T, C_dom_from, C_frag_from)
        
        ss_to_my = transitions_dict['ss_to_my'] # (T, C_dom_to, C_frag_to, (S_to \in MID) )
        ss_to_ii = transitions_dict['ss_to_ii'] # (T, C_dom_to, C_frag_to)
        ss_to_dd = transitions_dict['ss_to_dd'] # (T, C_dom_to, C_frag_to)
        ss_to_ee = transitions_dict['ss_to_ee'] #(T,)
        
        
        ### build
        # Row 1: Match -> Any
        m_to_m = mx_to_my[..., 0, 0] # (T, C_dom_from, C_frag_from, C_dom_to, C_frag_to)
        m_to_i = jnp.logaddexp( mx_to_my[..., 0, 1], mx_to_ii[..., 0]) # (T, C_dom_from, C_frag_from, C_dom_to, C_frag_to)
        m_to_d = jnp.logaddexp( mx_to_my[..., 0, 2], mx_to_dd[..., 0]) # (T, C_dom_from, C_frag_from, C_dom_to, C_frag_to)
        m_to_e = mx_to_ee[..., 0][:,:,:,None,None] # (T, C_dom_from, C_frag_from, 1, 1)
        m_to_e = jnp.broadcast_to( m_to_e, m_to_m.shape ) # (T, C_dom_from, C_frag_from, C_dom_to, C_frag_to)
        match_to_any = jnp.stack( [m_to_m, m_to_i, m_to_d, m_to_e], axis=-1 ) # (T, C_dom_from, C_frag_from, C_dom_to, C_frag_to, 4)
        del m_to_m, m_to_i, m_to_d, m_to_e

        # Row 2: Ins -> Any
        i_to_m = jnp.logaddexp( mx_to_my[..., 1, 0], ii_to_my[..., 0] ) # (T, C_dom_from, C_frag_from, C_dom_to, C_frag_to)
        i_to_i = logsumexp_with_arr_lst( [mx_to_my[..., 1, 1],
                                          mx_to_ii[..., 1],
                                          ii_to_my[..., 1], 
                                          ii_to_ii] ) # (T, C_dom_from, C_frag_from, C_dom_to, C_frag_to)
        i_to_d = logsumexp_with_arr_lst( [mx_to_my[..., 1, 2],
                                          mx_to_dd[..., 1],
                                          ii_to_my[..., 2], 
                                          ii_to_dd] ) # (T, C_dom_from, C_frag_from, C_dom_to, C_frag_to)
        i_to_e = jnp.logaddexp(mx_to_ee[..., 1], ii_to_ee)[:, :, :, None, None] # (T, C_dom_from, C_frag_from, 1, 1)
        i_to_e = jnp.broadcast_to( i_to_e, i_to_m.shape ) # (T, C_dom_from, C_frag_from, C_dom_to, C_frag_to)
        ins_to_any = jnp.stack( [i_to_m, i_to_i, i_to_d, i_to_e], axis=-1 ) # (T, C_dom_from, C_frag_from, C_dom_to, C_frag_to, 4)
        del i_to_m, i_to_i, i_to_d, i_to_e

        # Row 3: Del -> Any
        d_to_m = jnp.logaddexp( mx_to_my[..., 2, 0], dd_to_my[..., 0] ) # (T, C_dom_from, C_frag_from, C_dom_to, C_frag_to)
        d_to_i = logsumexp_with_arr_lst( [mx_to_my[..., 2, 1],
                                          mx_to_ii[..., 2],
                                          dd_to_my[..., 1], 
                                          dd_to_ii] ) # (T, C_dom_from, C_frag_from, C_dom_to, C_frag_to)
        d_to_d = logsumexp_with_arr_lst( [mx_to_my[..., 2, 2],
                                          mx_to_dd[..., 2],
                                          dd_to_my[..., 2],
                                          dd_to_dd] ) # (T, C_dom_from, C_frag_from, C_dom_to, C_frag_to)
        d_to_e = jnp.logaddexp(mx_to_ee[..., 2], dd_to_ee)[:, :, :, None, None] # (T, C_dom_from, C_frag_from, 1, 1)
        d_to_e = jnp.broadcast_to( d_to_e, d_to_m.shape ) # (T, C_dom_from, C_frag_from, C_dom_to, C_frag_to)
        del_to_any = jnp.stack( [d_to_m, d_to_i, d_to_d, d_to_e], axis=-1 ) # (T, C_dom_from, C_frag_from, C_dom_to, C_frag_to, 4)
        del d_to_m, d_to_i, d_to_d, d_to_e

        # Row 4: Start -> any
        s_to_e = ss_to_ee[:, None, None] #(T, 1, 1)
        s_to_e = jnp.broadcast_to( s_to_e, ss_to_my[..., 0].shape ) # (T, C_dom_to, C_frag_to)
        start_to_any = jnp.stack( [ss_to_my[..., 0],
                                   jnp.logaddexp( ss_to_my[..., 1], ss_to_ii),
                                   jnp.logaddexp( ss_to_my[..., 2], ss_to_dd),
                                   s_to_e ], axis=-1 ) # (T, C_dom_to, C_frag_to, 4)
        start_to_any = start_to_any[:, None, None, :, :, :] # (T, 1, 1, C_dom_to, C_frag_to, 4)
        start_to_any = jnp.broadcast_to(start_to_any, match_to_any.shape)  # (T, C_dom_from, C_frag_from, C_dom_to, C_frag_to, 4)
        del s_to_e

        # the transition matrix in LOG SPACE
        transit_mat = jnp.stack( [match_to_any, ins_to_any, del_to_any, start_to_any], axis=-2 ) # (T, C_dom_from, C_frag_from, C_dom_to, C_frag_to, 4, 4)
        
        # reshape and return
        T = transit_mat.shape[0]
        C_dom = transit_mat.shape[1]
        C_frag = transit_mat.shape[2]
        S = transit_mat.shape[-1]
        transit_mat = jnp.reshape( transit_mat, (T, C_dom*C_frag, C_dom*C_frag, S, S ) ) # (T, C_dom_from*C_frag_from, C_dom_to*C_frag_to, 4, 4)
        
        return transit_mat                         
        
        
    def _build_marginal_nested_tkf_matrix(self,
                                          log_domain_class_probs,
                                          log_frag_class_probs,
                                          frag_tkf_params_dict,
                                          frag_marginal_transit_mat,
                                          r_frag,
                                          log_T_mat):
        """
        the final marginal transition matrix
        
        I call the transitions names like MX and MY, but this should really be
          "emit-emit", or something like that; no pair emissions possible here
        
        C_dom: domains
        C_frag: fragments
        
        Arguments:
        ----------
        log_domain_class_probs : ArrayLike, (C_dom, )
        log_frag_class_probs : ArrayLike, (C_dom, C_frag)
        frag_tkf_params_dict : dict
        frag_marginal_transit_mat : ArrayLike, (C_dom, C_frag_from, C_frag_to, 2, 2)
        r_frag : ArrayLike, (C_dom, C_frag)
        log_T_mat : ArrayLike, (2, 2)
        
        Returns:
        ---------
        transit_mat : ArrayLike, (C_dom*C_frag, C_dom*C_frag, 2, 2)
        
        """
        
        ##############################################
        ### Precompute some values that get reused   #
        ##############################################
        # frag_marginal_transit_mat: (C_dom, C_frag_from, C_frag_to, S_from, S_to)
        # log_domain_class_probs: (C_dom,)
        # log_frag_class_probs: (C_dom, C_frag)
        # log_T_mat = (2, 2)

        ### v_m * (lam_m / mu_m) * w_{mg}; used to open single-sequence fragments
        start_single_seq_frag_g = ( log_domain_class_probs[:,None] +
                                    frag_tkf_params_dict['log_one_minus_offset'][:,None] + 
                                    log_frag_class_probs ) #(C_dom_to, C_frag_to)

        ### (1 - r_f) (1 - (lam_l / mu_l)); used to close single-sequence fragments
        end_single_seq_frag_f = jnp.log1p( -r_frag ) + frag_tkf_params_dict['log_offset'][:,None] #(C_dom_from, C_frag_from)
        

        ############################################
        ### Calculate all the transitions needed   #
        ############################################
        ### emit/emit -> emit/emit
        #   end_single_seq_frag_f: (C_dom_from, C_frag_from,        1,         1)
        # start_single_seq_frag_g: (         1,           1, C_dom_to, C_frag_to)
        #                mx_to_my: (C_dom_from, C_frag_from, C_dom_to, C_frag_to)
        mx_to_my = ( end_single_seq_frag_f[:, :, None, None] +
                     log_T_mat[0,0] +
                     start_single_seq_frag_g[None, None, :, :] ) #(C_dom_from, C_frag_from, C_dom_to, C_frag_to )
        
        # if extending the domain and/or fragment, add probabilities from 
        #   MARGINAL transitions of fragment-level mixture model
        idx = jnp.arange(self.num_domain_mixtures)
        
        # note: advanced indexing moves C_dom to first axis position
        prev_values = mx_to_my[idx, :, idx, :] #(C_dom, C_frag_from, C_frag_to)
        to_add = frag_marginal_transit_mat[idx, :, :, 0, 0] #(C_dom, C_frag_from, C_frag_to)
        new_values = jnp.logaddexp( prev_values, to_add )  #(C_dom, C_frag_from, C_frag_to )
        del prev_values, to_add
        
        mx_to_my = mx_to_my.at[idx, :, idx, :].set(new_values) # (C_dom_from, C_frag_from, C_dom_to, C_frag_to)
        del new_values
        
        
        ### All other transitions: emit/emit -> EE, SS -> emit/emit
        # ss_to_ee = is just log_T_mat[1,1]; no other mods needed
        mx_to_ee = (end_single_seq_frag_f + log_T_mat[0,1])[:, :, None, None] #(C_dom_from, C_frag_from, 1, 1)
        mx_to_ee = jnp.broadcast_to(mx_to_ee, mx_to_my.shape) #(C_dom_from, C_frag_from, C_dom_to, C_frag_to )
        ss_to_mx = (log_T_mat[1,0] + start_single_seq_frag_g)[None, None, :, :] #(1, 1, C_dom_to, C_frag_to)
        ss_to_mx = jnp.broadcast_to(ss_to_mx, mx_to_my.shape) #(C_dom_from, C_frag_from, C_dom_to, C_frag_to )
        ss_to_ee = jnp.ones( mx_to_my.shape ) * log_T_mat[1,1]  #(C_dom_from, C_frag_from, C_dom_to, C_frag_to )
        
        
        #####################################
        ### Build final transition matrix   #
        #####################################
        transit_mat = jnp.stack( [ jnp.stack([mx_to_my, mx_to_ee], axis=-1),
                                   jnp.stack([ss_to_mx, ss_to_ee], axis=-1) ],
                                axis=-2 ) #(C_dom_from, C_frag_from, C_dom_to, C_frag_to, 2, 2)
        
        # reshape and return
        C_dom = transit_mat.shape[0]
        C_frag = transit_mat.shape[1]
        S = transit_mat.shape[-1]
        transit_mat = jnp.reshape( transit_mat, (C_dom*C_frag, C_dom*C_frag, S, S ) ) # (C_dom_from*C_frag_from, C_dom_to*C_frag_to, 2, 2)
        
        return transit_mat
    
    
    def _get_transition_scoring_matrices(self,
                                    t_array,
                                    sow_flax_intermeds: bool,
                                    return_all_matrices: bool):
        # get fragment-level and domain-level transition matrices
        out_dict = self._retrieve_both_indel_matrices(t_array = t_array,
                                                      sow_flax_intermeds = sow_flax_intermeds)
        
        
        ### joint prob P(anc, desc, align)
        T = t_array.shape[0]
        C_dom = self.num_domain_mixtures
        S = out_dict['dom_joint_transit_mat'].shape[-1]
        
        if C_dom > 1:
            raw_joint_logT_mat = self._get_joint_domain_transit_matrix_without_null_cycles( log_domain_class_probs = out_dict['log_domain_class_probs'],
                                                                            frag_tkf_params_dict = out_dict['frag_tkf_params_dict'],
                                                                            dom_joint_transit_mat = out_dict['dom_joint_transit_mat'] ) # (T, S_from, S_to)
        
        elif C_dom == 1:
            # WARNING: will this trigger numerical instabilities...?
            raw_joint_logT_mat = jnp.ones( (T, S, S) ) * jnp.finfo(jnp.float32).min # (T, S_from, S_to)
            raw_joint_logT_mat = raw_joint_logT_mat.at[:, 3, 0].set(0.0)
            raw_joint_logT_mat = raw_joint_logT_mat.at[:, 0, 3].set(0.0)
            
            # replace the top-level TKF91 S->E transition with fragment-level TKF92 S->E transition
            new_val = out_dict['frag_joint_transit_mat'][:, 0, 0, 0, 3, 3]
            raw_joint_logT_mat = raw_joint_logT_mat.at[:, 3, 3].set(new_val)
        
        joint_transit_entries = self._retrieve_joint_transition_entries( log_domain_class_probs = out_dict['log_domain_class_probs'],
                                                                        log_frag_class_probs = out_dict['log_frag_class_probs'],
                                                                        frag_tkf_params_dict = out_dict['frag_tkf_params_dict'],
                                                                        frag_joint_transit_mat = out_dict['frag_joint_transit_mat'],
                                                                        frag_marginal_transit_mat = out_dict['frag_marginal_transit_mat'],
                                                                        r_frag = out_dict['r_frag'],
                                                                        log_T_mat = raw_joint_logT_mat )
        
        joint_transit_mat = self._build_joint_nested_tkf_matrix( transitions_dict = joint_transit_entries) # (T, C_dom*C_frag, C_dom*C_frag, S_from, S_to)
        log_domain_class_probs = out_dict['log_domain_class_probs']
        log_frag_class_probs = out_dict['log_frag_class_probs']

        # build output
        all_transit_matrices= {'joint': joint_transit_mat,
                               'lam_frag': out_dict['lam_frag'],
                               'mu_frag': out_dict['mu_frag'],
                               'r_frag': out_dict['r_frag'],
                               'lam_dom': out_dict['lam_dom'],
                               'mu_dom': out_dict['mu_dom'] }
        
        
        ### marginal and conditional (if applicable)
        if return_all_matrices:
            # marginal            
            if C_dom > 1:
                raw_marg_logT_mat = self._get_marginal_domain_transit_matrix_without_null_cycles( log_domain_class_probs = out_dict['log_domain_class_probs'],
                                                                            frag_tkf_params_dict = out_dict['frag_tkf_params_dict'],
                                                                            dom_marginal_transit_mat = out_dict['dom_marginal_transit_mat'] ) #(2, 2)
            
            elif C_dom == 1:
                # WARNING: will this trigger numerical instabilities...?
                raw_marg_logT_mat = jnp.ones( (2, 2) ) * jnp.finfo(jnp.float32).min # (T, S_from, S_to)
                raw_marg_logT_mat = raw_marg_logT_mat.at[1, 0].set(0.0)
                raw_marg_logT_mat = raw_marg_logT_mat.at[0, 1].set(0.0)
                
                # replace the top-level TKF91 S->E transition with fragment-level TKF92 S->E transition
                new_val = out_dict['frag_marginal_transit_mat'][0, 0, 0, 1, 1]
                raw_marg_logT_mat = raw_marg_logT_mat.at[1, 1].set(new_val)
                
            marginal_transit_mat = self._build_marginal_nested_tkf_matrix( log_domain_class_probs = out_dict['log_domain_class_probs'],
                                                                        log_frag_class_probs = out_dict['log_frag_class_probs'],
                                                                        frag_tkf_params_dict = out_dict['frag_tkf_params_dict'],
                                                                        frag_marginal_transit_mat = out_dict['frag_marginal_transit_mat'],
                                                                        r_frag = out_dict['r_frag'],
                                                                        log_T_mat = raw_marg_logT_mat ) # (C_dom*C_frag, C_dom*C_frag, 2, 2)
            
            # conditional
            conditional_transit_mat = get_cond_transition_logprobs( marg_matrix = marginal_transit_mat,
                                                                    joint_matrix = joint_transit_mat ) # (T, C_dom*C_frag, C_dom*C_frag, S_from, S_to)
            
            all_transit_matrices['conditional'] = conditional_transit_mat
            all_transit_matrices['marginal'] = marginal_transit_mat
        
        return ( log_domain_class_probs,
                 log_frag_class_probs,
                 all_transit_matrices )
            
            
    def _get_scoring_matrices( self,
                               t_array,
                               sow_flax_intermeds: bool,
                               return_all_matrices: bool,
                               return_intermeds: bool):
        """
        C_dom: number of mixtures associated with nested TKF92 models (domain-level)
        C_frag: number of mixtures associated with TKF92 fragments (fragment-level)
        C_tr: number of transition mixtures, C_dom * C_frag = C_tr
        
        B = batch size; number of samples
        T = number of branch lengths; this could be: 
            > an array of times for all samples (T; marginalize over these later)
            > an array of time per sample (T=B)
            > a quantized array of times per sample (T = T', where T' <= T)
        A: emission alphabet size (20 for proteins)
        S: number of transition states (4 here: M, I, D, start/end)
           
        
        Arguments
        ----------
        t_array : ArrayLike, (T,)
            branch lengths, times for marginalizing over
        
        return_all_matrices : bool
            if false, only return the joint (used for model training)
            if true, return joint, conditional, and marginal matrices
        
        return_intermeds : bool
            return other intermediates
        
        sow_flax_intermeds : bool
            switch for tensorboard logging
          
        Returns
        -------
        out_dict : dict
        
            always returns:
                out_dict['logprob_emit_at_indel'] : (C_trans, A)
                out_dict['joint_logprob_emit_at_match'] : (T, C_trans, A, A)
                out_dict['all_transit_matrices'] : dict
            
            if return_all_matrices:
                out_dict['cond_logprob_emit_at_match'] : (T, C_trans, A, A)
            
            if return_intermeds:
                out_dict['log_equl_dist_per_mixture'] : (C_trans, C_sites, A)
                out_dict['rate_multipliers'] : (C_trans, C_sites, C_subrate)
                out_dict['rate_matrix'] : (C_trans, C_sites, C_subrate)
                out_dict['exchangeabilities'] : (A, A)
                out_dict['cond_subst_logprobs_per_mixture'] : (T, C_trans, C_sites, C_subrate, A, A)
                out_dict['joint_subst_logprobs_per_mixture'] : (T, C_trans, C_sites, C_subrate, A, A)
                out_dict['log_domain_class_probs'] : (C_dom,)
                out_dict['log_fragment_class_probs'] : (C_dom, C_frag)
                out_dict['log_site_class_probs'] : (C_trans, C_sites)
                out_dict['log_rate_mult_probs'] : (C_trans, C_sites, C_subrate)
            
        """
        ######################################
        ### scoring matrix for TRANSITIONS   #
        ######################################
        out = self._get_transition_scoring_matrices( t_array = t_array,
                                                return_all_matrices = return_all_matrices,
                                                sow_flax_intermeds = sow_flax_intermeds ) 
        
        # P(c_domain)
        log_domain_class_probs = out[0] #(C_dom,)
         
        # P(c_fragment | c_domain)
        log_frag_class_probs = out[1] #(C_dom, C_frag)
        
        # all_transit_matrices['joint']: (T, C_dom*C_frag, C_dom*C_frag, S_from, S_to)
        # 
        # if return_all_matrices is True, also include:
        # all_transit_matrices['conditional']: (T, C_dom*C_frag, C_dom*C_frag, S_from, S_to)
        # all_transit_matrices['marginal']: (C_dom*C_frag, C_dom*C_frag, 2, 2)
        all_transit_matrices = out[2]
        
        
        ####################################
        ### scoring matrix for EMISSIONS   #
        ####################################
        log_transit_class_probs = log_domain_class_probs[:, None] + log_frag_class_probs #(C_dom, C_frag)
        log_transit_class_probs = log_transit_class_probs.flatten() #(C_dom*C_frag,)
        
        ### reuse function from FragAndSiteClasses
        # always returns:
        #     out_dict['logprob_emit_at_indel'] : (C_tr, A)
        #     out_dict['joint_logprob_emit_at_match'] : (T, C_tr, A, A)
        
        # if return_all_matrices:
        #     out_dict['cond_logprob_emit_at_match'] : (T, C_tr, A, A)
        
        # if return_intermeds:
        #     out_dict['log_equl_dist_per_mixture'] : (C_tr, C_sites, A)
        #     out_dict['rate_multipliers'] : (C_tr, C_sites, C_subrate)
        #     out_dict['rate_matrix'] : (C_tr, C_sites, C_subrate)
        #     out_dict['exchangeabilities'] : (A, A)
        #     out_dict['cond_subst_logprobs_per_mixture'] : (T, C_tr, C_sites, C_subrate, A, A)
        #     out_dict['joint_subst_logprobs_per_mixture'] : (T, C_tr, C_sites, C_subrate, A, A)
        #     out_dict['log_site_class_probs'] : (C_tr, C_sites)
        #     out_dict['log_rate_mult_probs'] : (C_tr, C_sites, C_subrate)
        out_dict = self._get_emission_scoring_matrices( log_transit_class_probs = log_transit_class_probs,
                                                        t_array = t_array,
                                                        sow_flax_intermeds = sow_flax_intermeds,
                                                        return_all_matrices = return_all_matrices, 
                                                        return_intermeds = return_intermeds)
        
        
        ##################################
        ### add to out_dict and return   #
        ##################################
        out_dict['all_transit_matrices'] = all_transit_matrices
        
        if return_intermeds:
            out_dict['log_domain_class_probs'] = log_domain_class_probs #(C_frag,)
            out_dict['log_frag_class_probs'] = log_frag_class_probs #(C_dom, C_frag)
        
        return out_dict
    
    
    def write_params(self,
                     t_array,
                     out_folder: str,
                     prefix: str,
                     write_time_static_objs: bool):
        #########################################################
        ### only write once: activations_times_used text file   #
        #########################################################
        if write_time_static_objs:
            with open(f'{out_folder}/activations_and_times_used.tsv','w') as g:
                if self.times_from in ['geometric','t_array_from_file']:
                    g.write(f't_array for all samples; possible marginalized over them\n')
                    g.write(f'{t_array}')
                    g.write('\n')
                
                elif self.times_from == 't_per_sample':
                    g.write(f'one branch length for each sample; times used for {prefix}:\n')
                    g.write(f'{t_array}')
                    g.write('\n')
        
        ###################################
        ### always write: Full matrices   #
        ###################################
        out = self._get_scoring_matrices(t_array=t_array,
                                        sow_flax_intermeds=False,
                                        return_all_matrices=True,
                                        return_intermeds=True)
        
        # final conditional and joint prob of match (before and after LSE over site/rate mixtures)
        for key in ['joint_logprob_emit_at_match',
                    'cond_subst_logprobs_per_mixture',
                    'joint_subst_logprobs_per_mixture']:
            mat = np.exp ( out[key] )
            new_key = f'{prefix}_{key}'.replace('log','')
            write_matrix_to_npy( out_folder, mat, new_key )
            maybe_write_matrix_to_ascii( out_folder, mat, new_key )
            del mat, new_key
            
        # transition matrix
        for loss_type in ['joint','conditional','marginal']:
            mat = np.exp(out['all_transit_matrices'][loss_type]) 
            new_key = f'{prefix}_{loss_type}_prob_transit_matrix'
            write_matrix_to_npy( out_folder, mat, new_key )
            maybe_write_matrix_to_ascii( out_folder, mat, new_key )
            del mat, new_key
        
        
        #####################################################################
        ### only write once: parameters, things that don't depend on time   #
        #####################################################################
        if write_time_static_objs:
            ### class probs
            # domain class probs; P(c_dom)
            if self.num_domain_mixtures > 1:
                log_domain_class_probs = out['log_domain_class_probs'] #(C_dom)
                mat = np.exp( log_domain_class_probs )
                key = f'{prefix}_domain_class_probs'
                write_matrix_to_npy( out_folder, mat, key )
                maybe_write_matrix_to_ascii( out_folder, mat, key )
                del key, mat, log_domain_class_probs
                
            # fragment class probs; P(c_frag | c_dom)
            if self.num_fragment_mixtures > 1:
                log_frag_class_probs = out['log_frag_class_probs'] #(C_dom, C_frag,)
                mat = np.exp( log_frag_class_probs )
                key = f'{prefix}_frag_class_probs'
                write_matrix_to_npy( out_folder, mat, key )
                maybe_write_matrix_to_ascii( out_folder, mat, key )
                del key, mat, log_frag_class_probs
            
            # site class probs; P( C_site | C_transit )
            if self.num_site_mixtures > 1:
                site_class_probs = np.exp(out['log_site_class_probs']) #(C_tr, C_sites)
                key = f'{prefix}_site_class_probs'
                write_matrix_to_npy( out_folder, site_class_probs, key )
                maybe_write_matrix_to_ascii( out_folder, site_class_probs, key )
                del key, site_class_probs
            
            # probability of substitution rate multiplier;
            # P( C_subrate | C_site, C_transit )
            if self.num_rate_mults > 1:
                rate_mult_probs = np.exp(out['log_rate_mult_probs']) #(C_tr, C_sites, C_subrate)
                key = f'{prefix}_rate_mult_probs'
                write_matrix_to_npy( out_folder, rate_mult_probs, key )
                maybe_write_matrix_to_ascii( out_folder, rate_mult_probs, key )
                del key
            
            
            ### substitution rate matrix
            rate_matrix = out['rate_matrix'] #(C_tr, C_sites, A, A) or None
            if rate_matrix is not None:
                key = f'{prefix}_rate_matrix'
                write_matrix_to_npy( out_folder, rate_matrix, key )
                del key

                for c_tr in range(rate_matrix.shape[0]):
                    for c_s in range(rate_matrix.shape[1]):
                        mat_to_save = rate_matrix[c_tr, c_s, ...]
                        key = f'{prefix}_trans-class-{c_tr}_site-class-{c_s}_rate_matrix'
                        maybe_write_matrix_to_ascii( out_folder, mat_to_save, key )
                        del mat_to_save, key
                        
                        
            ### equilibrium distribution
            # (BEFORE marginalizing over site clases)
            equl_dist = np.exp(out['log_equl_dist_per_mixture']) #(C_tr, C_sites, A)
            key = f'{prefix}_equilibriums-per-site-class'
            write_matrix_to_npy( out_folder, equl_dist, key )
            maybe_write_matrix_to_ascii( out_folder, equl_dist, key )
            del key, equl_dist
            
            # AFTER marginalizing out site and rate mixtures
            mat = np.exp( out['logprob_emit_at_indel'] ) #(C_tr, A)
            new_key = f'{prefix}_logprob_emit_at_indel'.replace('log','')
            write_matrix_to_npy( out_folder, mat, new_key )
            maybe_write_matrix_to_ascii( out_folder, mat, new_key )
            del mat, new_key

            
            ### rate multipliers 
            # \rho_{c,k} or \rho_k
            if not self.rate_mult_module.use_unit_rate_mult:
                rate_multipliers = out['rate_multipliers'] #(C_tr, C_sites, C_subrate)
                key = f'{prefix}_rate_multipliers'
                write_matrix_to_npy( out_folder, rate_multipliers, key )
                maybe_write_matrix_to_ascii( out_folder, rate_multipliers, key )
                del key
                
            
            ### exchangeabilities, if gtr or hky85
            exchangeabilities = out['exchangeabilities'] #(A, A) or None
            
            if self.subst_model_type == 'gtr':
                key = f'{prefix}_gtr-exchangeabilities'
                write_matrix_to_npy( out_folder, exchangeabilities, key )
                maybe_write_matrix_to_ascii( out_folder, exchangeabilities, key )
                del key
                
            elif self.subst_model_type == 'hky85':
                ti = exchangeabilities[0, 2]
                tv = exchangeabilities[0, 1]
                arr = np.array( [ti, tv] )
                key = f'{prefix}_hky85_ti_tv'
                write_matrix_to_npy( out_folder, arr, key )
                
                with open(f'{out_folder}/ASCII_{prefix}_hky85_ti_tv.txt','w') as g:
                    g.write(f'transition rate, ti: {ti}\n')
                    g.write(f'transition rate, tv: {tv}')
                del key, arr
                
            ### extract indel params
            if self.config['load_all']:
                # domain level tkf91
                lam_dom = self.domain_transitions_module.param_dict['lambda']
                mu_dom = self.domain_transitions_module.param_dict['mu']
                offset_dom = 1 - (lam_dom / mu_dom)
                
                # fragment level tkf92
                lam_frag = self.fragment_transitions_module.param_dict['lambda']
                mu_frag = self.fragment_transitions_module.param_dict['mu']
                r_frag = self.fragment_transitions_module.param_dict['r_extend']
                offset_frag = 1 - (lam_frag/mu_frag)
            
            elif not self.config['load_all']:
                # domain level tkf91
                mu_min_val = self.domain_transitions_module.mu_min_val #float
                mu_max_val = self.domain_transitions_module.mu_max_val #float
                offs_min_val = self.domain_transitions_module.offs_min_val #float
                offs_max_val = self.domain_transitions_module.offs_max_val #float
                mu_offset_logits = self.domain_transitions_module.tkf_mu_offset_logits #(2,)
            
                mu_dom = bound_sigmoid(x = mu_offset_logits[0,0],
                                    min_val = mu_min_val,
                                    max_val = mu_max_val).item() #float
                
                offset_dom = bound_sigmoid(x = mu_offset_logits[0,1],
                                          min_val = offs_min_val,
                                          max_val = offs_max_val).item() #float
                lam_dom = mu_dom * (1 - offset_dom)  #float
                
                del mu_min_val, mu_max_val, offs_min_val, offs_max_val
                del mu_offset_logits
                
                # fragment level tkf92
                mu_min_val = self.fragment_transitions_module.mu_min_val #float
                mu_max_val = self.fragment_transitions_module.mu_max_val #float
                offs_min_val = self.fragment_transitions_module.offs_min_val #float
                offs_max_val = self.fragment_transitions_module.offs_max_val #float
                mu_offset_logits = self.fragment_transitions_module.tkf_mu_offset_logits #(C_dom, 2)
                
                mu_frag = bound_sigmoid(x = mu_offset_logits[:,0],
                                    min_val = mu_min_val,
                                    max_val = mu_max_val) #(C_dom,)
                
                offset_frag = bound_sigmoid(x = mu_offset_logits[:,1],
                                          min_val = offs_min_val,
                                          max_val = offs_max_val) #(C_dom,)
                lam_frag = mu_frag * (1 - offset_frag)  #(C_dom,)
                
                r_extend_min_val = self.fragment_transitions_module.r_extend_min_val #float
                r_extend_max_val = self.fragment_transitions_module.r_extend_max_val #float
                r_extend_logits = self.fragment_transitions_module.r_extend_logits #(C_dom, C_frag)
                
                r_frag = bound_sigmoid(x = r_extend_logits,
                                         min_val = r_extend_min_val,
                                         max_val = r_extend_max_val) #(C_dom,C_frag)
                
            mean_indel_lengths = 1 / (1 - r_frag) #(C_dom, C_frag) 

            
            ### write
            # domain level
            with open(f'{out_folder}/ASCII_{prefix}_top_level_tkf91_indel_params.txt','w') as g:
                g.write(f'For TOP-LEVEL mixture of domains; tkf91\n')
                g.write(f'insert rate, lambda: {lam_dom}\n')
                g.write(f'deletion rate, mu: {mu_dom}\n')
                g.write(f'offset: {offset_dom}\n\n')
                
            out_dict = {'lambda': np.array(lam_dom), # shape=()
                        'mu': np.array(mu_dom), # shape=()
                        'offset': np.array(offset_dom)} # shape=()
            
            with open(f'{out_folder}/PARAMS-DICT_{prefix}_top_level_tkf91_indel_params.pkl','wb') as g:
                pickle.dump(out_dict, g)
            del out_dict, lam_dom, mu_dom, offset_dom
            
            # fragment level
            with open(f'{out_folder}/ASCII_{prefix}_fragment_tkf92_indel_params.txt','w') as g:
                g.write(f'For NESTED mixture of fragments; tkf92\n')
                g.write(f'insert rate, lambda:\n{lam_frag}\n\n')
                g.write(f'deletion rate, mu:\n{mu_frag}\n\n')
                g.write(f'offset:\n{offset_frag}\n\n')
                g.write(f'extension prob, r:\n{r_frag}\n\n')
                g.write(f'mean indel length:\n{mean_indel_lengths}\n\n')
                
            out_dict = {'lambda': np.array(lam_frag), # (C_dom,)
                        'mu': np.array(mu_frag), # (C_dom,)
                        'offset': np.array(offset_frag), #(C_dom,)
                        'r_extend': r_frag} # (C_dom, C_frag)
            
            with open(f'{out_folder}/PARAMS-DICT_{prefix}_fragment_tkf92_indel_params.pkl','wb') as g:
                pickle.dump(out_dict, g)



class NestedTKFLoadAll(NestedTKF):
    """
    NestedTKF, but load params from files
    
    
    C_dom: number of mixtures associated with nested TKF92 models (domain-level)
        > here, C_dom = 1
    C_frag: number of mixtures associated with TKF92 fragments (fragment-level)
    C_trans: C_dom * C_frag = C_frag
    B = batch size; number of samples
    T = number of branch lengths; this could be: 
        > an array of times for all samples (T; marginalize over these later)
        > an array of time per sample (T=B)
        > a quantized array of times per sample (T = T', where T' <= T)
    S: number of transition states (4 here: M, I, D, start/end)
    A: emission alphebet size (20 for proteins)
    
    
    Initialize with
    ----------------
    config : dict
        config['num_domain_mixtures'] : int
            number of larger TKF92 domain mixtures
            (one domain type here)
    
        config['num_fragment_mixtures'] : int
            number of TKF92 fragment mixtures 
        
        config['num_site_mixtures'] : int
            number of mixtures associated with the EMISSIONS
        
        config['subst_model_type'] : {gtr, f81}
            which substitution model
        
        config['exponential_dist_param'] : int, optional
            There is an exponential prior over time; this provides the
            parameter for this during marginalization over times
            Default is 1
        
        config['times_from'] : {geometric, t_array_from_file, t_per_sample}
        
    name : str
        class name, for flax
    
    Main methods here
    -----------------
    setup
    
    
    Inherited from NestedTKF
    ------------------------
    
    __call__
        unpack batch and calculate logP(anc, desc, align)
    
    calculate_all_loglikes
        calculate logP(anc, desc, align), logP(anc), logP(desc), and
        logP(desc, align | anc)
    
    write_params
        write parameters to files
        
    _get_scoring_matrices
    
    
    Methods inherited from neural_models.model_utils.BaseClasses.ModuleBase
    -----------------------------------------------------------------
    sow_flax_intermeds
        for tensorboard logging
    """
    config: dict
    name: str
    
    def setup(self):
        ###################
        ### read config   #
        ###################
        # required
        self.num_domain_mixtures = self.config['num_domain_mixtures']
        self.num_fragment_mixtures = self.config['num_fragment_mixtures']
        self.num_site_mixtures = self.config['num_site_mixtures']
        self.num_rate_mults = self.config['k_rate_mults']
        self.num_transit_mixtures = ( self.num_domain_mixtures * 
                                      self.num_fragment_mixtures ) # C_tr

        self.subst_model_type = self.config['subst_model_type']
        self.times_from = self.config['times_from']
        
        # optional or hard coded
        self.indp_rate_mults = False
        self.norm_reported_loss_by = 'desc_len'
        self.exponential_dist_param = self.config.get('exponential_dist_param', 1)
        self.gap_idx = self.config.get('gap_idx', 43)
        
        
        ###################################################################
        ### module for transition probabilities and class probabilities   #
        ###################################################################
        self.domain_transitions_module = TKF91DomainTransitionLogprobsFromFile(config = self.config,
                                                                       name = f'tkf91 domain indel model')
        
        self.fragment_transitions_module = TKF92TransitionLogprobsFromFile(config = self.config,
                                                                   name = f'tkf92 frag indel model')
        
        
        ###############################################################
        ### probability of having a particular subsitution rate       #
        ### rate multiplier, and the rate multipliers themselves      #
        ###############################################################
        if not self.indp_rate_mults:
            self.rate_mult_module = RateMultipliersPerClassFromFile(config = self.config,
                                                      name = f'get rate multipliers')
        
        elif self.indp_rate_mults:
            self.rate_mult_module = IndpRateMultipliersFromFile(config = self.config,
                                                      name = f'get rate multipliers')
        
        
        ###############################################################
        ### module for equilibrium distribution, and the site-level   # 
        ### mixture weights P(c_sites | c_frag)                       #
        ###############################################################
        self.equl_dist_module = EqulDistLogprobsFromFile(config = self.config,
                                                     name = f'get equilibrium')
        
        
        ###########################################
        ### module for substitution rate matrix   #
        ###########################################
        if self.subst_model_type == 'gtr':
            self.logprob_subst_module = GTRLogprobsFromFile( config = self.config,
                                                  name = f'gtr subst. model' )
            
        elif self.subst_model_type == 'f81':
            self.logprob_subst_module = F81LogprobsFromFile( config = self.config,
                                                     name = f'f81 subst. model' )

