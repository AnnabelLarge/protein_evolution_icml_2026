#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 2026


Like IndpSitesOldModels, but uses F81 + freely fitted rate multipliers
instead of LG08-GTR. Use latent_class_mixtures.IndpSites as template for
the emission side.
"""
import numpy as np
import pickle

# jumping jax and leaping flax
from flax import linen as nn
import jax
from jax._src.typing import Array, ArrayLike
import jax.numpy as jnp
from jax.scipy.linalg import expm
from jax.scipy.special import logsumexp

from utils.BaseClasses import ModuleBase
from latent_class_mixtures.emission_models import (EqulDistLogprobsFromCounts,
                                                    F81Logprobs,
                                                    F81LogprobsFromFile,
                                                    IndpRateMultipliers,
                                                    IndpRateMultipliersFromFile)
from older_indel_models.transition_models import ( OtherTransitionLogprobs,
                                                   TKF91TransitionLogprobsOldStyle,
                                                   TKF92TransitionLogprobsOldStyle,
                                                   OtherTransitionLogprobsFromFile,
                                                   TKF91TransitionLogprobsOldStyleFromFile,
                                                   TKF92TransitionLogprobsOldStyleFromFile )
from latent_class_mixtures.model_functions import ( bound_sigmoid,
                                                    safe_log,
                                                    lse_over_match_logprobs_per_mixture,
                                                    lse_over_equl_logprobs_per_mixture,
                                                    cond_prob_from_counts,
                                                    write_matrix_to_npy,
                                                    maybe_write_matrix_to_ascii)



class IndpSitesOldModels(ModuleBase):
    """
    pairHMM that finds conditional loglikelihood of alignments, P(Desc, Align | Anc)

    Like IndpSitesOldModels, but uses F81 + K freely fitted rate multipliers
    instead of LG08-GTR.


    differences from IndpSitesOldModels:
    -------------------------------------
    - substitution model: F81 with IndpRateMultipliers (K rate classes with
      freely fitted weights) instead of LG08-GTR
    - equilibrium distribution taken from training data counts (same as before,
      but now routed through EqulDistLogprobsFromCounts module)

    differences from latent_class_mixtures.IndpSites:
    ---------------------------------------------------
    - TKF91 and TKF92 use the old-style conditional transition matrices
    - Other indel models treat start and end states like match
    - computes conditional logP (not joint)

    B = batch size; number of samples
    T = number of branch lengths; this could be:
        > an array of times for all samples (T; marginalize over these later)
        > an array of time per sample (T=B)
        > a quantized array of times per sample (T = T', where T' <= T)
    S: number of transition states (4 here: M, I, D, start/end)
    A: emission alphabet size (20 for proteins)
    K: number of rate multipliers


    Initialize with
    ----------------
    config : dict
        config['indel_model_type'] : {tkf91, tkf92, None}
            which indel model, if any

        config['times_from'] : {geometric, t_array_from_file, t_per_sample}

        config['t_grid_step'] : int, optional
            exponential prior parameter for marginalization over times

        config['k_rate_mults'] : int
            number of freely fitted rate multipliers

        config['norm_rate_mults'] : bool, optional
            enforce constraint sum_k P(k)*rho_k = 1; default True

        config['rate_mult_range'] : (float, float), optional
            min and max rate multiplier; default (0.01, 10)

        config['norm_rate_matrix'] : bool, optional
            normalize F81 rate matrix; default True

        config['training_dset_emit_counts'] : ArrayLike, (A,)
            observed amino acid counts to build equilibrium distribution

    name : str
        class name, for flax


    Main methods here
    -----------------
    setup

    __call__
        unpack batch and calculate logP(desc, align | anc)

    write_params
        write parameters to files


    Methods inherited from neural_models.model_utils.BaseClasses.ModuleBase
    -----------------------------------------------------------------
    sow_flax_intermeds
        for tensorboard logging
    """
    config: dict
    name: str

    def setup(self):
        # not applicable here
        self.num_fragment_mixtures = 1
        self.num_domain_mixtures = 1
        self.num_transit_mixtures = 1
        self.num_site_mixtures = 1
        self.indp_rate_mults = True
        self.subst_model_type = 'f81'
        self.norm_reported_loss_by = 'desc_len'

        ### read config
        indel_model_type = self.config['indel_model_type']
        self.indel_model_type = indel_model_type.lower() if indel_model_type is not None else None
        self.exponential_dist_param = self.config.get('exponential_dist_param', 1)
        self.times_from = self.config['times_from'].lower()
        self.num_rate_mults = self.config['k_rate_mults']

        ### init emission models
        # equilibrium distribution from training data counts
        self.equl_dist_module = EqulDistLogprobsFromCounts(config = self.config,
                                                           name = 'get equilibrium')

        # rate multipliers (independent of site class; one mixture over K rates)
        self.rate_mult_module = IndpRateMultipliers(config = self.config,
                                                    name = 'get rate multipliers')

        # F81 substitution model
        self.logprob_subst_module = F81Logprobs(config = self.config,
                                                name = 'f81 subst. model')

        ### init indel model
        if self.indel_model_type == 'tkf91':
            self.transitions_module = TKF91TransitionLogprobsOldStyle(config = self.config,
                                                     name = f'tkf91 indel model')

        elif self.indel_model_type == 'tkf92':
            self.transitions_module = TKF92TransitionLogprobsOldStyle(config = self.config,
                                                     name = f'tkf92 indel model')

        else:
            self.transitions_module = OtherTransitionLogprobs(config = self.config,
                                                     name = f'{indel_model_type} indel model')


    def __call__(self,
                 batch: list[ArrayLike],
                 t_array: ArrayLike,
                 sow_flax_intermeds: bool):
        """
        Use this during active model training

        B = batch size; number of samples
        T = number of branch lengths; this could be:
            > an array of times for all samples (T; marginalize over these later)
            > an array of time per sample (T=B)
            > a quantized array of times per sample (T = T', where T' <= T)
        S: number of transition states (4 here: M, I, D, start/end)
        A: emission alphabet size (20 for proteins)


        Returns
        -------
        loss: average across the batch, based on conditional log-likelihood

        aux_dict: has the following keys and values
          1.) 'cond_neg_logP': sum down the length
          2.) 'cond_neg_logP_length_normed': sum down the length,
              normalized by descendant length
        """
        # which times to use for scoring matrices
        if self.times_from == 't_per_sample':
            times_for_matrices = batch[4] #(B,)

        elif self.times_from in ['geometric', 't_array_from_file']:
            times_for_matrices = t_array #(T,)

        # get the scoring matrices needed
        #
        # scoring_matrices_dict has the following keys:
        #   logprob_emit_at_indel: ArrayLike, (A, )
        #   cond_logprob_emit_at_match: ArrayLike, (T, A, A)
        #   all_transit_matrices: dict with keys 'conditional', 'log_corr'
        #   maybe_tkf_params: correction factor for tkf92; otherwise placeholder
        scoring_matrices_dict = self._get_scoring_matrices(t_array = times_for_matrices,
                                        sow_flax_intermeds = sow_flax_intermeds)

        # calculate loglikelihoods
        #
        # out_dict has the following keys:
        # cond_neg_logP: ArrayLike, (B,)
        # cond_neg_logP_length_normed: ArrayLike, (B,)
        out_dict = cond_prob_from_counts( batch = batch,
                                          times_from = self.times_from,
                                          score_indels = True,
                                          scoring_matrices_dict = scoring_matrices_dict,
                                          t_array = t_array,
                                          exponential_dist_param = self.exponential_dist_param,
                                          norm_reported_loss_by = self.norm_reported_loss_by,
                                          return_intermeds = False )

        # add scoring matrices to out_dict
        out_dict = {**out_dict, **scoring_matrices_dict}

        # calculate loss
        loss = jnp.mean( out_dict['cond_neg_logP'] ) #float

        return loss, out_dict


    def _get_scoring_matrices(self,
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
        A: emission alphabet size (20 for proteins)
        S: number of transition states (4 here: M, I, D, start/end)
        K: number of rate multipliers


        Arguments
        ----------
        t_array : ArrayLike, (T,)
            branch lengths, times for marginalizing over

        sow_flax_intermeds : bool
            switch for tensorboard logging

        Returns
        -------
        out_dict : dict
            out_dict['logprob_emit_at_indel'] : (A,)
            out_dict['cond_logprob_emit_at_match'] : (T, A, A) or (B, A, A)
            out_dict['all_transit_matrices'] : dict with keys:
                > 'conditional' : (T, S, S) or (B, S, S)
                > 'log_corr' : (B,)
            out_dict['maybe_tkf_params'] : TKF92 correction or placeholder
        """
        ##############################################################
        ### equilibrium distribution (from training data counts)     #
        ##############################################################
        # log_site_class_probs : (C_tr=1, C_sites=1)
        # log_equl_dist_per_mixture : (C_tr=1, C_sites=1, A)
        out = self.equl_dist_module(sow_flax_intermeds = sow_flax_intermeds)
        log_site_class_probs, log_equl_dist_per_mixture = out
        del out

        # P(x) = sum_c P(c) * P(x|c); with one class this is just P(x|c=0)
        # logprob_emit_at_indel : (C_tr=1, A)
        logprob_emit_at_indel = lse_over_equl_logprobs_per_mixture(
            log_site_class_probs = log_site_class_probs,
            log_equl_dist_per_mixture = log_equl_dist_per_mixture)
        logprob_emit_at_indel = logprob_emit_at_indel[0, ...] #(A,)


        ####################################################
        ### rate multipliers and their mixture weights     #
        ####################################################
        # log_rate_mult_probs : (C_tr=1, C_sites=1, K)
        # rate_multipliers : (C_tr=1, C_sites=1, K)
        log_rate_mult_probs, rate_multipliers = self.rate_mult_module(
            sow_flax_intermeds = sow_flax_intermeds,
            log_site_class_probs = log_site_class_probs,
            log_transit_class_probs = jnp.zeros(1,))


        ########################################################
        ### F81 conditional substitution logprobs per rate     #
        ########################################################
        # cond_subst_logprobs_per_mixture : (T, C_tr=1, C_sites=1, K, A, A)
        out = self.logprob_subst_module(
            log_equl_dist = log_equl_dist_per_mixture,
            rate_multipliers = rate_multipliers,
            t_array = t_array,
            sow_flax_intermeds = sow_flax_intermeds,
            return_cond = True)
        cond_subst_logprobs_per_mixture, _ = out
        del out

        # marginalize over rate classes
        # result : (T, C_tr=1, A, A)
        cond_logprob_emit_at_match = lse_over_match_logprobs_per_mixture(
            log_site_class_probs = log_site_class_probs,
            log_rate_mult_probs = log_rate_mult_probs,
            logprob_emit_at_match_per_mixture = cond_subst_logprobs_per_mixture)
        cond_logprob_emit_at_match = cond_logprob_emit_at_match[:, 0, ...] #(T, A, A)


        ##############################################
        ### indel model transition log-probabilities #
        ##############################################
        out = self.transitions_module(t_array = t_array,
                                      sow_flax_intermeds = sow_flax_intermeds)
        cond_transit_matrix, log_corr, maybe_tkf_params = out
        del out

        all_transit_matrices = {}
        all_transit_matrices['conditional'] = cond_transit_matrix
        all_transit_matrices['log_corr'] = log_corr
        del cond_transit_matrix, log_corr


        ### output
        out_dict = {'logprob_emit_at_indel': logprob_emit_at_indel, #(A,)
                    'cond_logprob_emit_at_match': cond_logprob_emit_at_match, #(T,A,A) or (B,A,A)
                    'all_transit_matrices': all_transit_matrices, #dict
                    'maybe_tkf_params': maybe_tkf_params} # TKF92 correction; otherwise placeholder
        return out_dict


    def write_params(self,
                     t_array,
                     out_folder: str,
                     prefix: str,
                     write_time_static_objs: bool):
        ###################################
        ### always write: Full matrices   #
        ###################################
        out = self._get_scoring_matrices(t_array = t_array,
                                         sow_flax_intermeds = False)

        # final conditional prob of match
        mat = np.exp( out['cond_logprob_emit_at_match'] ) #(T, A, A) or (B, A, A)
        new_key = f'{prefix}_cond_logprob_emit_at_match'.replace('log','')
        write_matrix_to_npy( out_folder, mat, new_key )
        maybe_write_matrix_to_ascii( out_folder, mat, new_key )
        del mat, new_key

        # transition matrix: conditional
        mat = np.exp(out['all_transit_matrices']['conditional'])  #(T, S, S) or (B, S, S)
        key = f'{prefix}_cond_prob_transit_matrix'
        write_matrix_to_npy( out_folder, mat, key )
        maybe_write_matrix_to_ascii( out_folder, mat, key )
        del mat, key


        #####################################################################
        ### only write once: parameters, things that don't depend on time   #
        #####################################################################
        if write_time_static_objs:
            ### equilibrium distribution
            mat = np.exp( out['logprob_emit_at_indel'] ) #(A,)
            new_key = f'{prefix}_logprob_emit_at_indel'.replace('log','')
            write_matrix_to_npy( out_folder, mat, new_key )
            maybe_write_matrix_to_ascii( out_folder, mat, new_key )
            del mat, new_key

            ### rate multipliers and their probabilities
            if self.num_rate_mults > 1:
                # need to call modules to get current values
                out_equl = self.equl_dist_module(sow_flax_intermeds = False)
                log_site_class_probs_w, _ = out_equl
                del out_equl

                log_rate_mult_probs_w, rate_multipliers_w = self.rate_mult_module(
                    sow_flax_intermeds = False,
                    log_site_class_probs = log_site_class_probs_w,
                    log_transit_class_probs = jnp.zeros(1,))

                # rate multipliers: (C_tr=1, C_sites=1, K) -> squeeze to (K,)
                rate_mults_arr = np.array( rate_multipliers_w[0, 0, :] )
                key = f'{prefix}_rate_multipliers'
                write_matrix_to_npy( out_folder, rate_mults_arr, key )
                maybe_write_matrix_to_ascii( out_folder, rate_mults_arr, key )
                del key

                # rate multiplier probabilities: (C_tr=1, C_sites=1, K) -> squeeze to (K,)
                rate_mult_probs_arr = np.exp( np.array( log_rate_mult_probs_w[0, 0, :] ) )
                key = f'{prefix}_rate_mult_probs'
                write_matrix_to_npy( out_folder, rate_mult_probs_arr, key )
                maybe_write_matrix_to_ascii( out_folder, rate_mult_probs_arr, key )
                del key

                del log_site_class_probs_w, log_rate_mult_probs_w, rate_multipliers_w
                del rate_mults_arr, rate_mult_probs_arr


            ### write indel params for TKF models
            if self.indel_model_type in ['tkf91', 'tkf92']:
                # always write lambda and mu
                if self.config['load_all']:
                    lam = self.transitions_module.param_dict['lambda']
                    mu = self.transitions_module.param_dict['mu']
                    offset = 1 - (lam/mu)

                elif not self.config['load_all']:
                    mu_min_val = self.transitions_module.mu_min_val #float
                    mu_max_val = self.transitions_module.mu_max_val #float
                    offs_min_val = self.transitions_module.offs_min_val #float
                    offs_max_val = self.transitions_module.offs_max_val #float
                    mu_offset_logits = self.transitions_module.tkf_mu_offset_logits #(2,)

                    mu = bound_sigmoid(x = mu_offset_logits[0,0],
                                       min_val = mu_min_val,
                                       max_val = mu_max_val).item() #float

                    if self.transitions_module.tie_params:
                        offset = jnp.array( 1e-4 ) #float

                    elif not self.transitions_module.tie_params:
                        offset = bound_sigmoid(x = mu_offset_logits[0,1],
                                                 min_val = offs_min_val,
                                                 max_val = offs_max_val).item() #float

                    lam = mu * (1 - offset)  #(1,)

                with open(f'{out_folder}/ASCII_{prefix}_{self.indel_model_type}_indel_params.txt','w') as g:
                    g.write(f'insert rate, lambda: {lam}\n')
                    g.write(f'deletion rate, mu: {mu}\n')
                    g.write(f'offset: {offset}\n\n')

                out_dict = {'lambda': np.array(lam),
                            'mu': np.array(mu),
                            'offset': np.array(offset)}

                # if tkf92, have extra r_ext param
                if self.indel_model_type == 'tkf92':
                    if self.config['load_all']:
                        r_extend = self.transitions_module.param_dict['r_extend']

                    elif not self.config['load_all']:
                        r_extend_min_val = self.transitions_module.r_extend_min_val
                        r_extend_max_val = self.transitions_module.r_extend_max_val
                        r_extend_logits = self.transitions_module.r_extend_logits #(C_dom=1, C_frag=1)

                        r_extend = bound_sigmoid(x = r_extend_logits,
                                                 min_val = r_extend_min_val,
                                                 max_val = r_extend_max_val) #(C_dom=1, C_frag=1)

                    mean_indel_lengths = 1 / (1 - r_extend) #(C_dom=1, C_frag=1)

                    with open(f'{out_folder}/ASCII_{prefix}_{self.indel_model_type}_indel_params.txt','a') as g:
                        g.write(f'extension prob, r: ')
                        [g.write(f'{elem}\t') for elem in r_extend.flatten()]
                        g.write('\n')
                        g.write(f'mean indel length: ')
                        [g.write(f'{elem}\t') for elem in mean_indel_lengths]
                        g.write('\n')

                    out_dict['r_extend'] = r_extend #(C_dom=1, C_frag=1)

                with open(f'{out_folder}/PARAMS-DICT_{prefix}_{self.indel_model_type}_indel_params.pkl','wb') as g:
                    pickle.dump(out_dict, g)
                del out_dict


            ### write indel params for other models
            elif self.indel_model_type in ['h20', 'rs07', 'lg05', 'km03']:
                if self.config['load_all']:
                    lam = self.transitions_module.param_dict['lam']
                    mu = self.transitions_module.param_dict['mu']
                    x = self.transitions_module.param_dict['x']
                    y = self.transitions_module.param_dict['y']

                elif not self.config['load_all']:
                    indel_logits = self.transitions_module.indel_logits #(4,)
                    lam_logits = indel_logits[0]
                    mu_logits = indel_logits[1]
                    x_logits = indel_logits[2]
                    y_logits = indel_logits[3]

                    lambda_min_val = self.transitions_module.lambda_min_val #float
                    lambda_max_val = self.transitions_module.lambda_max_val #float

                    mu_min_val = self.transitions_module.mu_min_val #float
                    mu_max_val = self.transitions_module.mu_max_val #float

                    x_min_val = self.transitions_module.x_min_val #float
                    x_max_val = self.transitions_module.x_max_val #float

                    y_min_val = self.transitions_module.y_min_val #float
                    y_max_val = self.transitions_module.y_max_val #float

                    mu = bound_sigmoid(x = mu_logits,
                                       min_val = mu_min_val,
                                       max_val = mu_max_val).item() #float

                    x = bound_sigmoid(x = x_logits,
                                       min_val = x_min_val,
                                       max_val = x_max_val).item() #float

                    if self.transitions_module.tie_params:
                        lam = mu #float
                        y = x #float

                    elif not self.transitions_module.tie_params:
                        lam = bound_sigmoid(x = lam_logits,
                                           min_val = lambda_min_val,
                                           max_val = lambda_max_val).item() #float

                        y = bound_sigmoid(x = y_logits,
                                           min_val = y_min_val,
                                           max_val = y_max_val).item() #float

                with open(f'{out_folder}/ASCII_{prefix}_{self.indel_model_type}_indel_params.txt','a') as g:
                    g.write(f'insert rate, lambda: {lam}\n')
                    g.write(f'deletion rate, mu: {mu}\n')
                    g.write(f'probability of extending insertion, x: {x}\n')
                    g.write(f'probability of extending deletion, y: {y}\n')


class IndpSitesOldModelsLoadAll(IndpSitesOldModels):
    """
    Like IndpSitesOldModels, but load all parameters from files (excluding
    time and exponential distribution parameter).

    B = batch size; number of samples
    T = number of branch lengths; this could be:
        > an array of times for all samples (T; marginalize over these later)
        > an array of time per sample (T=B)
        > a quantized array of times per sample (T = T', where T' <= T)
    S: number of transition states (4 here: M, I, D, start/end)


    Initialize with
    ----------------
    config : dict
        config['filenames'] : dict of parameter files to load
            config['filenames']['rate_mults'] : str
                file of rate multipliers
            config['filenames']['rate_mult_probs'] : str
                file of rate multiplier probabilities

        config['indel_model_type'] : {tkf91, tkf92, None}
            which indel model, if any

        config['times_from'] : {geometric, t_array_from_file, t_per_sample}

        config['k_rate_mults'] : int
            number of rate multipliers

        config['training_dset_emit_counts'] : ArrayLike, (A,)
            observed amino acid counts to build equilibrium distribution

    name : str
        class name, for flax


    Main methods here
    -----------------
    setup


    Methods inherited from IndpSitesOldModels_F81
    -----------------------------------------------
    __call__
        unpack batch and calculate logP(desc, align | anc)

    write_params
        write parameters to files


    Methods inherited from ModuleBase
    ---------------------------------
    sow_flax_intermeds
        for tensorboard logging
    """
    config: dict
    name: str

    def setup(self):
        # not applicable here
        self.num_fragment_mixtures = 1
        self.num_domain_mixtures = 1
        self.num_transit_mixtures = 1
        self.num_site_mixtures = 1
        self.indp_rate_mults = True
        self.subst_model_type = 'f81'
        self.norm_reported_loss_by = 'desc_len'

        ### read config
        indel_model_type = self.config['indel_model_type']
        self.indel_model_type = indel_model_type.lower() if indel_model_type is not None else None
        self.exponential_dist_param = self.config.get('exponential_dist_param', 1)
        self.t_grid_step = self.config.get('t_grid_step', jnp.nan)
        self.times_from = self.config['times_from'].lower()
        self.num_rate_mults = self.config['k_rate_mults']

        ### init emission models
        # equilibrium distribution from training data counts
        self.equl_dist_module = EqulDistLogprobsFromCounts(config = self.config,
                                                           name = 'get equilibrium')

        # rate multipliers loaded from files
        self.rate_mult_module = IndpRateMultipliersFromFile(config = self.config,
                                                            name = 'get rate multipliers')

        # F81 substitution model (no file needed; F81LogprobsFromFile == F81Logprobs)
        self.logprob_subst_module = F81LogprobsFromFile(config = self.config,
                                                        name = 'f81 subst. model')

        ####################
        ### CHANGE HERE    #
        ####################
        ### init indel model (load from file)
        if self.indel_model_type == 'tkf91':
            self.transitions_module = TKF91TransitionLogprobsOldStyleFromFile(config = self.config,
                                                     name = f'tkf91 indel model')

        elif self.indel_model_type == 'tkf92':
            self.transitions_module = TKF92TransitionLogprobsOldStyleFromFile(config = self.config,
                                                     name = f'tkf92 indel model')

        else:
            self.transitions_module = OtherTransitionLogprobsFromFile(config = self.config,
                                                     name = f'{indel_model_type} indel model')
