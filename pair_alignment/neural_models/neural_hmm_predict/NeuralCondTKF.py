#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 17:11:33 2025

"""
import sys

from flax import linen as nn
from flax.core import FrozenDict, copy
import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm as matrix_exp
from jax.scipy.special import logsumexp
import optax

from utils.BaseClasses import ModuleBase
from neural_models.neural_hmm_predict.scoring_fns import (score_f81_substitutions_marg_over_times,
                                                   score_f81_substitutions_t_per_samp,
                                                   score_indels,
                                                   score_transitions)
from neural_models.neural_hmm_predict.emission_models import (GlobalEqul,
                                                       LocalEqul,
                                                       LocalF81)
from neural_models.neural_hmm_predict.transition_models import TKF92LocalRateLocalFragSize

from neural_models.neural_shared.postprocessing_models import (FeedforwardPostproc,
                                                       SelectMask)


class NeuralCondTKF(ModuleBase):
    config: dict
    name: str
    
    def setup(self):
        """
        Attributes from pred_config:
        -----------------------------
        self.times_from : str
        self.subst_model_type : str
        self.exponential_dist_param : float
        self.transitions_postproc_config : dict
        self.emissions_postproc_config : dict
        
        Attributes created after model initialization:
        -----------------------------------------------
        self.postproc_equl
        self.equl_module
        
        self.postproc_subs
        self.subs_module
        
        self.postproc_trans
        self.trans_module
        
        decide the post-processing architecture to combine 
        position-specific embeddings with:
        ---------------------------------------------------
            - config['emissions_postproc_module']
              - provide config for this called config['emissions_postproc_config']
            - config['transitions_postproc_module']
              - provide config for this called config['transitions_postproc_config']
        
        entries in config['global_or_local'] (all keep at "local"):
        -----------------------------------------------------------
        equl_dist, rate_mult, tkf_rates, tkf92_frag_size
        """
        ###################
        ### read config   #
        ###################
        ### required
        self.times_from = self.config['times_from']
        self.subst_model_type = 'f81'
        self.indel_model_type = 'tkf92'
        global_or_local_dict = self.config['global_or_local']
        times_from = self.config['times_from'].lower()
        
        
        ### optional: for regularization
        default_reg = {'equl_dist': 0,
                       'rate_mult': 0,
                       'tkf_lambda': 0,
                       'tkf_mu': 0,
                       'tkf92_frag_size': 0}
        
        regularization_rates = self.config.get('regularization_rates', default_reg)

        # all emissions
        if 'training_dset_emit_counts' in self.config.keys():
            aa_counts = self.config['training_dset_emit_counts']
            
        elif not 'training_dset_emit_counts' in self.config.keys():
            aa_counts = jnp.array( [1]*self.config['emission_alphabet_size'] )
        
        default_pi = aa_counts / sum( aa_counts )
        
        # dict of priors
        default_priors = {'equl_dist': default_pi,
                          'rate_mult': 1,
                          'tkf_lambda': 0.029351761429568933,
                          'tkf_mu': 0.03000306870475785,
                          'tkf92_frag_size': 0.6843917135902318}
        
        regularization_priors = self.config.get('regularization_priors', default_priors)
        if isinstance( regularization_priors['equl_dist'], tuple):
            regularization_priors = copy( regularization_priors, 
                                          {'equl_dist': jnp.array(regularization_priors['equl_dist'])} )
            
        self.regularization_rates = regularization_rates
        self.regularization_priors = regularization_priors
        self.skip_regularization = all([v==0.0 for v in regularization_rates.values()])
        del default_reg, default_priors, default_pi
        
        
        ### other optional dictionaries
        self.transitions_postproc_config = self.config.get('transitions_postproc_config', dict() )
        self.emissions_postproc_config = self.config.get('emissions_postproc_config', dict() )
        self.transitions_postproc_model_type = self.config.get('transitions_postproc_model_type', None)
        emissions_postproc_model_type = self.config.get('emissions_postproc_model_type', None)
        self.exponential_dist_param = 1
        
        # handle time
        if times_from =='t_per_sample':
            self.unique_time_per_sample = True
        
        elif times_from in ['geometric','t_array_from_file']:
            self.unique_time_per_sample = False
        
        
        ######################################################################
        ### pick architecture for preprocessing features, for local params   #
        ######################################################################
        postproc_module_registry = {'selectmask': SelectMask,
                                    'feedforward': FeedforwardPostproc,
                                    None: lambda *args, **kwargs: lambda *args, **kwargs: None}
        
        transitions_postproc_module = postproc_module_registry[self.transitions_postproc_model_type]
        emissions_postproc_module = postproc_module_registry[emissions_postproc_model_type]
        
        
        ###########################################
        ### module for equilibrium distribution   #
        ###########################################
        # postprocess the concatenated embeddings
        self.postproc_equl = emissions_postproc_module(config = self.emissions_postproc_config,
                                           name = f'{self.name}/{emissions_postproc_model_type}_to_equl_module')
        
        # get equilibrium distribution
        if global_or_local_dict['equl_dist'].lower() == 'global':
            self.equl_module = GlobalEqul(config = self.emissions_postproc_config,
                                          name = f'{self.name}/get_equl')
        
        elif global_or_local_dict['equl_dist'].lower() == 'local':
            self.equl_module = LocalEqul(config = self.emissions_postproc_config,
                                          name = f'{self.name}/get_equl')
        
        
        ########################################
        ### module for logprob substitutions   #
        ########################################
        # postprocess the concatenated embeddings
        self.postproc_subs =  emissions_postproc_module(config = self.emissions_postproc_config,
                                            name = f'{self.name}/{emissions_postproc_model_type}_to_subs_module')
        
        self.subs_module = LocalF81(config = self.emissions_postproc_config,
                                    name = f'{self.name}/get_subs')

            
        ###########################################
        ### module for transition probabilities   #
        ###########################################
        # postprocess the concatenated embeddings
        self.postproc_trans =  transitions_postproc_module(config = self.transitions_postproc_config,
                                             name = f'{self.name}/{self.transitions_postproc_model_type}_to_trans_module')
        
        self.trans_module = TKF92LocalRateLocalFragSize(config = self.transitions_postproc_config,
                                                 name = f'{self.name}/get_trans')
            
    def __call__(self, 
                 datamat_lst: list[jnp.array], 
                 padding_mask: jnp.array,  #(B, L)
                 t_array: jnp.array, #(B,) or (T,)
                 training: bool, 
                 sow_flax_intermeds: bool=False,
                 *args,
                 **kwargs):
        """
        unlike pairHMM implementation, this ONLY generates scoring matrices
        don't feed times here; they get incorporated during scoring only!
        """
        # elements of datamat_lst are:
        # anc_embeddings: (B, L, H)
        # desc_embeddings: (B, L, H)
        # prev_align_one_hot_vec: (B, L, 5)

        ### equilibrium distribution; used to score emissions from indel sites
        equl_feats = self.postproc_equl(anc_emb = datamat_lst[0],
                                        desc_causal_emb = datamat_lst[1],
                                        prev_align_one_hot_vec = datamat_lst[2],
                                        padding_mask = padding_mask,
                                        training = training,
                                        sow_flax_intermeds = sow_flax_intermeds,
                                        t_array = None)  #(B, L, H_out)
        
        logprob_emit_indel = self.equl_module(datamat = equl_feats,
                                              sow_flax_intermeds = sow_flax_intermeds) #(B, L, A)
        
        # add pseudocounts (via pseudo-frequencies)
        if self.regularization_rates['equl_dist'] > 0:
            """
            p_{corrected} = ( p_{model} + rate * p_{observed} ) / (1 + rate)
            """
            log_observed_freq = jnp.where( self.regularization_priors['equl_dist'] > 0,
                                           jnp.log( self.regularization_priors['equl_dist'] ),
                                           jnp.finfo(jnp.float32).min
                                           ) #(A,)
            log_observed_freq = log_observed_freq[None, None, :] #(1, 1, A)
            
            #  p_{model} + \lambda * p_{observed}
            log_num = jnp.logaddexp(logprob_emit_indel,
                                    jnp.log(self.regularization_rates['equl_dist']) + log_observed_freq) #(B, L, A)
            
            # ( p_{model} + \lambda * p_{observed} ) / (1 + \lambda)
            logprob_emit_indel = log_num - jnp.log1p( self.regularization_rates['equl_dist'] ) #(B, L, A)
        
        
        ### substitution model; used to score emissions from match sites
        # don't feed times here; they get incorporated during scoring only!
        sub_feats = self.postproc_subs( anc_emb = datamat_lst[0],
                                        desc_causal_emb = datamat_lst[1],
                                        prev_align_one_hot_vec = datamat_lst[2],
                                        padding_mask = padding_mask,
                                        training = training,
                                        sow_flax_intermeds = sow_flax_intermeds,
                                        t_array = None )  #(B, L, H_out)
        
        # logprob_emit_match is either (T, B, L, A, A) or (B, L, A, A)
        # subs_model_params is a dictionary of parameters; see module for more details
        logprob_emit_match, subs_model_params = self.subs_module(datamat = sub_feats,
                                                                 padding_mask = padding_mask,
                                                                 log_equl = logprob_emit_indel,
                                                                 t_array = t_array,
                                                                 unique_time_per_sample = self.unique_time_per_sample,
                                                                 sow_flax_intermeds = sow_flax_intermeds)
        
        
        ### transition model; used to score markovian alignment path
        # don't feed times here; they get incorporated during scoring only!
        trans_feats = self.postproc_trans(anc_emb = datamat_lst[0],
                                        desc_causal_emb = datamat_lst[1],
                                        prev_align_one_hot_vec = datamat_lst[2],
                                        padding_mask = padding_mask,
                                        training = training,
                                        sow_flax_intermeds = sow_flax_intermeds,
                                        t_array = None)  #(B, L, H_out)
        
        out = self.trans_module(datamat = trans_feats,
                                t_array = t_array,
                                unique_time_per_sample = self.unique_time_per_sample,
                                sow_flax_intermeds = sow_flax_intermeds) 
        
        # logprob_transits is either (T, B, L, S, S) or (B, L, S, S)
        # see module for more details
        logprob_transits, indel_model_params = out
        del out
        
        # out dictionary
        out_dict = {'logprob_emit_match': logprob_emit_match,  # (T, B, L, A, A) or (B, L, A, A)
                    'logprob_emit_indel': logprob_emit_indel,  #(B, L, A)
                    'logprob_transits': logprob_transits, # (T, B, L, S, S) or (B, L, S, S)
                    'subs_model_params': subs_model_params, #dict
                    'indel_model_params': indel_model_params}  #dict
        
        
        ### correction to conditional logprob, if tkf92
        ### if alignment begins with S -> ins, then this class will
        ###   see ancestor transition path as em -> em -> ... instead 
        ###   of S -> em -> ...; that is, it will omit an (S->em) 
        ###   transition and add an extra (em -> em) transition.
        
        lam = indel_model_params['lambda'] #(B, L) or (1,1)
        mu = indel_model_params['mu'] #(B, L) or (1,1)
        r_extend = indel_model_params['r_extend'] #(B, L) or (1,1)

        corr = ( lam / mu ) / ( r_extend + (1-r_extend)*(lam/mu) )
        log_corr = jnp.log(corr)

        out_dict['corr'] = log_corr #(B, L) or (1,1)
        return out_dict
        
    
    def neg_loglike_in_scan_fn(self, 
                              logprob_emit_match: jnp.array, 
                              logprob_emit_indel: jnp.array,
                              logprob_transits: jnp.array,
                              corr: jnp.array,
                              true_out: jnp.array,
                              gap_idx: int=43,
                              padding_idx: int=0,
                              start_idx: int=1,
                              end_idx: int=2,
                              return_result_before_sum: bool=False,
                              return_transit_emit: bool=False,
                              *args,
                              **kwargs):
        """
        loss of alignment path, given by alignment_state
        
        true_out is (B, L, 4):
            - dim2 = 0: gapped anc at currrent position b
            - dim2 = 1: gapped desc at currrent position b
            - dim2 = 2: from state (0-6) "prev_state", a
            - dim2 = 3: to_state (0-6) "curr_state", b
        """
        B = logprob_emit_indel.shape[0]
        L = logprob_emit_indel.shape[1]
        
        # unpack inputs
        staggered_alignment_state = true_out[...,2:] #(B, length_for_scan, 2)
        curr_state = true_out[...,3] #(B,length_for_scan)
        true_alignment_without_start = true_out[...,:2] #(B, length_for_scan, 2)
        
        
        ### score transitions
        # if unique_time_per_sample, tr is (B, length_for_scan)
        # elif not unique_time_per_sample, tr is (T, B, length_for_scan)
        tr = score_transitions(staggered_alignment_state = staggered_alignment_state,
                               logprob_trans_mat = logprob_transits, 
                               unique_time_per_sample = self.unique_time_per_sample,
                               padding_idx=padding_idx)
        
        # extra correction factors for S -> I transition
        s_i_corr_mask = jnp.all( staggered_alignment_state == jnp.array([4, 2]), axis=-1 ) #(B, length_for_scan)
        
        if len(tr.shape) == 3:
            s_i_corr_mask = jnp.broadcast_to( s_i_corr_mask[None,...], tr.shape ) #(T, B, length_for_scan)
            s_i_correction = jnp.broadcast_to( corr[None,...], tr.shape ) #(T, B, length_for_scan)
        
        elif len(tr.shape) == 2:
            s_i_correction = jnp.broadcast_to( corr, tr.shape ) #(B, length_for_scan)
            
        # make corrections selectively, per sample and per position
        tr = jnp.where( s_i_corr_mask,
                        tr - s_i_correction,
                        tr ) #(T, B, length_for_scan) or (B, length_for_scan)
        
        
        
        ### score emissions
        # if unique_time_per_sample, e is (B, length_for_scan)
        # elif not unique_time_per_sample, e is (T, B, length_for_scan)
        if self.unique_time_per_sample:
            e = jnp.zeros( (B,L) )
        elif not self.unique_time_per_sample:
            T = logprob_emit_match.shape[0]
            e = jnp.zeros( (T,B,L) )
        
        # match positions: decide function
        if self.unique_time_per_sample:
            score_substitutions = score_f81_substitutions_t_per_samp
        else:
            score_substitutions = score_f81_substitutions_marg_over_times
            
        # match positions: score
        e = e + jnp.where( curr_state == 1,
                           score_substitutions( true_alignment_without_start = true_alignment_without_start,
                                                logprob_scoring_mat = logprob_emit_match,
                                                unique_time_per_sample = self.unique_time_per_sample,
                                                gap_idx = gap_idx,
                                                padding_idx = padding_idx,
                                                start_idx = start_idx,
                                                end_idx = end_idx),
                           0 )
        
        # insert positions: score with descendant tok
        e = e + jnp.where( curr_state == 2,
                           score_indels(true_alignment_without_start = true_alignment_without_start,
                                        logprob_scoring_vec = logprob_emit_indel,
                                        which_seq = 'desc',
                                        gap_idx = gap_idx,
                                        padding_idx = padding_idx,
                                        start_idx = start_idx,
                                        end_idx = end_idx),
                           0 )
        # conditional logprob, so don't score "emissions" from ancestor tokens
        
        ### final logprob(sequences)
        # if unique_time_per_sample, logprob_perSamp_perPos_perTime is (B, length_for_scan)
        # elif not unique_time_per_sample, logprob_perSamp_perPos_perTime is (T, B, length_for_scan)
        logprob_perSamp_perPos_perTime = tr + e
        
        if return_result_before_sum:
            # if unique_time_per_sample, logprob_perSamp_perPos_perTime is (B, length_for_scan)
            # elif not unique_time_per_sample, logprob_perSamp_perPos_perTime is (T, B, length_for_scan)
            out = {'logprob_perSamp_perPos_perTime': logprob_perSamp_perPos_perTime} #(T, B, length_for_scan) or (B, length_for_scan)
        
        elif not return_result_before_sum:
            # take the SUM down the length to get logprob_perSamp_perTime:
            # if unique_time_per_sample, logprob_perSamp_perTime is (B)
            # elif not unique_time_per_sample, logprob_perSamp_perTime is (T, B)
            logprob_perSamp_perTime = logprob_perSamp_perPos_perTime.sum(axis=-1) #(T,B) or (B,)
            out = {'logprob_perSamp_perTime': logprob_perSamp_perTime} #(T,B) or (B,)
        
        # possible return intermediate scores (for debugging)
        if return_transit_emit:
            out['tr'] = tr
            out['e'] = e
        
        return out
    
    def accumulate_parameter_sums_in_scan_fn(self,
                                             true_out,
                                             subs_model_params,
                                             indel_model_params):
        if not self.skip_regularization:
            curr_state = true_out[...,3] #(B,length_for_scan)
            valid_alignment_cols = (curr_state != 0) #(B,length_for_scan)
            
            # mask and sum
            rate_multiplier_sum = jnp.multiply( subs_model_params['rate_multiplier'], 
                                                valid_alignment_cols ).sum() #(B,length_for_scan)
            tkf_lambda_sum = jnp.multiply( indel_model_params['lambda'], 
                                           valid_alignment_cols ).sum() #(B,length_for_scan)
            tkf_mu_sum = jnp.multiply( indel_model_params['mu'], 
                                       valid_alignment_cols ).sum() #(B,length_for_scan)
            tkf92_frag_size_sum = jnp.multiply( indel_model_params.get('r_extend',0), 
                                                valid_alignment_cols ).sum() #(B,length_for_scan)
            
            out = {'total_seen_toks': valid_alignment_cols.sum(),
                   'rate_multiplier_sum': rate_multiplier_sum,
                   'tkf_lambda_sum': tkf_lambda_sum,
                   'tkf_mu_sum': tkf_mu_sum,
                   'tkf92_frag_size_sum': tkf92_frag_size_sum}
            return out
        
        else:
            return {'total_seen_toks': 0,
                    'rate_multiplier_sum': 0,
                    'tkf_lambda_sum': 0,
                    'tkf_mu_sum': 0,
                    'tkf92_frag_size_sum': 0}
        
    def evaluate_loss_after_scan(self, 
                                 loss_dict: dict,
                                 length_for_normalization_for_reporting: jnp.array,
                                 t_array: jnp.array,
                                 padding_idx: int = 0,
                                 *args,
                                 **kwargs):
        """
        postprocessing after accumulating logprobs in a scan function
        """
        ### handle time, if needed
        logprob_perSamp_perTime = loss_dict['logprob_perSamp_perTime'] #(B,) or (T, B)

        # marginalize over time grid
        if not self.unique_time_per_sample:
            if t_array.shape[0] > 1:
                # logP(t_k) = exponential distribution
                logP_time = ( jnp.log(self.exponential_dist_param) - 
                              (self.exponential_dist_param * t_array) ) #(T,)
                log_t_grid = jnp.log( t_array[1:] - t_array[:-1] ) #(T-1,)
                
                # kind of a hack, but repeat the last time array value
                log_t_grid = jnp.concatenate( [log_t_grid, log_t_grid[-1][None] ], axis=0)  #(T,)
                
                logP_perSamp_perTime_withConst = ( logprob_perSamp_perTime +
                                                   logP_time[:,None] +
                                                   log_t_grid[:,None] ) #(T,B)
                
                logprob_perSamp = logsumexp(logP_perSamp_perTime_withConst, axis=0) #(B,)
            
            elif t_array.shape[0] == 1:
                logprob_perSamp = logprob_perSamp_perTime[0,...]
        
        # otherwise, just rename the variable
        elif self.unique_time_per_sample:
            logprob_perSamp = logprob_perSamp_perTime
        
        # calculate loss
        loss = -jnp.mean(logprob_perSamp)
        
        ### collect loss and intermediate values
        intermediate_vals = { 'sum_neg_logP': -logprob_perSamp,
                              'neg_logP_length_normed': -logprob_perSamp/length_for_normalization_for_reporting}
        
        return loss, intermediate_vals
    
    def regularize_loss_using_mean_evoparams( self,
                                              raw_loss,
                                              sums_dict ):
        if not self.skip_regularization:
            ### unpack dictionary
            total_seen_toks = sums_dict['total_seen_toks']
            rate_multiplier_sum = sums_dict['rate_multiplier_sum']
            tkf_lambda_sum = sums_dict['tkf_lambda_sum']
            tkf_mu_sum = sums_dict['tkf_mu_sum']
            tkf92_frag_size_sum = sums_dict['tkf92_frag_size_sum']
            
            # push mean rate multiplier to be close to rate_mult prior
            out = self._regularize_based_on_mean_params( prev_loss = raw_loss,
                                                         observed_sum = rate_multiplier_sum,
                                                         total_aligned_toks = total_seen_toks,
                                                         key_for_reg_dicts = 'rate_mult' )
            loss, mean_rate_mult = out
            del out, raw_loss
            
            # push mean tkf insert rate to be close to tkf_insert prior
            out = self._regularize_based_on_mean_params( prev_loss = loss,
                                                         observed_sum = tkf_lambda_sum,
                                                         total_aligned_toks = total_seen_toks,
                                                         key_for_reg_dicts = 'tkf_lambda' )
            loss, mean_tkf_lambda = out
            del out
            
            # push mean tkf delete rate to be close to tkf_insert prior
            out = self._regularize_based_on_mean_params( prev_loss = loss,
                                                         observed_sum = tkf_mu_sum,
                                                         total_aligned_toks = total_seen_toks,
                                                         key_for_reg_dicts = 'tkf_mu' )
            loss, mean_tkf_mu = out
            del out
            
            # push mean tkf fragment extension prob to be close to tkf92_frag_size prior
            out = self._regularize_based_on_mean_params( prev_loss = loss,
                                                         observed_sum = tkf92_frag_size_sum,
                                                         total_aligned_toks = total_seen_toks,
                                                         key_for_reg_dicts = 'tkf92_frag_size' )
            loss, mean_tkf92_frag_size = out
            del out
            
            all_means = {'mean_rate_mult': mean_rate_mult,
                         'mean_tkf_lambda': mean_tkf_lambda,
                         'mean_tkf_mu': mean_tkf_mu,
                         'mean_tkf92_frag_size': mean_tkf92_frag_size}
            return loss, all_means
        
        
        elif self.skip_regularization:
            # return -1 to indicate that this step was skipped
            all_means = {'mean_rate_mult': -1,
                         'mean_tkf_lambda': -1,
                         'mean_tkf_mu': -1,
                         'mean_tkf92_frag_size': -1}
            return raw_loss, all_means
        
    def _regularize_based_on_mean_params( self, 
                                          prev_loss,
                                          observed_sum, 
                                          total_aligned_toks, 
                                          key_for_reg_dicts ):
        mean = observed_sum / total_aligned_toks
        distance = jnp.square( self.regularization_priors[key_for_reg_dicts] - mean )
        reg_loss = prev_loss + ( self.regularization_rates[key_for_reg_dicts] * distance )
        return reg_loss, mean
    
    def get_perplexity_per_sample(self,
                                  loss_fn_dict):
        neg_logP_length_normed = loss_fn_dict['neg_logP_length_normed']
        perplexity_perSamp = jnp.exp(neg_logP_length_normed) #(B,)
        return perplexity_perSamp
    

