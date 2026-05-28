#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 17:42:19 2025

"""
import jax
from jax import numpy as jnp
import flax
import numpy as np
import pickle
import sys
from tqdm import tqdm
from functools import partial
from copy import copy
from datetime import datetime
import time

from typing import Callable
import argparse
from torch.utils.tensorboard import SummaryWriter
from jax._src.typing import ArrayLike

from train_eval_fns.general_training_wrapper.training_wrapper_helpers import ( timers,
                                                      write_timing_file,
                                                      metrics_for_epoch,
                                                      jit_compile_determine_seqlen_bin,
                                                      jit_compile_determine_alignlen_bin )

from train_eval_fns.general_training_wrapper.batch_metric_helpers import (record_stats_during_hmm_training,
                                                 record_stats_during_neural_training,
                                                 write_stats_to_npz)


###############################################################################
### Base class  ###############################################################
###############################################################################
class TrainingWrapper:
    def __init__(self,
                 args : argparse.Namespace,
                 initial_training_rngkey,
                 dataloader_dict : dict,
                 train_fn_jitted : Callable,
                 eval_fn_jitted : Callable,
                 all_save_model_filenames: list[str],
                 writer : SummaryWriter): 
        """
        larger class to manage training; need this to record things during 
            training/eval
        
        arguments
        ---------
        args : Argparse Object
            > arugments from JSON config file
          
        initial_training_rngkey : PRNGKeyArray
            
        dataloader_dict : dict
            > dictionary of items from the dataloader
            > possibilities
                >> train_dset
                >> dev_dset
                >> test_dset
                >> train_dloader
                >> dev_dloader
                >> test_dloader
            
        train_fn_jitted, eval_fn_jitted : Callable
            > functions to train and eval; already jit-compiled and parted 
              (i.e. with functools.partial)
        
        all_save_model_filenames : list of strings
            > names for saving parameters from ancestor encoder, 
              descendant decoder, and final prediction model
            
        writer : torch.utils.tensorboard.SummaryWriter
        
        
        argparse object needs
        ---------------------
        use_scan_fns,
        num_epochs,
        record_every,
        
        """
        ### read arguments
        # initialize as-is
        self.args = args
        self.rngkey = initial_training_rngkey
        self.train_fn_jitted = train_fn_jitted
        self.eval_fn_jitted = eval_fn_jitted
        self.all_save_model_filenames = all_save_model_filenames
        self.writer = writer
        
        # read from argparse
        self.use_scan_fns = getattr(args, "use_scan_fns", False)
        self.num_epochs = args.num_epochs
        self.time_from = args.pred_config['times_from']
        
        # unpack dataloader dict
        self.training_dset = dataloader_dict['training_dset']
        self.training_dl = dataloader_dict['training_dl']
        
        
        ### smaller classes/attributes to remember
        # timers
        self.whole_epoch_timer = timers( num_epochs = self.num_epochs )
        
        # checkpointing, early stopping
        self.record_metrics_every_n_steps = getattr(args, "record_metrics_every_n_steps", 50)
        self.checkpoint_every_t_seconds = getattr(args, "checkpoint_every_t_seconds", 1800)
        self.early_stopping_counter = 0
        self.last_checkpoint_time = time.time()
        
        
        ### model-specific initializations
        self._model_specific_inits(dataloader_dict = dataloader_dict)


    ###########################################################################
    ### Main training function: called in CLIs   ##############################
    ###########################################################################
    def run_train_loop( self, 
                        all_trainstates ):
        # early stopping criteria, saving criteria
        best_epoch = -1
        best_trainstates = copy(all_trainstates)
        best_dev_loss = jnp.finfo(jnp.float32).max
        prev_dev_loss = jnp.finfo(jnp.float32).max
        record_count = 1
        early_stop_count = 0
        
        # record loss trajectory to a flat text file too
        loss_file = f'{self.args.logfile_dir}/losses_flat.tsv'
        with open(loss_file,'w') as g:
            g.write( ('time\t' + 
                      'epoch\t' + 
                      'ave_train_loss\t' + 
                      'ave_dev_loss\t' + 
                      'best_model\n') )
        
        # stack intermediates
        all_intermeds = []
        
        
        ##################
        ### start loop   #
        ##################
        for epoch_idx in tqdm( range(self.num_epochs) ): 
            ### train and update gradients
            # loop through train set; mini-batch updates
            train_out = self.train_one_epoch( all_trainstates,
                                              epoch_idx )
            
            all_trainstates = train_out['tstates']
            epoch_train_loss = train_out['train_loss']
            epoch_training_intermediates = train_out['training_intermediates']
            all_intermeds.append(epoch_training_intermediates)
            del train_out, epoch_training_intermediates
            
            # loop through eval set 
            epoch_dev_loss = self.eval_one_epoch( all_trainstates,
                                                  epoch_idx )
            
            # update loss text file
            with open(loss_file,'a') as g:
                best_so_far = epoch_dev_loss < best_dev_loss
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                g.write( (f'{now}\t'+
                          f'{epoch_idx}\t'+
                          f'{epoch_train_loss}\t'+
                          f'{epoch_dev_loss}\t' +
                          f'{best_so_far}\n') )
                del now
                
            
            ### if this is the best model, save it
            if best_so_far:
                with open(self.args.logfile_name,'a') as g:
                    g.write( f'New best dev loss at epoch {epoch_idx}: {epoch_dev_loss}\n' )
                
                # update "best" recordings
                best_dev_loss = epoch_dev_loss
                best_trainstates = copy(all_trainstates)
                best_epoch = epoch_idx
                
                # save the trainstates
                self._save_model( filenames = self.all_save_model_filenames,
                                  trainstate_objs = best_trainstates,
                                  suffix = 'BEST' )
            
            
            ### check if early stopping conditions are met
            early_stop = self._maybe_early_stop(prev_loss = prev_dev_loss,
                                                curr_loss = epoch_dev_loss,
                                                best_loss = best_dev_loss)
            
            if early_stop:
                # record in the raw ascii logfile
                with open(self.args.logfile_name,'a') as g:
                    g.write(f'\n\nEARLY STOPPING AT {epoch_idx}:\n')
                
                break

            # remember this epoch's loss for next iteration
            prev_dev_loss = epoch_dev_loss
        
        
        ### write finial files
        # timing
        write_timing_file( outdir = self.args.logfile_dir,
                           total_times = self.whole_epoch_timer.all_times )
        
        # training intermeds
        self._write_intermediates_after_training(all_intermeds = all_intermeds)
        
        return (early_stop, best_epoch, best_trainstates)


    ###########################################################################
    ### Epoch-level   #########################################################
    ###########################################################################
    def train_one_epoch( self, 
                         all_trainstates,
                         epoch_idx ):
        # empty dictionary to accumulate batch-level intermediates
        batch_level_intermediates = {}
        
        # epoch-level metrics
        train_metrics_recorder = metrics_for_epoch( have_acc = self.have_acc,
                                                    epoch_idx = epoch_idx ) 
        
        # start timer
        self.whole_epoch_timer.start_timer()
        
        ##################
        ### start loop   #
        ##################
        for batch_idx, batch in enumerate(self.training_dl):
            record_interms_this_batch = ( batch_idx % self.record_metrics_every_n_steps ) == 0
            epoch_batch_idx = epoch_idx * len(self.training_dl) + batch_idx
            
            ### train, sow intermediates
            out = self.train_one_batch( previous_trainstates = all_trainstates,
                                        batch = batch,
                                        epoch_idx = epoch_idx,
                                        batch_idx = batch_idx,
                                        record_interms_this_batch = record_interms_this_batch)
            train_metrics, all_trainstates = out
            del out
            
            
            ### update epoch-level metrics
            train_metrics_recorder.update_after_batch( batch_weight = batch[0].shape[0] / len(self.training_dset),
                                        batch_loss = train_metrics['batch_loss'],
                                        batch_ece = train_metrics['batch_ece'],
                                        batch_acc = train_metrics.get('batch_ave_acc', None) )
            
            
            ### possibly save model
            now = time.time()
            if now - self.last_checkpoint_time >= self.checkpoint_every_t_seconds:
                self.checkpoint_model( epoch_idx = epoch_idx,
                                       batch_idx = batch_idx,
                                       batch_loss = train_metrics['batch_loss'],
                                       trainstate_objs = all_trainstates )
                self.last_checkpoint_time = now
            del now
               
            
            ### possibly record batch-level metrics to numpy arrays
            if record_interms_this_batch:
                batch_level_intermediates = self.optional_outputs_writing_fn(step = epoch_batch_idx,
                                                 ave_loss = train_metrics['batch_loss'],
                                                 all_trainstates = all_trainstates,
                                                 which_interms_to_record = self.args.interms_for_tboard,
                                                 all_intermediates_at_curr_step = train_metrics,
                                                 dict_to_update = batch_level_intermediates)
        
        #####################
        ### post training   #
        #####################
        # flatten the batch-level intermediates
        batch_level_intermediates = {k: np.squeeze( np.stack(v, axis=0) ) for k, v in batch_level_intermediates.items()} #(1, num_batches_recorded, ...)
        
        # record metrics to tensorboard
        train_metrics_recorder.write_epoch_metrics_to_tensorboard( writer = self.writer,
                                                                    tag = 'training set')
        # return new trainstates
        out = {'tstates': all_trainstates,
               'train_loss': train_metrics_recorder.epoch_ave_loss,
               'training_intermediates': batch_level_intermediates}
        return out
    
    
    def eval_one_epoch( self, 
                        all_trainstates,
                        epoch_idx ):
        # epoch-level metrics
        dev_metrics_recorder = metrics_for_epoch( have_acc = self.have_acc,
                                                  epoch_idx = epoch_idx ) 
        
        ##################
        ### start loop   #
        ##################
        for batch_idx, batch in enumerate(self.dev_dl):
            batch_epoch_idx = epoch_idx * len(self.dev_dl) + batch_idx
            
            ### eval, sow intermediates
            batch_max_seqlen, batch_max_alignlen = self._set_sequence_lengths_for_jit(batch)
            eval_metrics = self.eval_fn_jitted( batch = batch, 
                                                all_trainstates = all_trainstates, 
                                                max_seq_len = batch_max_seqlen,
                                                max_align_len = batch_max_alignlen )
                        
            
            ### update epoch-level metrics
            dev_metrics_recorder.update_after_batch( batch_weight = batch[0].shape[0] / len(self.dev_dset),
                                                     batch_ece = eval_metrics['batch_ece'],
                                                     batch_loss = eval_metrics['batch_loss'],
                                                     batch_acc = eval_metrics.get('batch_ave_acc', None) )
            
            del eval_metrics
        
        
        #################
        ### post eval   #
        #################
        # final records for the full epoch
        dev_metrics_recorder.write_epoch_metrics_to_tensorboard( writer = self.writer,
                                                                 tag = 'dev set' )
            
        # record total time spent doing train + eval
        self.whole_epoch_timer.end_timer_and_write_to_tboard( epoch_idx = epoch_idx,
                                                              writer = self.writer,
                                                              tag = 'Process one epoch' )
        
        return dev_metrics_recorder.epoch_ave_loss
    

    ###########################################################################
    ### Batch-level   #########################################################
    ###########################################################################
    def train_one_batch( self, 
                         previous_trainstates,
                         batch,
                         epoch_idx,
                         batch_idx,
                         record_interms_this_batch ):
        ### change random key
        self.rngkey, rngkey_for_batch = jax.random.split(self.rngkey)
        batch_epoch_idx = epoch_idx * len(self.training_dl) + batch_idx
        rngkey_for_batch = jax.random.fold_in(rngkey_for_batch, batch_epoch_idx) 
        
        batch_max_seqlen, batch_max_alignlen = self._set_sequence_lengths_for_jit(batch)

        ### main loop
        # train
        out = self.train_fn_jitted(batch=batch, 
                              training_rngkey = rngkey_for_batch, 
                              all_trainstates = previous_trainstates, 
                              max_seq_len = batch_max_seqlen,
                              max_align_len = batch_max_alignlen,
                              record_interms_this_batch = record_interms_this_batch)
        train_metrics, updated_trainstates = out
        del out
        
        # check for nan loss; will quit if this happens
        self._check_for_nan_train_loss( loss = train_metrics['batch_loss'],
                                        epoch_idx = epoch_idx,
                                        trainstate_objs = updated_trainstates,
                                        batch = batch )
        
        return train_metrics, updated_trainstates
    
    

    ###########################################################################
    ### Internal helpers   ####################################################
    ###########################################################################
    def _save_model( self,
                      filenames: list,
                      trainstate_objs: list,
                      suffix = None ):
        for i in range(len(trainstate_objs)):
            new_outfile = filenames[i]
            
            if suffix is not None:
                new_outfile = new_outfile.replace('.pkl',f'_{suffix}.pkl')
            
            with open(new_outfile, 'wb') as g:
                model_state_dict = flax.serialization.to_state_dict(trainstate_objs[i])
                pickle.dump(model_state_dict, g)    
    
    def checkpoint_model( self,
                            epoch_idx,
                            batch_idx,
                            batch_loss,
                            trainstate_objs ):
        # save some metadata about the trainstate files
        with open(f'{self.args.model_ckpts_dir}/INPROGRESS_trainstates_info.txt','w') as g:
            g.write(f'Checkpoint created at: epoch {epoch_idx}, batch {batch_idx}\n')
            g.write(f'Current loss for the training set batch is: {batch_loss}\n')
        
        # save the trainstates
        self._save_model( filenames = self.all_save_model_filenames,
                          trainstate_objs = trainstate_objs, 
                          suffix = 'INPROGRESS' )
        
        # update the general logfile
        with open(self.args.logfile_name,'a') as g:
            g.write(f'\tCheckpoint created! Train loss at epoch {epoch_idx}, batch {batch_idx}: {batch_loss}\n')
        
            
    def _maybe_early_stop(self,
                          prev_loss,
                          curr_loss,
                          best_loss):
        # condition 1: if dev loss stagnates or starts to go up, compared
        #              to previous epoch's dev loss
        cond1 = jnp.allclose( prev_loss, 
                              jnp.minimum (prev_loss, curr_loss), 
                              atol=self.args.early_stop_cond1_atol,
                              rtol=0 )

        # condition 2: if dev loss is substatially worse than best dev loss
        cond2 = (curr_loss - best_loss) > self.args.early_stop_cond2_gap

        if cond1 or cond2:
            self.early_stopping_counter += 1
        else:
            self.early_stopping_counter = 0
        
        return (self.early_stopping_counter  == self.args.patience)
    
    def _check_for_nan_train_loss( self, 
                                    loss,
                                    epoch_idx,
                                    trainstate_objs,
                                    batch ):
        if jnp.isnan( loss ):
            # save the argparse object by itself
            self.args.epoch_idx = epoch_idx
            with open(f'{self.args.model_ckpts_dir}/TRAINING_ARGPARSE_BROKEN.pkl', 'wb') as g:
                pickle.dump(self.args, g)
            
            # save the trainstates after the parameter update
            self._save_model( filenames = self.all_save_model_filenames,
                              trainstate_objs = trainstate_objs,
                              suffix = 'BROKEN' )
            
            # save the time array for the batch, if using a time per sample
            if self.times_from =='t_per_sample':
                times_for_matrices = batch[4] #(B,)
            
            with open(f'{self.args.model_ckpts_dir}/TIMES_AT_BROKEN_BATCH.pkl', 'wb') as g:
                np.save(g, times_for_matrices)
            
            # record timing so far (if any)
            write_timing_file( outdir = self.args.logfile_dir,
                               total_times = self.whole_epoch_timer.all_times )
            
            raise RuntimeError( ('NaN loss detected; saved intermediates '+
                                'and quit training') )
    
    def _set_sequence_lengths_for_jit( self, 
                                        batch ):
        # unpack briefly to get max len and number of samples in the 
        #   batch; place in some bin (this controls how many jit 
        #   compilations you do)
        batch_max_seqlen = self.seqlen_bin_fn(batch = batch).item()
        batch_max_alignlen = self.alignlen_bin_fn(batch = batch).item()
        
        # if function returns -1, replace with None
        batch_max_seqlen = None if batch_max_seqlen==-1 else batch_max_seqlen
        batch_max_alignlen = None if batch_max_alignlen==-1 else batch_max_alignlen
        
        if self.use_scan_fns:
            err = (f'batch_max_alignlen (not including bos) is: '+
                    f'{batch_max_alignlen - 1}'+
                    f', which is not divisible by length for scan '+
                    f'({self.args.chunk_length})')
            assert (batch_max_alignlen - 1) % self.args.chunk_length == 0, err
        
        return batch_max_seqlen, batch_max_alignlen
    
    def _write_intermediates_after_training(self,
                                            all_intermeds: list[dict]):
        dict_to_write = {}
        keys = all_intermeds[0].keys()
        for key in all_intermeds[0].keys():
            concat_arr = np.stack( [d[key] for d in all_intermeds], axis=0 )
            dict_to_write[key] = concat_arr
        del all_intermeds
        
        # save
        keyword_lst = ['ANC_INTERMS',
                       'DESC_INTERMS',
                       'FINALPRED_INTERMS',
                       'WEIGHTS',
                       'GRADIENTS',
                       'ADAM_OPTIMIZER',
                       'PARAM_UPDATE']
        
        write_stats_to_npz(dict_to_write = dict_to_write,
                               keyword_lst = keyword_lst,
                               folder = self.args.out_arrs_dir,
                               file_prefix = 'TRAIN_SET')
    
    def _model_specific_inits(self):
        raise NotImplementedError('depends on model!')


###############################################################################
### Model-specific subclasses  ################################################
###############################################################################
class NeuralTKFTrainingWrapper(TrainingWrapper):
    def _model_specific_inits(self, dataloader_dict):
        # check model type again
        assert self.args.pred_model_type == 'neural_hmm'
        
        # add dev set
        self.dev_dset = dataloader_dict['dev_dset']
        self.dev_dl = dataloader_dict['dev_dl']
        
        # continue init
        self.seqlen_bin_fn = jit_compile_determine_seqlen_bin(self.args)
        self.alignlen_bin_fn = jit_compile_determine_alignlen_bin(self.args)
        self.have_acc = False
        self.use_tkf_funcs = True
        self.optional_outputs_writing_fn = record_stats_during_neural_training
        

class FeedforwardTrainingWrapper(TrainingWrapper):
    def _model_specific_inits(self, dataloader_dict):
        # check model type again
        assert self.args.pred_model_type == 'feedforward'
        
        # add dev set
        self.dev_dset = dataloader_dict['dev_dset']
        self.dev_dl = dataloader_dict['dev_dl']
        
        # continue init
        self.seqlen_bin_fn = jit_compile_determine_seqlen_bin(self.args)
        self.alignlen_bin_fn = jit_compile_determine_alignlen_bin(self.args)
        self.have_acc = True
        self.use_tkf_funcs = False
        self.optional_outputs_writing_fn = record_stats_during_neural_training


class TransitMixesTrainingWrapper(TrainingWrapper):
    def _model_specific_inits(self, dataloader_dict):
        # check model type again
        assert self.args.pred_model_type in ['pairhmm_frag_and_site_classes',
                                             'pairhmm_nested_tkf']
        
        # replace "dev set" with a copy of the test set
        self.dev_dset = dataloader_dict['test_dset']
        self.dev_dl = dataloader_dict['test_dl']
        
        # continue init
        self.seqlen_bin_fn = lambda *args, **kwargs: -jnp.ones(())
        self.alignlen_bin_fn = jit_compile_determine_alignlen_bin(self.args)
        self.have_acc = False
        self.use_tkf_funcs = True
        self.optional_outputs_writing_fn = record_stats_during_hmm_training

class IndpSitesTrainingWrapper(TrainingWrapper):
    def _model_specific_inits(self, dataloader_dict):
        # check model type again
        assert 'pairhmm_indp_sites' in self.args.pred_model_type
        
        # replace "dev set" with a copy of the test set
        self.dev_dset = dataloader_dict['test_dset']
        self.dev_dl = dataloader_dict['test_dl']
        
        # continue init
        self.seqlen_bin_fn = lambda *args, **kwargs: -jnp.ones(())
        self.alignlen_bin_fn = lambda *args, **kwargs: -jnp.ones(())
        self.have_acc = False
        self.use_tkf_funcs = True
        self.optional_outputs_writing_fn = record_stats_during_hmm_training
