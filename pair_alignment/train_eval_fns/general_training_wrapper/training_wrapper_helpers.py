#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 18:06:40 2025


'clip_by_bins',
'determine_alignlen_bin',
'determine_seqlen_bin',
'jit_compile_determine_alignlen_bin',
'jit_compile_determine_seqlen_bin',
'metrics_for_epoch',
'record_postproc_time_table',
'selective_squeeze',
'wall_clock_time',
'write_times_while_training',
'write_timing_file'

"""
import os
from functools import partial
from time import perf_counter as wall_clock_time
from time import process_time
import platform
import numpy as np
import pandas as pd
import subprocess
from datetime import datetime

import jax
import jax.numpy as jnp


###############################################################################
### Handle metrics   ##########################################################  
###############################################################################  
class metrics_for_epoch:
    def  __init__(self,
                  have_acc,
                  epoch_idx):
        self.have_acc = have_acc
        self.epoch_idx = epoch_idx
        
        self.epoch_ave_loss = 0
        self.epoch_ave_loss_length_normed = 0
        
        if self.have_acc:
            self.epoch_ave_acc = 0
            
    def update_after_batch(self,
                            batch_weight,
                            batch_loss,
                            batch_ece,
                            batch_acc = None):
        # loss
        self.epoch_ave_loss += batch_loss * batch_weight
        
        # loss normalized by descendant length (for ece)
        # jnp.exp( joint_neg_logP_length_normed.mean() )
        batch_loss_length_normed = jnp.log( batch_ece )
        self.epoch_ave_loss_length_normed += batch_loss_length_normed * batch_weight
        
        # acc
        if self.have_acc and (batch_acc is not None):
            self.epoch_ave_acc += batch_acc * batch_weight
        
    
    def write_epoch_metrics_to_tensorboard(self,
                                            writer,
                                            tag):
        writer.add_scalar( tag = f'Loss/{tag}', 
                           scalar_value = self.epoch_ave_loss.item(), 
                           global_step = self.epoch_idx )
        
        writer.add_scalar( tag = f'ECE/{tag}',
                           scalar_value = np.exp( self.epoch_ave_loss_length_normed.item() ),
                           global_step = self.epoch_idx )
        
        if self.have_acc:
            writer.add_scalar( tag=f'Accuracy/{tag}',
                               scalar_value=self.epoch_ave_acc.item(), 
                               global_step=self.epoch_idx )    
        

###############################################################################
### Deal with time   ##########################################################
###############################################################################
class timers:
    def __init__(self, 
                 num_epochs):
        self.num_epochs = num_epochs
        self.all_times = np.zeros( (self.num_epochs, 2) )
        self.cache = None
        
    def start_timer(self):
        real = wall_clock_time()
        cpu = process_time()
        self.cache = (real, cpu)
    
    def _end_timer(self):
        real_start, cpu_start = self.cache
        real_end = wall_clock_time()
        cpu_end = process_time() 
        
        # clear cache
        self.cache = None
        
        # return all times
        return {'real_start': real_start,
                'cpu_start': cpu_start,
                'real_end': real_end,
                'cpu_end': cpu_end}
    
    def end_timer_get_deltas(self):
        out = self._end_timer()
        
        real_start = out['real_start']
        cpu_start = out['cpu_start']
        real_end = out['real_end']
        cpu_end = out['cpu_end']
        
        real_delta = real_end - real_start
        cpu_delta = cpu_end - cpu_start
        
        return (real_delta, cpu_delta)
        
    def end_timer_and_write_to_tboard(self, 
                                      epoch_idx,
                                      writer,
                                      tag ):
        out = self._end_timer()
        
        real_start = out['real_start']
        cpu_start = out['cpu_start']
        real_end = out['real_end']
        cpu_end = out['cpu_end']
        
        # record for later
        real_delta = real_end - real_start
        cpu_delta = cpu_end - cpu_start
        self.all_times[epoch_idx, 0] = real_delta
        self.all_times[epoch_idx, 1] = cpu_delta

        # write to tensorboard
        write_times_while_training(cpu_start = cpu_start, 
                                   cpu_end = cpu_end, 
                                   real_start = real_start, 
                                   real_end = real_end, 
                                   tag = tag, 
                                   step = epoch_idx, 
                                   writer_obj = writer)  

def write_times_while_training(cpu_start, 
                               cpu_end, 
                               real_start, 
                               real_end, 
                               tag, 
                               step, 
                               writer_obj):
    """
    add code timing to tensorboard; rough way of timing my functions
    
    arguments
    ----------
    cpu_start, cpu_end : float
        > timestamps from time.time()
    
    real_start, real_end: float
        > timestamps from time.time()
    
    tag : str
        > tag to describe where time was taken
    
    step : int
        > when to record
    
    writer_obj : Tensorboard writer object
        > tensorboard object to write to
    """
    writer_obj.add_scalar(tag =f'Code Timing | {tag}/CPU+sys time', 
                          scalar_value = cpu_end - cpu_start, 
                          global_step = step)
    
    writer_obj.add_scalar(tag =f'Code Timing | {tag}/Real time', 
                          scalar_value = real_end - real_start, 
                          global_step = step)

def write_timing_file(outdir,
                      total_times):
    """
    record real and cpu times during training
    """
    # first epoch is guaranteed to have jit compilation
    real_with_jit_comp = total_times[0, 0].item()
    cpu_with_jit_comp = total_times[0, 1].item()
    
    # more timepoints use cached function
    real_cached = total_times[1:, 0].mean(axis=0).item()
    cpu_cached = total_times[1:, 1].mean(axis=0).item()
    
    with open(f'{outdir}/TIMING.txt','w') as g:
        g.write('# First epoch (with jit compilation)\n')
        g.write(f'Real:\t{real_with_jit_comp}\n')
        g.write(f'CPU:\t{cpu_with_jit_comp}\n')
        g.write(f'\n')
    
        g.write('# Average over subsequent epochs (uses cached functions) \n')
        g.write(f'Real:\t{real_cached}\n')
        g.write(f'CPU:\t{cpu_cached}\n')
        g.write(f'\n')
        
def record_postproc_time_table( already_started_timer_class,
                                writer ):
    elapsed_real_time, elapsed_cpu_sys_time = already_started_timer_class.end_timer_get_deltas()
    df = pd.DataFrame({'label': ['Real time', 'CPU+sys time'],
                       'value': [elapsed_real_time, elapsed_cpu_sys_time]})
    markdown_table = df.to_markdown()
    writer.add_text(tag = 'Code Timing | Post-training actions',
                    text_string = markdown_table,
                    global_step = 0)


###############################################################################
### Helpers for sequence shapes   #############################################  
###############################################################################  
def selective_squeeze(mat):
    """
    jnp.squeeze, but ignore batch dimension (dim0)
    """
    new_shape = tuple( [mat.shape[0]] + [s for s in mat.shape[1:] if s != 1] )
    return jnp.reshape(mat, new_shape)

def clip_by_bins(datamat, 
                 chunk_length: int = 512, 
                 padding_idx = 0):
    """
    Clip excess paddings by binning according to chunk_length
    
    For example, if chunk_length is 3, then possible places to clip include:
        > up to length 3, if longest sequence is <= 3 in length
        > up to length 6, if longest sequence is > 3 and <= 6 in length
        > up to length 9, if longest sequence is > 6 and <= 9 in length
        > etc., until maximum length of batch_seqs
    
    overall, this helps jit-compile different versions of the functions
      for different max lengths (semi-dynamic batching)
     
        
    Arguments:
    ----------
    datamat : ArrayLike
        dim 1 MUST be a length dim!!!
    
    chunk_length : int = 512
        length of the chunk
    
    padding_idx : int = 0
        padding token
    """
    # lengths
    L_max = datamat.shape[1]
    max_len_without_padding = (datamat != padding_idx).sum(axis=1).max()
    
    # determine the number of chunks
    def cond_fun(num_chunks):
        return chunk_length * num_chunks < max_len_without_padding

    def body_fun(num_chunks):
        return num_chunks + 1
    
    num_chunks = jax.lax.while_loop(cond_fun, body_fun, 1)
    length_with_all_chunks = chunk_length * num_chunks
    
    # if length_with_all_chunks is greater than max_len, 
    #   use max_len instead
    clip_to = jnp.where( length_with_all_chunks > L_max,
                         L_max,
                         length_with_all_chunks )
    return clip_to


def determine_seqlen_bin(batch,
                         chunk_length: int,
                         seq_padding_idx: int = 0):
    ### batch has 4 entries:
    ### 0.) unaligned seqs: (B, L, 2)
    ### 1.) aligned matrices: (B, L, 2)
    ### 2.) time (optional): (B,) or None
    ### 3.) dataloader idx (B,)
    unaligned_seqs = batch[0]
    batch_max_seqlen = clip_by_bins(datamat = unaligned_seqs, 
                                    chunk_length = chunk_length, 
                                    padding_idx = seq_padding_idx)
    return batch_max_seqlen

def jit_compile_determine_seqlen_bin(args):
    parted_determine_seqlen_bin = partial(determine_seqlen_bin,
                                          chunk_length = args.chunk_length, 
                                          seq_padding_idx = args.seq_padding_idx)
    jitted_determine_seqlen_bin = jax.jit(parted_determine_seqlen_bin)
    return jitted_determine_seqlen_bin

def determine_alignlen_bin(batch,
                           chunk_length: int,
                           seq_padding_idx: int = 0):
    ### batch has 4 entries:
    ### 0.) unaligned seqs: (B, L, 2)
    ### 1.) aligned matrices: (B, L, 2)
    ### 2.) time (optional): (B,) or None
    ### 3.) dataloader idx (B,)
    # use the first sequence from aligned matrix for this (gapped ancestor for 
    #   neural_pairhmm, alignment-augmented descendant for feedforward); 
    #   exclude <bos> for the clip_by_bins function
    aligned_mats_excluding_bos = batch[1][:, 1:, 0]
    
    # get length
    batch_max_alignlen = clip_by_bins(datamat = aligned_mats_excluding_bos, 
                                      chunk_length = chunk_length, 
                                      padding_idx = seq_padding_idx)
      
    # add one again, to re-include <bos>
    return (batch_max_alignlen + 1)

def jit_compile_determine_alignlen_bin(args):
    parted_determine_alignlen_bin = partial(determine_alignlen_bin,  
                                            chunk_length = args.chunk_length,
                                            seq_padding_idx = args.seq_padding_idx)
    jitted_determine_alignlen_bin = jax.jit(parted_determine_alignlen_bin)
    return jitted_determine_alignlen_bin