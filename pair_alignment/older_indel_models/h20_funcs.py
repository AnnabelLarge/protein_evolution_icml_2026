#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 16:27:33 2024


Functions to calculate alignment-depending log likelihoods 
for the H20 model

functions of interest:
-----------------------
transitionMatrix
logTransitionMatrix
vmappedLogTransitionMatrix

"""
import jax
from jax import numpy as jnp
import diffrax
from diffrax import (diffeqsolve, ODETerm, Dopri5, PIDController, 
                     ConstantStepSize, SaveAt)

from functools import partial

smallest_float32 = jnp.finfo('float32').smallest_normal


##############################################
### FINDING A, B, U, Q                       #
### these are used in finding the small time #
### transition matrix functions              #
### (smallTimeTransitionMatrix)              #
##############################################
# calculate derivatives of (a,b,u,q)
def derivs (t, counts, indelParams):
    lam,mu,x,y = indelParams
    a,b,u,q = counts
    L = lm (t, lam, x)
    M = lm (t, mu, y)
    num = mu * (b*M + q*(1.-M))
    unsafe_denom = M*(1.-y) + L*q*y + L*M*(y*(1.+b-q)-1.)
    denom = jnp.where (unsafe_denom > 0., unsafe_denom, 1.)   # avoid NaN gradient at zero
    one_minus_m = jnp.where (M < 1., 1. - M, smallest_float32)   # avoid NaN gradient at zero
    
    if_true = jnp.array( ( (mu*b*u*L*M*(1.-y)/denom - (lam+mu)*a,
                            -b*num*L/denom + lam*(1.-b),
                            -u*num*L/denom + lam*a,
                            ((M*(1.-L)-q*L*(1.-M))*num/denom - q*lam/(1.-y))/one_minus_m ) ) ) #(4, 1)
    
    if_false = jnp.array( (-lam-mu,
                           lam,
                           lam, 
                           jnp.zeros_like(lam)) )  #(4, 1)
    
    return jnp.where( unsafe_denom > 0., 
                      jnp.squeeze( if_true ), 
                      jnp.squeeze( if_false ) )  #(4,)

# calculate counts (a,b,u,q) by numerical integration
def initCounts(indelParams):
    return jnp.array ((1., 0., 0., 0.))
    
def integrateCounts (t, indelParams, step = None, rtol = None, atol = None, **kwargs):
    term = ODETerm(derivs)
    solver = Dopri5()
    if step is None and rtol is None and atol is None:
        raise Exception ("please specify step, rtol, or atol")
    if step is not None:
        stepsize_controller = ConstantStepSize()
    else:
        stepsize_controller = PIDController (rtol, atol)
    y0 = initCounts(indelParams)
    sol = diffeqsolve (term, solver, 0., t, step, y0, args=indelParams,
                       stepsize_controller=stepsize_controller,
                       **kwargs)
    return sol.ys[-1]



################################
### FUNCTIONS FOR DIFF EQS     #
################################
### calculate L, M
def lm (t, rate, prob):
    num = -rate * t
    denom = 1. - prob
    frac = (num/denom)
    return jnp.exp (frac)

def indels (t, rate, prob):
    return 1. / lm(t,rate,prob) - 1.

# test whether time is past threshold of alignment signal being undetectable
def alignmentIsProbablyUndetectable (t, indelParams, alphabetSize):
    lam,mu,x,y = indelParams
    expectedMatchRunLength = 1. / (1. - jnp.exp(-mu*t))
    expectedInsertions = indels(t,lam,x)
    expectedDeletions = indels(t,mu,y)
    kappa = 2.
    return jnp.where (t > 0.,
                      ((expectedInsertions + 1) * (expectedDeletions + 1)) > kappa * (alphabetSize ** expectedMatchRunLength),
                      False)

# convert counts (a,b,u,q) to transition matrix ((a,b,c),(f,g,h),(p,q,r))
def smallTimeTransitionMatrix (t, indelParams, **kwargs):
    """
    t is (1,) (vmap happens outside of this function)
    
    
    matrix output should be:
    
    [[a,                     b,                             1-a-b],
     [u*L/one_minus_L,       1-(b+q*(1-M)/M)*L/one_minus_L, (b+q*(1-M)/M-u)*L/one_minus_L],
     [(1-a-u)*M/one_minus_M, q,                             1-q-(1-a-u)*M/one_minus_M]])
    """
    lam,mu,x,y = indelParams
    step = None
    rtol = kwargs.get('rtol', None)
    atol = kwargs.get('atol', None)
    
    # run integrateCounts for every mixture model
    out = integrateCounts( t,
                           indelParams = indelParams,
                           step = step,
                           rtol = rtol,
                           atol = atol ) #(4,)
    
    # unpack outputs; order of outputs should be: a, b, u, q
    a = out[0][None] #(1,)
    b = out[1][None] #(1,)
    u = out[2][None] #(1,)
    q = out[3][None] #(1,)
    
    L = lm(t,lam,x)
    M = lm(t,mu,y)
    one_minus_L = jnp.where (L < 1., 1. - L, smallest_float32)   # avoid NaN gradient at zero
    one_minus_M = jnp.where (M < 1., 1. - M, smallest_float32)   # avoid NaN gradient at zero
    
    # row 1: M -> (M, I, D)
    mat_Mrow = jnp.array( [a, b, 1-a-b] )[None, :, 0] #(1, 3)
    
    # row 2:  I -> (M, I, D)
    mat_f = u*L/one_minus_L
    mat_g = 1-(b+q*(1-M)/M)*L/one_minus_L
    mat_h = (b+q*(1-M)/M-u)*L/one_minus_L
    mat_Irow = jnp.array( [mat_f, mat_g, mat_h] )[None, :, 0] #(1, 3)
    
    # row 3:  D -> (M, I, D)
    mat_p = (1-a-u)*M/one_minus_M
    mat_q = q
    mat_r = 1-q-(1-a-u)*M/one_minus_M
    mat_Drow = jnp.array( [mat_p, mat_q, mat_r] )[None, :, 0] #(1, 3)
    
    # concatenate all and output 
    out_matrix = jnp.concatenate([mat_Mrow, mat_Irow, mat_Drow], axis=0) #(3, 3)
    return out_matrix
    

# get limiting transition matrix for large times
def largeTimeTransitionMatrix (t, indelParams):
    """
    t is (1,) (vmap happens outside of this function)
    
    
    matrix output should be:
        
    [[(1-g)*(1-r), g, (1-g)*r],
     [(1-g)*(1-r), g, (1-g)*r],
     [(1-r),       0, r]]
    """
    lam,mu,x,y = indelParams
    g = 1. - lm(t,lam,x) 
    r = 1. - lm(t,mu,y)  
    
    # row 1: M -> (M, I, D)
    mat_a = (1-g)*(1-r)
    mat_b = g
    mat_c = (1-g)*r
    mat_Mrow = jnp.array( [mat_a, mat_b, mat_c] )[None, :, 0] #(1,3)
    
    # row 2:  I -> (M, I, D)
    mat_f = (1-g)*(1-r)
    mat_g = g
    mat_h = (1-g)*r
    mat_Irow = jnp.array( [mat_f, mat_g, mat_h] )[None, :, 0] #(1,3)
    
    # row 3:  D -> (M, I, D)
    mat_Drow = jnp.array( [ (1-r), jnp.zeros_like(r), r] )[None, :, 0] #(1,3)
    
    # concatenate all and output
    out_matrix = jnp.concatenate([mat_Mrow, mat_Irow, mat_Drow], axis=0) #(3, 3)
    return out_matrix
    


#####################################
### TRANSITION MATRIX FUNCTIONS     #
#####################################
def transitionMatrix (t, indelParams, alphabetSize=20, **kwargs):
    # assume t is always greater than 0
    lam,mu,x,y = indelParams
    
    if_true = largeTimeTransitionMatrix(t,
                                        indelParams)
    if_false = smallTimeTransitionMatrix(t,
                                         indelParams,
                                         rtol=1e-3,
                                         atol=1e-6,
                                         step=None)
    
    return jnp.where( alignmentIsProbablyUndetectable(t,indelParams,alphabetSize),
                      if_true,
                      if_false ) #(3, 3)

def logTransitionMatrix (t, indelParams, alphabetSize=20, **kwargs):
    transmat = transitionMatrix( t, indelParams, alphabetSize, **kwargs ) #(T, 3, 3)
    
    # if any position in transmat is zero, replace with smallest float
    transmat = jnp.where(transmat != 0, transmat, smallest_float32) #(T, 3, 3)
    logprob_transition_at_t = jnp.log(transmat) #(T, 3, 3)
        
    return logprob_transition_at_t #(T, 3, 3)

def vmappedLogTransitionMatrix( lam, mu, x, y, t_array ):
    indelParams = lam, mu, x, y
    parted = partial( logTransitionMatrix,
                      indelParams = indelParams )
    vmapped_fn = jax.vmap( parted )
    return vmapped_fn( t_array ) #(T, 3, 3)
    



