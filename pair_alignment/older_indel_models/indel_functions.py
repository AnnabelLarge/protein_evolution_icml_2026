
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 19:13:58 2024


Additional functions

'KM03LogTransitionMatrix',
'LG05LogTransitionMatrix',
'RS07LogTransitionMatrix',

'KM03_Ftransitions',
'LG05_Ftransitions',
'RS07_Ftransitions'
"""
import jax.numpy as jnp

smallest_float32 = jnp.finfo('float32').smallest_normal

# LG05
# Löytynoja and Goldman, 2005
# An algorithm for progressive multiple alignment of sequences with insertions.
# Proc. Natl. Acad. Sci. USA  102: 10557–10562.
def LG05_Ftransitions (lam, mu, x, y, t):
    """
    domain problem if x=y=1, but that's really unlikely
    """
    epsilon = (x + y)/2
    gamma = epsilon
    maxDelta = .49999
    # delta = jnp.minimum ( maxDelta, 1 - jnp.exp( -(lam + mu)*t/(1-gamma) ) )
    
    # epsilon cannot be zero, or this denom is undefined
    exponentiated =  -(lam + mu)*t/(1-gamma)
    delta = jnp.minimum (maxDelta, 
                         1 - jnp.exp(exponentiated) 
                         )

    Mrow = jnp.stack( [gamma + (1-gamma)*(1-2*delta), 
                       (1-gamma)*delta, 
                       (1-gamma)*delta], axis=-1 )[:, None, :] #(T, 1, 3)
    
    Irow = jnp.stack( [(1-epsilon)*(1-2*delta), 
                       epsilon + (1-epsilon)*delta, 
                       (1-epsilon)*delta], axis=-1 )[:, None, :] #(T, 1, 3)
    
    Drow = jnp.stack( [(1-epsilon)*(1-2*delta), 
                       epsilon + (1-epsilon)*delta, 
                       (1-epsilon)*delta], axis=-1 )[:, None, :] #(T, 1, 3)
    
    out = jnp.concatenate([Mrow, Irow, Drow], axis=1) #(T, 3, 3)
    
    # out = jnp.array ([[gamma + (1-gamma)*(1-2*delta), (1-gamma)*delta, (1-gamma)*delta],
    #                   [(1-epsilon)*(1-2*delta), epsilon + (1-epsilon)*delta, (1-epsilon)*delta],
    #       		    [(1-epsilon)*(1-2*delta), epsilon + (1-epsilon)*delta, (1-epsilon)*delta]]) #(3, 3, T)
    
    return out

def LG05LogTransitionMatrix (lam, mu, x, y, t_array):
    transmat = LG05_Ftransitions (lam = lam, 
                                  mu = mu, 
                                  x = x, 
                                  y = y, 
                                  t = t_array)  #(T, 3, 3)
    
    # if any position in transmat is zero, replace with smallest float
    transmat = jnp.where(transmat != 0, transmat, smallest_float32)
    logprob_transition_at_t = jnp.log(transmat) #(T, 3, 3)
    
    return logprob_transition_at_t
    

# RS07
# Redelings and Suchard, 2007
# Incorporating indel information into phylogeny estimation for rapidly emerging pathogens.
# BMC Evol. Biol.  7: 40.
def RS07_Ftransitions (lam, mu, x, y, t):
    """
    domain problem if x=y=1, but that's really unlikely
    """
    epsilon = (x + y)/2
    maxDelta = .49999
    # delta = jnp.minimum (maxDelta, 1 / (1 + 1 / (1 - jnp.exp(-(lam + mu)*t/(1-epsilon)))))
    
    # epsilon cannot be zero, or this denom is undefined
    exponentiated = -(lam + mu)*t/(1-epsilon)
    delta = jnp.minimum (maxDelta, 
                         1 / (1 + 1 / (1 - jnp.exp( exponentiated )))
                         )
    
    Mrow = jnp.stack( [epsilon + (1-epsilon)*(1-2*delta), 
                       (1-epsilon)*delta, 
                       (1-epsilon)*delta], axis = -1 )[:, None, :] #(T, 1, 3)
    
    Irow = jnp.stack( [(1-epsilon)*(1-2*delta), 
                       epsilon + (1-epsilon)*delta, 
                       (1-epsilon)*delta], axis = -1 )[:, None, :] #(T, 1, 3)
    
    Drow = jnp.stack( [(1-epsilon)*(1-2*delta), 
                       epsilon + (1-epsilon)*delta, 
                       (1-epsilon)*delta], axis = -1 )[:, None, :] #(T, 1, 3)
    
    out = jnp.concatenate([Mrow, Irow, Drow], axis=1) #(T, 3, 3)
    
    # out = jnp.array ([[epsilon + (1-epsilon)*(1-2*delta), (1-epsilon)*delta, (1-epsilon)*delta],
    #   		          [(1-epsilon)*(1-2*delta), epsilon + (1-epsilon)*delta, (1-epsilon)*delta],
    #   		          [(1-epsilon)*(1-2*delta), epsilon + (1-epsilon)*delta, (1-epsilon)*delta]]) #(3, 3, T)
    
    return out

def RS07LogTransitionMatrix (lam, mu, x, y, t_array):
    transmat = RS07_Ftransitions (lam = lam, 
                                  mu = mu, 
                                  x = x, 
                                  y = y, 
                                  t = t_array)  #(T, 3, 3)
    
    # if any position in transmat is zero, replace with smallest float
    transmat = jnp.where(transmat != 0, transmat, smallest_float32)
    logprob_transition_at_t = jnp.log(transmat) #(T, 3, 3)
    
    return logprob_transition_at_t
    

# KM03
# Knudsen and Miyamoto, 2003
# Sequence Alignments and Pair Hidden Markov Models Using Evolutionary History
# J. Mol. Biol. 333:2, 453-460.
def KM03_Ftransitions (lam, mu, x, y, t):
    """
    domain problem if x=y=1, but that's really unlikely; probably don't have to
      worry about that
    """
    r = (lam + mu) / 2
    a = (x + y) / 2
    
    Pid = 1 - jnp.exp(-2*r*t)
    Pid_prime = 1 - (1 - jnp.exp(-2*r*t)) / (2*r*t)
    
    T00 = 1 - Pid*(1-Pid_prime*(1-a)/(4+4*a))
    T01 = (1-T00)/2
    T02 = T01
    
    E10 = 1-a + Pid_prime*a*(1-a)/(2+2*a) - Pid*(7-7*a)/8
    E11 = a + Pid_prime*a*a/(1-a*a) + Pid*(1-a)/2 # domain problem if a*a=1
    E12 = Pid_prime*a*a/(2+2*a) + Pid*(3-3*a)/8
    E1 = 1 + Pid_prime*a/(2-2*a) # domain problem if a = 1
    
    T10 = E10/E1 #(T,)
    T11 = E11/E1 #(T,)
    T12 = E12/E1 #(T,)
    T20 = T10 #(T,)
    T22 = T11 #(T,)
    T21 = T12 #(T,)
    
    Mrow = jnp.stack( [T00, T01, T02], axis = -1 )[:, None, :] #(T, 1, 3)
    Irow = jnp.stack( [T10, T11, T12], axis = -1 )[:, None, :] #(T, 1, 3)
    Drow = jnp.stack( [T20, T21, T22], axis = -1 )[:, None, :] #(T, 1, 3)
    
    out = jnp.concatenate([Mrow, Irow, Drow], axis=1) #(T, 3, 3)
    
    # out = jnp.array ([[T00, T01, T02],
    #                   [T10, T11, T12],
    #                   [T20, T21, T22]]) #(3, 3, T)
    
    return out

def KM03LogTransitionMatrix (lam, mu, x, y, t_array):
    transmat = KM03_Ftransitions (lam = lam, 
                                  mu = mu, 
                                  x = x, 
                                  y = y, 
                                  t = t_array)  #(T, 3, 3)
    
    # if any position in transmat is zero, replace with smallest float
    transmat = jnp.where(transmat != 0, transmat, smallest_float32)
    logprob_transition_at_t = jnp.log(transmat) #(T, 3, 3)
    
    return logprob_transition_at_t

