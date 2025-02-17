import math

from .constants import *

def m_to_lj(x):
    return x/sigma
    
def ang_to_lj(x):
    return m_to_lj(x*1e-10) 
    
def kelvin_to_lj(T):
    return T * kB/eps

def amu_to_lj(m):
    return m/M
    
def second_to_lj(t):
    return t*math.sqrt(eps/M)/sigma
    
def femtosecond_to_lj(t):
    return second_to_lj(t*1e-15)
