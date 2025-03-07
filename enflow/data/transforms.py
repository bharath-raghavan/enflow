import torch

from ..utils.conversion import m_to_lj, kelvin_to_lj
from ..utils.helpers import get_box

import numpy as np
from rdkit import Chem
from scipy.special import erf
from scipy.interpolate import interp1d as interp

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data

class ConvertPositionsFrom:
    def __init__(self, input_unit):
        if input_unit == 'ang':
            self.factor = 1e-10
        elif input_unit == 'nm':
            self.factor = 1e-9

    def __call__(self, data):
        data.pos = m_to_lj(data.pos*self.factor)
        return data
        
class Center:
    def __init__(self):
        pass

    def __call__(self, data):
        pos = data.pos
        data.pos = pos - pos.mean(dim=0, keepdim=True)
        return data

class RandomizeVelocity:
    def __init__(self, temp):
        self.kBT = kelvin_to_lj(temp)
        self.pse = Chem.GetPeriodicTable()
        
    def generate_velocities(self, n):
        """ generate a set of velocity vectors in 3D from the MB inverse CDF function """
        """ Adapted from: https://github.com/tpogden/quantum-python-lectures/blob/master/11_Monte-Carlo-Maxwell-Boltzmann-Distributions.ipynb """
        
        # Cumulative Distribution function of the Maxwell-Boltzmann speed distribution
        m = 1
        a = np.sqrt(self.kBT/m)
        MB_CDF = lambda v : erf(v/(np.sqrt(2)*a)) - np.sqrt(2/np.pi)* v* np.exp(-v**2/(2*a**2))/a
        v = np.arange(0,2500,0.1)
        cdf = MB_CDF(v)

        #create interpolation function to CDF
        inv_cdf = interp(cdf,v)
                                 
        rand_nums = np.random.random(n)
        speeds = inv_cdf(rand_nums)
    
        theta = np.arccos(np.random.uniform(-1,1,n))
        phi = np.random.uniform(0,2*np.pi,n)
    
        # convert to cartesian units
        vx = speeds * np.sin(theta) * np.cos(phi) 
        vy = speeds * np.sin(theta) * np.sin(phi)
        vz = speeds * np.cos(theta)
    
        return np.stack([vx, vy, vz], axis=1, dtype=np.float64)

    def __call__(self, data):
        data.vel = torch.from_numpy(self.generate_velocities(data.N))
        return data
        

