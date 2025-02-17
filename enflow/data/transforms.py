import torch
from torch_geometric.transforms import BaseTransform

from ..units.conversion import femtosecond_to_lj, m_to_lj
from ..units.conversion import ang_to_lj, amu_to_lj, kelvin_to_lj

import numpy as np
from rdkit import Chem

class ConvertPositions(BaseTransform):
    def __init__(self, input_unit):
        if input_unit == 'ang':
            self.factor = 1e-10
        elif input_unit == 'nm':
            self.factor = 1e-9

    def forward(self, data):
        data.pos = m_to_lj(data.pos*self.factor)
        return data
        
class RandomizeVelocity(BaseTransform):
    def __init__(self, temp):
        self.kBT = kelvin_to_lj(temp)
        self.pse = Chem.GetPeriodicTable()

    def forward(self, data):        
        m = torch.tensor([amu_to_lj(self.pse.GetAtomicWeight(i)) for i in data.z])

        a = np.sqrt(self.kBT/m)
        velx = (torch.rand(12).sum()-6)*a
        vely = (torch.rand(12).sum()-6)*a
        velz = (torch.rand(12).sum()-6)*a
        
        data.vel = torch.stack([velx, vely, velz], dim=1)
        
        return data
