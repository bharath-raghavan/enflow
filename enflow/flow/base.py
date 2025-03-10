import math
import torch
import numpy as np
            
class BaseFlow(torch.nn.Module):
    def __init__(self, network, n_iter, dt, r_cut, kBT, box, partition_func=10, softening=0, dequant_scale=1, device='cpu'):
        super().__init__()
        self.n_iter = n_iter
        self.networks = torch.nn.ModuleList(self.make_networks(network))
        self.dt = dt
        self.dt_2 = 0.5*dt
        self.r_cut = r_cut
        self.kBT = kBT
        self.z_lj = partition_func
        self.box = box
        self.softening = softening
        self.dequant_scale = dequant_scale
        self.to(device)
    
    def get_lj_potential(self, data):
        H = 0
        for mol in data:
            dist_sq = torch.triu((mol.pos.unsqueeze(1) - mol.pos).pow(2).sum(dim=2))
            r_sq = dist_sq[dist_sq != 0] + self.softening
            r_6 = r_sq.pow(3)
            r_12 = r_6.pow(2)
            H += 4*(1/r_12 - 1/r_6).sum()
        return H 

    def nll(self, out, ldj):
        H = self.get_lj_potential(out) + ((out.vel**2).sum() + (out.h**2).sum() + (out.g**2).sum())*0.5
        logZ = - out.num_atoms*( math.log(self.z_lj) - (1.5+out.h.shape[1])*math.log(2*math.pi*self.kBT) )
        log_px = - H/self.kBT + logZ + ldj
        return -log_px/out.num_mols 
    
    def dequantize(self, z):
        z = z.to(torch.float64)
        return z + self.dequant_scale*torch.rand_like(z).detach()
    
    def quantize(self, z): return torch.floor(z)
        
    def forward(self, data):
        pass

    def reverse(self, data):
        pass