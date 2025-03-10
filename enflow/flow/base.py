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

    def nll(self, out, ldj):
        z = torch.cat([out.h, out.g])
        log_pz = - out.get_lj_hamiltonian(self.softening)/self.kBT - out.num_atoms*self.z_lj - (z*z).sum()*0.5 - math.log(math.sqrt(2*math.pi))
        log_px = ldj + log_pz
        return -log_px/out.num_mols 
    
    def dequantize(self, z):
        z = z.to(torch.float64)
        return z + self.dequant_scale*torch.rand_like(z).detach(), 0
    
    def quantize(self, z): return torch.floor(z)
    
    #def dequantize(self, z):
    #    # Transform discrete values to continuous volumes
    #    z = z.to(torch.float64)
    #    z = z + torch.rand_like(z).detach()
    #    z = z / self.quants
    #    ldj = -np.log(self.quants) * np.prod(z.shape[1:])
    
    #    z = z * (1 - self.alpha) + 0.5 * self.alpha  # Scale to prevent boundaries 0 and 1
    #    ldj += np.log(1 - self.alpha) * np.prod(z.shape[1:])
    #    ldj += (-torch.log(z) - torch.log(1-z)).sum()
    #    z = torch.log(z) - torch.log(1-z)
        
    #    return z, ldj
        
    #def quantize(self, z):
    #    z = torch.sigmoid(z)
    #    z = (z - 0.5 * self.alpha) / (1 - self.alpha)
    #    z = z * self.quants
    #    return torch.floor(z).clamp(min=0, max=self.quants-1).to(torch.int32)
        
    def forward(self, data):
        pass

    def reverse(self, data):
        pass