import math
import torch
import numpy as np
            
class BaseFlow(torch.nn.Module):
    def __init__(self, network, n_iter, dt, dequant_scale=1):
        super().__init__()
        self.n_iter = n_iter
        self.networks = torch.nn.ModuleList(self.make_networks(network))
        self.dt = dt
        self.dt_2 = 0.5*dt
        self.dequant_scale = dequant_scale
        self.to(torch.double)
    
    def dequantize(self, z):
        z = z.to(torch.float64)
        return z + self.dequant_scale*torch.rand_like(z).detach()
    
    def quantize(self, z): return torch.floor(z)
        
    def forward(self, data):
        pass

    def reverse(self, data):
        pass