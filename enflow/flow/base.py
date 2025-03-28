import math
import torch
import numpy as np
            
class BaseFlow(torch.nn.Module):
    def __init__(self, network, n_iter, dt, r_cut, box, dequant_scale=1, precision=60):
        super().__init__()
        self.n_iter = n_iter
        self.networks = torch.nn.ModuleList(self.make_networks(network))
        self.dt = dt
        self.dt_2 = 0.5*dt
        self.r_cut = r_cut
        self.box = box
        self.dequant_scale = dequant_scale
        self.precision =  precision
        self.to(torch.double)
    
    def dequantize(self, z):
        z = z.to(torch.float64)
        return z + self.dequant_scale*torch.rand_like(z).detach()
    
    def quantize(self, z): return torch.floor(z)
    
    def to_fixed(self, x):
        # decimal = 0, 1, , 62
        return x#(x * 2**self.precision).type(torch.int64)

    def from_fixed(self, x):
        # decimal = 0, 1, , 62
        return x#x.type(torch.float64) * 2**-self.precision  
                
    def forward(self, data):
        pass

    def reverse(self, data):
        pass