import torch
from torch import nn
import math
from .egcl import EGCL

class ENFlow(nn.Module):
    def __init__(self, node_nf, hidden_nf, n_iter, dt, r_cut, kBT, partition_func=10, softening=0, device='cpu', act_fn=nn.SiLU(), coords_weight=1.0, norm_diff=False, tanh=False):
        super().__init__()
        self.device = device
        networks = []
        self.node_nf = node_nf
        for i in range(n_iter):
            networks.append( EGCL(node_nf, node_nf, hidden_nf, act_fn=act_fn, coords_weight=coords_weight, norm_diff=norm_diff, tanh=tanh) )
        self.networks = nn.ModuleList(networks)
        self.to(self.device)
        self.dt = dt
        self.r_cut = r_cut
        self.kBT = kBT
        self.z_lj = partition_func
        self.softening = softening
    
    def forward(self, data):
        ldj = 0
        for network in self.networks:
            edges = data.get_edges(self.r_cut)
            scaling_factor, force, h_force = network(data.h, edges, data.pos)
            data.vel = torch.exp(scaling_factor) * data.vel + force*self.dt
            data.pos = data.pos + data.vel*self.dt
            data.g = data.g + h_force
            data.h = data.h + data.g
            ldj += scaling_factor
        return data, ldj.sum()

    def reverse(self, data):
        for network in reversed(self.networks):
            data.h = data.h - data.g
            data.pos = data.pos - data.vel*self.dt
            edges = data.get_edges(self.r_cut)
            scaling_factor, force, h_force = network(data.h, edges, data.pos)
            data.g = data.g - h_force
            data.vel = (data.vel - force*self.dt)/torch.exp(scaling_factor)
        return data

    def nll(self, out, ldj):
        z = torch.cat([out.h, out.g])
        log_pz = - out.get_lj_hamiltonian(self.softening)/self.kBT - out.num_atoms*self.z_lj - (z*z).sum()*0.5 - math.log(math.sqrt(2*math.pi))
        log_px = - ldj + log_pz
        return -log_px/out.num_mols 