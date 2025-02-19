import torch
from torch import nn
import math
from .egcl import EGCL

class ENFlow(nn.Module):
    def __init__(self, node_nf, n_iter, dt, r_cut, temp, partition_func=10, softening=0, dh=1, device='cpu', act_fn=nn.SiLU(), coords_weight=1.0, norm_diff=False, tanh=False):
        super().__init__()
        self.device = device
        networks = []
        for i in range(n_iter):
            networks.append( EGCL(node_nf, node_nf, node_nf, act_fn=act_fn, coords_weight=coords_weight, norm_diff=norm_diff, tanh=tanh) )
        self.networks = nn.ModuleList(networks)
        self.to(self.device)
        self.dt = dt
        self.dh = dh
        self.r_cut = r_cut
        self.kBT = temp
        self.z_lj = partition_func
        self.softening = softening
        
    def get_edges(self, data):
        # get nieghbour list
        r_sq = self.r_cut*self.r_cut

        edge_index = torch.empty((2, 0), dtype=torch.int)

        N_cnt = 0
        for i in range(len(data)):
            mol = data[i]
            dist_sq = (mol.pos.unsqueeze(1) - mol.pos).pow(2).sum(dim=2)
            edge_index_mol = (dist_sq < r_sq).nonzero() + N_cnt
            edge_index_mol = edge_index_mol[torch.nonzero(edge_index_mol[:, 0] - edge_index_mol[:, 1])].squeeze(1)
            edge_index = torch.cat((edge_index, edge_index_mol.T), dim=1)
            N_cnt += mol.N.item()

        return edge_index
        
    def forward(self, data):
        ldj = 0
        for network in self.networks:
            edges = self.get_edges(data)
            scaling_factor, force, h_force = network(data.h, edges, data.pos)
            data.vel = torch.exp(scaling_factor) * data.vel + force*self.dt
            data.pos = data.pos + data.vel*self.dt
            data.g = data.g + h_force*self.dh
            data.h = data.h + data.g*self.dh
            ldj += scaling_factor
        return data, ldj.sum()

    def reverse(self, data):
        for network in reversed(self.networks):
            data.h = data.h - data.g*self.dh
            data.pos = data.pos - data.vel*self.dt
            edges = self.get_edges(data)
            scaling_factor, force, h_force = network(data.h, edges, data.pos)
            data.g = data.g - h_force*self.dh
            data.vel = (data.vel - force*self.dt)/torch.exp(scaling_factor)
        return data

    def lj_hamiltonian(self, mol):
        ke = (mol.vel**2).sum()/2
        dist_sq = torch.triu((mol.pos.unsqueeze(1) - mol.pos).pow(2).sum(dim=2))
        r_sq = dist_sq[dist_sq != 0] + self.softening
        r_6 = r_sq.pow(3)
        r_12 = r_6.pow(2)
        pe = 4*(1/r_12 - 1/r_6).sum()
        return pe + ke

    def nll(self, out, ldj):
        H = 0
        for i in range(len(out)):
            H += self.lj_hamiltonian(out[i])

        z = torch.cat([out.h, out.g])
        log_pz = - H/self.kBT - out.N.sum()*self.z_lj - (z*z).sum()*0.5 - math.log(math.sqrt(2*math.pi))
        log_px = ldj + log_pz
        N_batch = len(out)
        return -log_px/N_batch 