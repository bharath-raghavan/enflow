import torch
from torch import nn
import math

from .units.constants import kBT
from .units.conversion import femtosecond_to_lj, ang_to_lj
    
class EGCL(nn.Module):
    """Equivarent Graph Convolution Layer.
    Adapted from Ref: xxx
    """


    def __init__(self, input_nf, output_nf, hidden_nf, act_fn=nn.ReLU(), coords_weight=1.0, attention=False, clamp=False, norm_diff=False, tanh=False):
        super().__init__()
        input_edge = input_nf * 2
        self.coords_weight = coords_weight
        self.attention = attention
        self.norm_diff = norm_diff
        self.tanh = tanh
        edge_coords_nf = 1


        self.edge_nn = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)

        self.node_nn = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))

        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        self.clamp = clamp
        coord_nn = []
        coord_nn.append(nn.Linear(hidden_nf, hidden_nf))
        coord_nn.append(act_fn)
        coord_nn.append(layer)
        if self.tanh:
            coord_nn.append(nn.Tanh())
            self.coords_range = nn.Parameter(torch.ones(1))*3
        self.coord_nn = nn.Sequential(*coord_nn)


        if self.attention:
            self.att_nn = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())
            
        self.norm_diff = norm_diff
        self.vel_scaling_nn = nn.Sequential(
            nn.Linear(input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, 1))

    def edge_model(self, source, target, radial):
        out = torch.cat([source, target, radial], dim=1)
        out = self.edge_nn(out)
        if self.attention:
            att_val = self.att_nn(out)
            out = out * att_val
        return out

    def node_model(self, x, edge_index, edge_attr):
        row, col = edge_index
        agg = self.unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        agg = torch.cat([x, agg], dim=1)
        out = self.node_nn(agg)
        return out

    def force_model(self, coord, edge_index, coord_diff, edge_feat):
        row, col = edge_index
        trans = coord_diff * self.coord_nn(edge_feat)
        trans = torch.clamp(trans, min=-100, max=100) #This is never activated but just in case it case it explosed it may save the train
        agg = self.unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        return agg*self.coords_weight


    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum((coord_diff)**2, 1).unsqueeze(1)

        if self.norm_diff:
            norm = torch.sqrt(radial) + 1
            coord_diff = coord_diff/(norm)

        return radial, coord_diff

    def unsorted_segment_sum(self, data, segment_ids, num_segments):
        """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`."""
        result_shape = (num_segments, data.size(1))
        result = data.new_full(result_shape, 0)  # Init empty result tensor.
        segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
        result.scatter_add_(0, segment_ids, data)
        return result


    def unsorted_segment_mean(self, data, segment_ids, num_segments):
        result_shape = (num_segments, data.size(1))
        segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
        result = data.new_full(result_shape, 0)  # Init empty result tensor.
        count = data.new_full(result_shape, 0)
        result.scatter_add_(0, segment_ids, data)
        count.scatter_add_(0, segment_ids, torch.ones_like(data))
        return result / count.clamp(min=1)
        
    def forward(self, h, edge_index, coord):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)
        edge_attr = self.edge_model(h[row], h[col], radial)
        return self.vel_scaling_nn(h),\
            self.force_model(coord, edge_index, coord_diff, edge_attr),\
            self.node_model(h, edge_index, edge_attr)

class ENFlow(nn.Module):
    def __init__(self, node_nf, n_iter, dt, dh, r_cut, device='cpu', act_fn=nn.SiLU(), coords_weight=1.0, norm_diff=False, tanh=False):
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
        # Create prior distribution for final latent space of h
        self.h_prior = torch.distributions.normal.Normal(loc=0.0, scale=1.0)

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
        print(data.pos)
        data.vel = torch.rand(data.pos.shape)
        for network in self.networks:
            edges = self.get_edges(data)
            scaling_factor, force, node_model = network(data.h, edges, data.pos)
            data.vel = torch.exp(scaling_factor) * data.vel + force*self.dt
            data.pos = data.pos + data.vel*self.dt
            data.g = data.g + node_model*self.dh
            data.h = data.h + data.g*self.dh
            
            ldj += scaling_factor
        
        print(data.pos)
        return data, ldj.sum()

    def reverse(self, data):        
        for network in reversed(self.networks):
            data.h -= data.g*self.dh
            edges = self.get_edges(data)
            scaling_factor, force, node_model = network(data.h, edges, data.pos)
            data.g -= node_model*self.dh
            data.pos -= data.vel*self.dt
            data.vel = (data.vel - force*self.dt)/torch.exp(scaling_factor)
        return data

    def lj_hamiltonian(self, mol):
        ke = (mol.vel**2).sum()/2
        r_6 = (mol.pos.unsqueeze(1) - mol.pos).pow(2).sum(dim=2).pow(3)
        r_12 = r_6.pow(2)
        pe = 4*(torch.triu(1/r_12 - 1/r_6).nan_to_num(0).sum())
        return pe + ke


    def nll(self, out, ldj):
        H = 0
        for i in range(len(out)):
            H += self.lj_hamiltonian(out[i])
        
        z = torch.cat([out.h, out.g])
        log_pz = -H/kBT + (z*z).sum()*0.5
        log_px = ldj + log_pz
        N_batch = len(out)

        return -log_px/N_batch