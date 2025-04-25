import os
from abc import ABC, abstractmethod
import torch
from enflow.utils.helpers import get_box

class Data:
    def __init__(self, z=None, h=None, g=None, pos=None, vel=None, N=None, label=None, device='cpu'):
        self.z = z
        self.h = h
        self.g = g
        self.pos = pos
        self.vel = vel
        self.N = N
        self.label = label
        self.device = device
    
    def get_mol(self, i):
        if self.N.ndim == 0:
            return self
        start_id = self.N[:i].sum()
        end_id = self.N[i].item()+start_id
        return Data(
                z=self.z[start_id:end_id],
                h=self.h[start_id:end_id,:],
                g=self.g[start_id:end_id,:],
                pos=self.pos[start_id:end_id,:],
                vel=self.vel[start_id:end_id,:],
                N=self.N[i],
                label=self.label[start_id:end_id],
                device=self.device
            )
    
    @property      
    def num_atoms(self):
        return self.N.sum().item()
        
    @property      
    def num_mols(self):
        if self.N.ndim == 0:
            return 1
        else:
            return len(self.N)
        
    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.num_mols:
            raise StopIteration
        else:
            mol = self.get_mol(self.i)
            self.i += 1
            return mol
    
    def to(self, device):
        h = self.h.to(device)
        g = self.g.to(device)
        pos = self.pos.to(device)
        vel = self.vel.to(device)
        N = self.N.to(device)
        
        return Data(
                z=self.z,
                h=h,
                g=g,
                pos=pos,
                vel=vel,
                N=N,
                label=self.label,
                device=device
            )
    
    def pbc(self, box, reverse=False):
        def get_push(coord):
            box_len = box[coord]
            return (self.pos[:,coord]/box_len).round()*box_len

        self.pos = self.pos - torch.stack((get_push(0), get_push(1), get_push(2)), dim=1)
    
    def get_edges(self, r_cut):
        # get nieghbour list
        r_sq = r_cut*r_cut

        edge_index = torch.empty((2, 0), dtype=torch.int, device=self.device)

        N_cnt = 0
        for mol in self:
            dist_sq = (mol.pos.unsqueeze(1) - mol.pos).pow(2).sum(dim=2)
            edge_index_mol = (dist_sq < r_sq).nonzero() + N_cnt
            edge_index_mol = edge_index_mol[torch.nonzero(edge_index_mol[:, 0] - edge_index_mol[:, 1])].squeeze(1)
            edge_index = torch.cat((edge_index, edge_index_mol.T), dim=1)
            N_cnt += mol.num_atoms

        return edge_index       
        
class DataLoader(torch.utils.data.DataLoader):
    def __init__(
        self,
        dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        **kwargs,
    ):

        super().__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=self.collater,
            **kwargs,
        )
        
    def collater(self, dataset):
        
        return Data(
                z=[d.z for d in dataset],
                h=torch.cat([d.h for d in dataset]),
                g=torch.cat([d.g for d in dataset]),
                pos=torch.cat([d.pos for d in dataset]),
                vel=torch.cat([d.vel for d in dataset]),
                N=torch.tensor([d.N for d in dataset]),
                label=[d.label for d in dataset]
            )


class BaseDataset(torch.utils.data.Dataset, ABC):
    def __init__(self, **input_params):
        if 'transform' in input_params:
            self.transform = input_params['transform']
            input_params.pop('transform')
        else:
            self.transform = None
        
        self.data_list = []
        
        if 'processed_file' in input_params:
            processed_file = input_params['processed_file']
            input_params.pop('processed_file')
            
            if os.path.exists(processed_file):
                self.data_list = torch.load(processed_file, weights_only=False)
            else:
                self.process(**input_params)
                torch.save(self.data_list, processed_file)
        else:
            self.process(**input_params)
    
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]
    
    @property  
    def node_nf(self):
        return self.data_list[0].h.shape[1]
    
    @property  
    def num_atoms_per_mol(self):
        return self.data_list[0].N   
        
    def append(self, z, h, pos, vel, N, label):
        self.N_prev = 0
        
        data = Data(
            z=z,
            h=h,
            g=torch.normal(0, 1, size=h.shape, dtype=torch.float64),
            pos=pos,
            vel=vel,
            N=N,
            label=label
        )
        
        if self.N_prev != 0: # check that all Ns same
            if N != self.N_prev: print("error!")
        
        self.N_prev = N
    
        if self.transform:
            self.data_list.append(self.transform(data))
        else:
            self.data_list.append(data)
        
    @abstractmethod
    def process(self, **input_params):
        pass