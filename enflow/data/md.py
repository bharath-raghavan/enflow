import torch
from .base import BaseDataset, Data
import MDAnalysis
from tqdm import tqdm
from ..utils.constants import atom_types
from ..utils.helpers import one_hot, ELEMENTS, get_element

class MDDataset(BaseDataset):
    
    def process_traj(self, u):
        for frame, ts in enumerate(u.trajectory):
            z = [get_element(a.element, a.mass) for a in u.atoms]
            #type_idx = [atom_types[i] for i in z]
            #h = one_hot(torch.tensor(type_idx), num_classes=len(atom_types), dtype=torch.float64)

            #self.append(
            #    z=z,
            #    h=h,
            #    g=torch.normal(0, 1, size=h.shape),
            #    pos=torch.tensor(u.atoms.positions, dtype=torch.float64),
            #    vel=torch.tensor(u.atoms.velocities, dtype=torch.float64),
            #    N=len(u.atoms),
            #    label=traj + ' frame: ' + str(frame)
            #)
            
            # below is nonesense for the lj dataset
            N=len(u.atoms)
            node_nf = 5
            pos=torch.tensor(u.atoms.positions, dtype=torch.float64)
            self.append(
                z=z,
                h=torch.rand(N, node_nf, dtype=torch.float64),
                g=torch.rand(N, node_nf, dtype=torch.float64), 
                pos=pos,
                vel=torch.zeros_like(pos),#torch.tensor(u.atoms.velocities, dtype=torch.float64),
                N=len(u.atoms),
                label='bulla'
            )
    
    def process(self, **input_params):
        
        for top, traj in tqdm(zip(input_params['top_file'], input_params['traj_file'])):
            u = MDAnalysis.Universe(top,  traj)
            self.process_traj(u)
