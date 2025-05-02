import torch
from .base import BaseDataset, Data
import MDAnalysis
from tqdm import tqdm
from ..utils.constants import atom_types

class MDDataset(BaseDataset):
    
    def process_traj(self, u, traj, dist_scale, time_scale):
        for frame, ts in enumerate(u.trajectory):
            z = [get_element(a.element, a.mass) for a in u.atoms]
            
            self.append(
                z=z,
                pos=torch.tensor(u.atoms.positions*dist_scale, dtype=torch.float64),
                vel=torch.tensor(u.atoms.velocities*dist_scale/time_scale, dtype=torch.float64),
                N=len(u.atoms),
                label=traj + ' frame: ' + str(frame)
            )
    
    def process(self, **input_params):
        
        dist_units = input_params['dist_unit']
        time_units = input_params['time_unit']
        
        if dist_units == 'ang':
            dist_scale = 1
        elif dist_units == 'nm':
            dist_scale = 0.1
            
        if time_units == 'pico':
            time_scale = 1
        elif time_units == 'femto':
            time_scale = 1e-3
        
        for top, traj in tqdm(zip(input_params['top_file'], input_params['traj_file'])):
            u = MDAnalysis.Universe(top,  traj)
            self.process_traj(u, traj, dist_scale, time_scale)
