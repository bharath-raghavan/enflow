import torch
from .md import MDDataset
import MDAnalysis
from tqdm import tqdm

class COORDSDataset(MDDataset):    
    def process(self, **input_params):
        
        for coords in tqdm(input_params['coords_file']):
            u = MDAnalysis.Universe(coords)
            self.process_traj(u)
