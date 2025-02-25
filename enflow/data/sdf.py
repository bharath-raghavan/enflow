import torch
from .base import BaseDataset, Data
from rdkit import Chem
from tqdm import tqdm
from ..units.constants import atom_types

def one_hot(index, num_classes=None, dtype=None):
    if index.dim() != 1:
        raise ValueError("'index' tensor needs to be one-dimensional")

    if num_classes is None:
        num_classes = int(index.max()) + 1

    out = torch.zeros((index.size(0), num_classes), dtype=dtype,
                      device=index.device)
    return out.scatter_(1, index.unsqueeze(1), 1)

class SDFDataset(BaseDataset):       
    def process(self, **input_params):
        suppl = Chem.SDMolSupplier(input_params['raw_file'], removeHs=False, sanitize=False)

        for mol in tqdm(suppl):
            N = mol.GetNumAtoms()

            conf = mol.GetConformer()
            pos = conf.GetPositions()
            pos = torch.tensor(pos, dtype=torch.float64)
    
            z = [atom.GetSymbol() for atom in mol.GetAtoms()]
            type_idx = [atom_types[i] for i in z]
            h = one_hot(torch.tensor(type_idx), num_classes=len(atom_types), dtype=torch.float64)

            name = mol.GetProp('_Name')
            smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
            
            label = name + ' ' + smiles
    
            self.append(
                h=h,
                g=torch.zeros_like(h),
                pos=pos,
                vel=torch.zeros_like(pos),
                N=N,
                label=label
            )
