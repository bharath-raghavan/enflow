import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import one_hot

from rdkit import Chem, RDLogger
from scipy.special import erf
import numpy as np

from tqdm import tqdm

from ..units.constants import atom_types

class BaseDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])

    def process(self):
        with open(self.raw_paths[2]) as f:
            skip = [int(x.split()[0]) - 1 for x in f.read().split('\n')[9:-2]]

        suppl = Chem.SDMolSupplier(self.raw_paths[0], removeHs=False,
                                   sanitize=False)

        data_list = []
        for i, mol in enumerate(tqdm(suppl)):
            if i in skip:
                continue

            N = mol.GetNumAtoms()

            conf = mol.GetConformer()
            pos = conf.GetPositions()
            pos = torch.tensor(pos, dtype=torch.float)
            
            z = [atom.GetSymbol() for atom in mol.GetAtoms()]
            type_idx = [atom_types[i] for i in z]
            h = one_hot(torch.tensor(type_idx), num_classes=len(atom_types))

            name = mol.GetProp('_Name')
            smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
            
            data = Data(
                z=z,
                h=h,
                g=torch.zeros_like(h),
                pos=pos,
                vel=torch.zeros_like(pos),
                N=N,
                smiles=smiles,
                name=name,
                idx=i,
            )

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        self.save(data_list, self.processed_paths[0])