from enflow.flow.dynamics import LeapFrogIntegrator
from enflow.nn.egcl import EGCL
from enflow.data import transforms
import torch
from enflow.data.sdf import SDFDataset
from enflow.data.base import DataLoader
from enflow.utils.conversion import ang_to_lj, kelvin_to_lj, picosecond_to_lj, femtosecond_to_lj
import torch_geometric.transforms as T
from enflow.utils.constants import sigma
from enflow.utils.helpers import get_box
import numpy as np

def write_xyz(out, file):
    with open(file, 'w') as f:
        f.write("%d\n%s\n" % (out.N.item(), ' '))
        for x in out.pos:
            x = x*sigma*1e10
            f.write("%s %.18g %.18g %.18g\n" % ('Ar', x[0].item(), x[1].item(), x[2].item()))
            
temp = 300

dataset = SDFDataset(raw_file="data/qm9/raw.sdf", processed_file="data/qm9/processed.pt", transform=transforms.Compose([transforms.ConvertPositionsFrom('ang'), transforms.Center(), transforms.RandomizeVelocity(temp)]))
loader = DataLoader(dataset, batch_size=10, shuffle=True)

model = LeapFrogIntegrator(network=EGCL(dataset.node_nf, dataset.node_nf, hidden_nf=128), n_iter=10, dt=picosecond_to_lj(10), r_cut=ang_to_lj(3), kBT=kelvin_to_lj(temp), box=get_box(dataset))
model.to(torch.double)

for i, data in enumerate(loader):
    print(data.h)
    dequant_h, ldj = model.dequantize(data.h)

    print(dequant_h)
    print(ldj)
    
    quant_h = model.quantize(dequant_h)
    print(quant_h)
    
    print(data.h == quant_h)
    break
 