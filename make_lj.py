import os
import sys
import yaml
import numpy as np
from datetime import timedelta
import time

import torch

from enflow.flow.dynamics import LeapFrogIntegrator
from enflow.flow.loss import Alchemical_NLL
from enflow.nn.egcl import EGCL
from enflow.data.sdf import SDFDataset
from enflow.data.base import DataLoader
from enflow.data import transforms
from enflow.utils.conversion import ang_to_lj, kelvin_to_lj, picosecond_to_lj, femtosecond_to_lj
from enflow.utils.constants import sigma
from enflow.utils.helpers import get_box

def write_xyz(out, file):
    with open(file, 'w') as f:
        f.write("%d\n%s\n" % (10, ' '))
        for x in out.pos:
            x = x*sigma*1e10
            f.write("%s %.18g %.18g %.18g\n" % ('Ar', x[0].item(), x[1].item(), x[2].item()))

yaml_file = 'config.yaml'
with open(yaml_file, 'r') as f: args = yaml.load(f, Loader=yaml.FullLoader)
temp = float(args['temp'])
hidden_nf = int(args['hidden_nf'])
n_iter = int(args['n_iter'])
dt = femtosecond_to_lj(float(args['dt']))
r_cut = ang_to_lj(float(args['r_cut']))
softening = float(args['softening'])

dataset = SDFDataset(raw_file="data/qm9/raw.sdf", processed_file="data/qm9/processed.pt", transform=transforms.Compose([transforms.ConvertPositionsFrom('ang'), transforms.Center(), transforms.RandomizeVelocity(temp)]))
loader = DataLoader(dataset, batch_size=10, shuffle=True)

checkpoint_path = args['checkpoint_path']
box = get_box(dataset) + 0.5 # padding

node_nf=dataset.node_nf
model = LeapFrogIntegrator(network=EGCL(node_nf, node_nf, hidden_nf), n_iter=n_iter, dt=dt, r_cut=r_cut, box=box)
model.to(torch.double)

checkpoint = torch.load(checkpoint_path, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])

for data in loader: 
    out, _ = model(data)
    break

write_xyz(out, 'lj.xyz')

 