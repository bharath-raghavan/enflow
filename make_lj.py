from enflow.model import ENFlow
from enflow.data.qm9 import QM9
from enflow.data import transforms
import torch
from torch_geometric.loader import DataLoader
from enflow.units.conversion import ang_to_lj, kelvin_to_lj, femtosecond_to_lj
import torch_geometric.transforms as T
from enflow.units.constants import sigma

def write_xyz(out, file):
    with open(file, 'w') as f:
        f.write("%d\n%s\n" % (out.N.item(), ' '))
        for x in out.pos:
            x = x*sigma*1e10
            f.write("%s %.18g %.18g %.18g\n" % ('Ar', x[0].item(), x[1].item(), x[2].item()))

temp = 120

dataset = QM9(root="moldata/qm9", transform=T.Compose([transforms.ConvertPositions('ang'), transforms.RandomizeVelocity(temp)]))
loader = DataLoader(dataset, batch_size=1, shuffle=False)

checkpoint_path = "model.cpt"

model = ENFlow(node_nf=5, n_iter=4, dt=femtosecond_to_lj(2), dh=femtosecond_to_lj(2), r_cut=ang_to_lj(3), temp=kelvin_to_lj(temp))

checkpoint = torch.load(checkpoint_path, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])

for data in loader: 
    write_xyz(data, 'data.xyz')
    out, _ = model(data)
    break
    write_xyz(out, 'lj.xyz')

 