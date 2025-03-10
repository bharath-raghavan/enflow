from enflow.nn.model import ENFlow
from enflow.data.sdf import SDFDataset
from enflow.data import transforms
import torch
from enflow.data.base import DataLoader
from enflow.units.conversion import ang_to_lj, kelvin_to_lj, picosecond_to_lj
import torch_geometric.transforms as T
from enflow.units.constants import sigma

def write_xyz(out, file):
    with open(file, 'w') as f:
        f.write("%d\n%s\n" % (out.N.item(), ' '))
        for x in out.pos:
            x = x*sigma*1e10
            f.write("%s %.18g %.18g %.18g\n" % ('Ar', x[0].item(), x[1].item(), x[2].item()))

temp = 300

dataset = SDFDataset(raw_file="data/qm9/raw.sdf", processed_file="data/qm9/processed.pt", transform=transforms.Compose([transforms.ConvertPositionsFrom('ang'), transforms.Center(), transforms.RandomizeVelocity(temp)]))
loader = DataLoader(dataset, batch_size=10, shuffle=True)

checkpoint_path = "model.cpt"

model = ENFlow(node_nf=dataset.h.shape[1], n_iter=10, dt=picosecond_to_lj(100), dh=1, r_cut=ang_to_lj(3), temp=kelvin_to_lj(temp))
model.to(torch.double)

checkpoint = torch.load(checkpoint_path, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])

for data in loader: 
    out, _ = model(data)
    break

write_xyz(out, 'lj.xyz')

 