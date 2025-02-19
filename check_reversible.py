from enflow.nn.model import ENFlow
from enflow.data.qm9 import QM9
from enflow.data import transforms
import torch
from torch_geometric.loader import DataLoader
from enflow.units.conversion import ang_to_lj, kelvin_to_lj, picosecond_to_lj
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

model = ENFlow(node_nf=dataset.h.shape[1], n_iter=10, dt=picosecond_to_lj(100), dh=1, r_cut=ang_to_lj(3), temp=kelvin_to_lj(temp))
model.to(torch.double)

#checkpoint = torch.load(checkpoint_path, weights_only=False)
#model.load_state_dict(checkpoint['model_state_dict'])

for i, data in enumerate(loader): 
    if i==3: break

write_xyz(data, 'data.xyz')

out, _ = model(data.clone())

write_xyz(out, 'out.xyz')

data_ = model.reverse(out.clone())

print(torch.allclose(data_.pos, data.pos, atol=1e-8))
 