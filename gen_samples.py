from enflow.nn.model import ENFlow
from enflow.data.qm9 import QM9
from enflow.data import transforms
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, InMemoryDataset
from enflow.units.conversion import ang_to_lj, kelvin_to_lj, picosecond_to_lj
import torch_geometric.transforms as T
from enflow.units.constants import sigma

def make_lj(transform):
    data_list = []
    for i in range(1):
        N = 5

        a = 10
        b = 0
    
        data = Data(
            h=torch.rand(N, 5, dtype=torch.float64),
            g=torch.rand(N, 5, dtype=torch.float64),
            pos=(b - a)*torch.rand(N, 3, dtype=torch.float64) + a,
            vel=torch.zeros(N, 3),
            N=N,
            idx=i,
        )

        data_list.append(transform.forward(data))
        
    return data_list

def write_xyz(out, file):
    with open(file, 'w') as f:
        f.write("%d\n%s\n" % (out.N.item(), ' '))
        for x in out.pos:
            x = x*sigma*1e10
            f.write("%s %.18g %.18g %.18g\n" % ('Ar', x[0].item(), x[1].item(), x[2].item()))
            
temp = 120

dataset = make_lj(transform=T.Compose([transforms.ConvertPositions('ang'), transforms.RandomizeVelocity(temp)]))
loader = DataLoader(dataset, batch_size=1, shuffle=False)

for i, data in enumerate(loader): 
    if i==0: break
    
write_xyz(data, 'test.xyz')

model = ENFlow(node_nf=5, n_iter=10, dt=picosecond_to_lj(100), dh=1, r_cut=ang_to_lj(3), temp=kelvin_to_lj(temp), softening=0.1)
model.to(torch.double)

checkpoint_path = "model.cpt"
checkpoint = torch.load(checkpoint_path, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])

out = model.reverse(data.clone())

print(out.h)

write_xyz(out, 'test_out.xyz')

data_, _ = model(out.clone())

print(torch.allclose(data_.pos, data.pos, atol=1e-8))
