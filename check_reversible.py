from enflow.nn.model import ENFlow
from enflow.data import transforms
import torch
from enflow.data.sdf import SDFDataset
from enflow.data.base import DataLoader
from enflow.units.conversion import ang_to_lj, kelvin_to_lj, picosecond_to_lj
import torch_geometric.transforms as T
from enflow.units.constants import sigma
import numpy as np

def write_xyz(out, file):
    with open(file, 'w') as f:
        f.write("%d\n%s\n" % (out.N.item(), ' '))
        for x in out.pos:
            x = x*sigma*1e10
            f.write("%s %.18g %.18g %.18g\n" % ('Ar', x[0].item(), x[1].item(), x[2].item()))
            
temp = 300

dataset = SDFDataset("data/qm9/raw.sdf", "data/qm9/processed.pt", transform=transforms.Compose([transforms.ConvertPositionsFrom('ang'), transforms.RandomizeVelocity(temp)]))
loader = DataLoader(dataset, batch_size=10, shuffle=True)

checkpoint_path = "model.cpt"

model = ENFlow(node_nf=dataset.node_nf, hidden_nf=128, n_iter=10, dt=picosecond_to_lj(100), r_cut=ang_to_lj(3), kBT=kelvin_to_lj(temp))
model.to(torch.double)

#checkpoint = torch.load(checkpoint_path, weights_only=False)
#model.load_state_dict(checkpoint['model_state_dict'])

for i, data in enumerate(loader): 
    print(i)
    print(data.pos)
    out, _ = model(data)
    print(out.pos)
    rmsd = np.sqrt(((data.pos.detach().numpy() - out.pos.detach().numpy())**2).sum(-1).mean())
    
    data_ = model.reverse(out)
    print(torch.allclose(data_.pos, data.pos, atol=1e-8))
    
    if i > 0: break
        
print("Done")
 