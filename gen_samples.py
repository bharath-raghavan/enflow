from enflow.nn.model import ENFlow
import torch
from enflow.data.base import DataLoader
from enflow.units.conversion import ang_to_lj, kelvin_to_lj, picosecond_to_lj, femtosecond_to_lj
from enflow.units.constants import sigma
from enflow.data.lj import LJDataset

def write_xyz(out, file):
    with open(file, 'w') as f:
        f.write("%d\n%s\n" % (out.N.item(), ' '))
        for x in out.pos:
            x = x*sigma*1e10
            f.write("%s %.18g %.18g %.18g\n" % ('Ar', x[0].item(), x[1].item(), x[2].item()))
            
kBT = kelvin_to_lj(120)
softening = 0.1

dataset = LJDataset(node_nf=5, softening=0.1, target_kBT=kBT, dt=femtosecond_to_lj(1), nu=10000, box=[0,ang_to_lj(10)], discard=10, simulations=[(5, 50000)], log_file="data/lj/log.txt") #processed_file="data/lj/processed.pt", 
loader = DataLoader(dataset, batch_size=1, shuffle=False)

for i, data in enumerate(loader): 
    if i==0:
        break
    
write_xyz(data, 'test.xyz')

checkpoint_path = "model.cpt"
checkpoint = torch.load(checkpoint_path, weights_only=False)
model = ENFlow(node_nf=dataset.node_nf, hidden_nf=checkpoint['hidden_nf'], n_iter=checkpoint['n_iter'], dt=checkpoint['dt'], r_cut=checkpoint['r_cut'], kBT=checkpoint['kBT'], softening=checkpoint['softening'])
model.to(torch.double)
model.load_state_dict(checkpoint['model_state_dict'])

print(f"Trained till {checkpoint['epoch']} epochs")

out = model.reverse(data)

print(out.h)

write_xyz(out, 'test_out.xyz')

data_, _ = model(out)

print(torch.allclose(data_.pos, data.pos, atol=1e-8))
