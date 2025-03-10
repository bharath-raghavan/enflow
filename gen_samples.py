from enflow.nn.model import ENFlow
import torch
from enflow.data.base import DataLoader
from enflow.utils.conversion import ang_to_lj, kelvin_to_lj, picosecond_to_lj, femtosecond_to_lj
from enflow.utils.constants import sigma
from enflow.data.lj import LJDataset
from enflow.utils.helpers import get_box

def write_xyz(out, file):
    with open(file, 'w') as f:
        f.write("%d\n%s\n" % (out.N.item(), ' '))
        for x in out.pos:
            x = x*sigma*1e10
            f.write("%s %.18g %.18g %.18g\n" % ('Ar', x[0].item(), x[1].item(), x[2].item()))
           
temp = 300 
kBT = kelvin_to_lj(temp)
softening = 0.1
box_len = ang_to_lj(10)

dataset = LJDataset(node_nf=5, softening=0.1, target_kBT=kBT, dt=femtosecond_to_lj(1), nu=10000, box=[0,box_len], discard=10, simulations=[(5, 50000)], log_file="data/lj/log.txt") #processed_file="data/lj/processed.pt", 
loader = DataLoader(dataset, batch_size=1, shuffle=False)

for i, data in enumerate(loader): 
    if i==0:
        break
    
write_xyz(data, 'test.xyz')

checkpoint_path = "model.cpt"
checkpoint = torch.load(checkpoint_path, weights_only=False)

node_nf=dataset.node_nf
hidden_nf = 128
n_iter = 10
dt = picosecond_to_lj(1)
r_cut = ang_to_lj(3)
box = torch.tensor([box_len, box_len, box_len], dtype=torch.float64)

print(f"Model params: hidden_nf={node_nf} hidden_nf={hidden_nf} n_iter={n_iter} dt={dt} r_cut={r_cut} kBT={kBT} softening={softening} box={box}", flush=True)
model = ENFlow(node_nf=node_nf, hidden_nf=hidden_nf, n_iter=n_iter, dt=dt, r_cut=r_cut, kBT=kBT, softening=softening, box=box)    
model.to(torch.double)
model.load_state_dict(checkpoint['model_state_dict'])

print(f"Trained till {checkpoint['epoch']} epochs")

out = model.reverse(data)

print(out.h)

write_xyz(out, 'test_out.xyz')

data_, _ = model(out)

print(torch.allclose(data_.pos, data.pos, atol=1e-8))
