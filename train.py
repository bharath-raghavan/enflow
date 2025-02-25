import os
import numpy as np
from enflow.nn.model import ENFlow
from enflow.data.qm9 import QM9
from enflow.data.sdf import SDFDataset
from enflow.data.base import DataLoader
from enflow.data import transforms
import torch
from enflow.units.conversion import ang_to_lj, kelvin_to_lj, picosecond_to_lj

temp = 300

dataset = SDFDataset(raw_file="data/qm9/raw.sdf", processed_file="data/qm9/processed.pt", transform=transforms.Compose([transforms.ConvertPositionsFrom('ang'), transforms.RandomizeVelocity(temp)]))
train_loader = DataLoader(dataset, batch_size=5000, shuffle=False)
print("Loaded dataset")

start_epoch = 0
checkpoint_path = "model.cpt"
num_epochs = 60
checkpoint = None

device = 'cpu'#torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device used is {device}")

node_nf=dataset.node_nf
hidden_nf = 128
n_iter = 10
dt = picosecond_to_lj(10)
kBT = kelvin_to_lj(temp)
r_cut = ang_to_lj(3)
softening = 0.1

print(f"Model params: hidden_nf={node_nf} hidden_nf={hidden_nf} n_iter={n_iter} dt={dt} r_cut={r_cut} kBT={kBT} softening={softening}", flush=True)
model = ENFlow(node_nf=node_nf, hidden_nf=hidden_nf, n_iter=n_iter, dt=dt, r_cut=r_cut, kBT=kBT, softening=softening, device=device)    
model.to(torch.double)

lr = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.5, total_iters=60)

if os.path.exists(checkpoint_path):
    print("Loading from saved state", flush=True)
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch']+1

for epoch in range(start_epoch, num_epochs):
    epoch_losses = []
    losses = []
    
    print(f"Starting epoch {epoch}:", flush=True)
    
    print('Batch/Total \tTraining Loss', flush=True)
    for i, data in enumerate(train_loader):
        out, ldj = model(data.to(device))
        loss = model.nll(out, ldj)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if not i % 10:
            mean = np.mean(losses)
            print('%.5i/%.5i \t    %.2f' % (i, len(train_loader), mean), flush=True)
            epoch_losses.append(mean)
            losses = []
    
    before_lr = optimizer.param_groups[0]["lr"]
    scheduler.step()
    after_lr = optimizer.param_groups[0]["lr"]
    print("Learning rate updated from %.4f to %.4f" % (before_lr, after_lr), flush=True)
                
    print('Total loss: \t%.2f' % (np.mean(epoch_losses)), flush=True)
    # save checkpoint
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }, checkpoint_path)
    print("State saved", flush=True)
                
        
