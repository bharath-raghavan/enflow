import os
import numpy as np
from enflow.model import ENFlow
from enflow.data.qm9 import QM9
from enflow.data import transforms
import torch
from torch_geometric.loader import DataLoader
from enflow.units.conversion import ang_to_lj, kelvin_to_lj, femtosecond_to_lj
import torch_geometric.transforms as T

temp = 120

dataset = QM9(root="moldata/qm9", transform=T.Compose([transforms.ConvertPositions('ang'), transforms.RandomizeVelocity(temp)]))
train_loader = DataLoader(dataset, batch_size=5000, shuffle=False)

start_epoch = 0

checkpoint_path = "model.cpt"
num_epochs = 60

model = ENFlow(node_nf=5, n_iter=4, dt=0.1, dh=0.1, r_cut=ang_to_lj(3), temp=kelvin_to_lj(temp))

lr = 1e-3
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.5, total_iters=60)

# load from checkpoint of it exists
if os.path.exists(checkpoint_path):
    print("Loading from saved state")
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch']+1

for epoch in range(start_epoch, num_epochs):
    epoch_losses = []
    losses = []
    
    print(f"Starting epoch {epoch}:")
    
    print('Batch/Total \tTraining Loss')
    for i, data in enumerate(train_loader):
        out, ldj = model(data)
        loss = model.nll(out, ldj)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if not i % 10:
            mean = np.mean(losses)
            print('%.5i/%.5i \t    %.2f' % (i, len(train_loader), mean))
            epoch_losses.append(mean)
            losses = []
    
    before_lr = optimizer.param_groups[0]["lr"]
    scheduler.step()
    after_lr = optimizer.param_groups[0]["lr"]
    print("Learning rate updated from %.4f to %.4f" % (before_lr, after_lr))
                
    print('Total loss per atom: \t%.2f' % (np.mean(epoch_losses)))
    # save checkpoint
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, checkpoint_path)
    print("State saved")
                
        
