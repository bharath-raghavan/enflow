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

checkpoint_path = "/ccs/home/bharathrn/pytorch_tests/enflow/test/model.cpt"
num_epochs = 60

model = ENFlow(node_nf=5, n_iter=4, dt=femtosecond_to_lj(2), dh=femtosecond_to_lj(2), r_cut=ang_to_lj(3), temp=kelvin_to_lj(temp), loss_norm_const=1e2)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# load from checkpoint of it exists
if os.path.exists(checkpoint_path):
    print("Loading from saved state")
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']+1

print('Epoch\tBatch/Total \tLoss train')
for epoch in range(start_epoch, num_epochs):
    epoch_losses = []
    losses = []
    for i, data in enumerate(train_loader):
        out, ldj = model(data)
        loss = model.nll(out, ldj)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        epoch_losses.append(loss.item())
        
        if not i % 10:
            print('%.3i \t%.5i/%.5i \t%.2f' % (epoch,i, len(train_loader), np.mean(losses)), flush=True)
            losses = []
            
    print(f"Epoch loss: {np.mean(epoch_losses)}")    
    # save checkpoint
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)
    print("State saved")
                
        
