from enflow.data.sdf import SDFDataset, DataLoader
from enflow.data import transforms
import torch


temp = 300

dataset = SDFDataset("data/qm9/raw.sdf", "data/qm9/processed.pt", transform=transforms.Compose([transforms.ConvertPositionsFrom('ang'), transforms.RandomizeVelocity(temp)]))
train_loader = DataLoader(dataset, batch_size=2, shuffle=False)
for i, data in enumerate(train_loader):
    if i==0: continue
    print(data.pos)
    print(data.N)
    print(data.num_atoms)
    break
    
for a in data:
    for m in a:    
        print(m.pos)
        print(m.N)
        print(m.num_atoms)
