from enflow.data.sdf import SDFDataset
from enflow.data.base import DataLoader
from enflow.data import transforms
from enflow.utils.constants import sigma
import torch

def write_xyz(file, pos, N, z):
    with open(file, 'w') as f:
        f.write("%d\n%s\n" % (N, ' '))
        for zi, x in zip(z, pos):
            x = x*sigma*1e10
            f.write("%s %.18g %.18g %.18g\n" % (zi, x[0].item(), x[1].item(), x[2].item()))
            
def write_xyz_(file, data):
    with open(file, 'w') as f:
        f.write("%d\n%s\n" % (data.N.item(), ' '))
        for x in data.pos:
            x = x*sigma*1e10
            f.write("%s %.18g %.18g %.18g\n" % ('Ar', x[0].item(), x[1].item(), x[2].item()))

temp = 300

dataset = SDFDataset(raw_file="data/qm9/raw.sdf", processed_file="data/qm9/processed.pt", transform=transforms.Compose([transforms.ConvertPositionsFrom('ang'), transforms.Center(), transforms.RandomizeVelocity(temp)]))
train_loader = DataLoader(dataset, batch_size=1, shuffle=False)

for i, data in enumerate(train_loader):
    print(data.pos)
    data.pos = data.pos + 10
    print(data.pos)
    break
    
for i, data in enumerate(train_loader):
    print(data.pos)
    b = data.pos
    b += 10
    print(data.pos)
    print(b)
    break

#print(dataset.box)

#box = [0.4,0.4,0.4]

#pos = torch.tensor([[0,0,0], [0.1, -0.1, 0.1], [0.15, 0.1, -0.1], [0.35, 0.1, 0.1], [-0.3, 0.1, 0]])
#z = ['H', 'H', 'C', 'N', 'P']

#print(pos)
#write_xyz('test1.xyz', pos, len(pos), z)

#def do_push(coord, pos):
#    box_len = box[coord]
#    box_edge = box[coord]*0.5
#    pushing = - ( (pos[:,coord] >= box_edge)*box_len ) + ( (pos[:,coord] < -box_edge)*box_len )
#    return pushing

#pos_ = pos+torch.stack((do_push(0, pos), do_push(1, pos), do_push(2, pos)), dim=1)
#print(pos_)
#write_xyz('test2.xyz', pos_, len(pos), z)

#pos_ = pos_-torch.stack((do_push(0, pos_), do_push(1, pos_), do_push(2, pos_)), dim=1)
#print(pos_)
#write_xyz('test3.xyz', pos_, len(pos), z)
