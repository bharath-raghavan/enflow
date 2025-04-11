import os
import numpy as np
import yaml
import torch

from enflow.flow.dynamics import LeapFrogIntegrator
from enflow.nn.egcl import EGCL
from enflow.data.sdf import SDFDataset
from enflow.data.base import DataLoader
from enflow.data import transforms
from enflow.flow.loss import Alchemical_NLL
from enflow.utils.conversion import ang_to_lj, kelvin_to_lj, picosecond_to_lj, femtosecond_to_lj
from enflow.utils.helpers import get_box

def write_xyz(out, file):
    with open(file, 'a') as f:
        f.write("%d\n%s\n" % (10, ' '))
        for x in out.pos:
            #x = x*sigma*1e10
            f.write("%s %.18g %.18g %.18g\n" % ('Ar', x[0].item(), x[1].item(), x[2].item()))

def main(temp, num_epochs, hidden_nf, n_iter, dt, r_cut, softening, batch_size, lr, checkpoint_path, log_interval):
        start_epoch = 0
        kBT = kelvin_to_lj(temp)
        dataset = SDFDataset(raw_file="data/qm9/raw.sdf", processed_file="data/qm9/processed.pt", transform=transforms.Compose([transforms.ConvertPositionsFrom('ang'), transforms.Center(), transforms.RandomizeVelocity(temp)]))
        train_loader = DataLoader(dataset, batch_size=5000, shuffle=False)
        print("Loaded dataset")
        box = get_box(dataset) + 0.5 # padding
        node_nf=dataset.node_nf
        print(f"Box size: {box}", flush=True)
        model = LeapFrogIntegrator(network=EGCL(node_nf, node_nf, hidden_nf), n_iter=n_iter, dt=dt, r_cut=r_cut, box=box)
        model.to(torch.double)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        #scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.5, total_iters=60)
        nll = Alchemical_NLL(kBT=kBT, softening=softening)

        for epoch in range(start_epoch, num_epochs):
            epoch_losses = []
            losses = []

            print(f"Starting epoch {epoch}:", flush=True)

            print('Batch/Total \tTraining Loss', flush=True)
            for i, data in enumerate(train_loader):
                out, ldj = model(data)
                loss = nll(data, ldj)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.append(loss.item())
            
                write_xyz(out, f'traj.xyz')
            
                if not i % 10:
                    mean = np.mean(losses)
                    print('%.5i/%.5i \t    %.2f' % (i, len(train_loader), mean), flush=True)
                    epoch_losses.append(mean)
                    losses = []

            #before_lr = optimizer.param_groups[0]["lr"]
            #scheduler.step()
            #after_lr = optimizer.param_groups[0]["lr"]
            #print("Learning rate updated from %.4f to %.4f" % (before_lr, after_lr), flush=True)
            
            print('Total loss: \t%.2f' % (np.mean(epoch_losses)), flush=True)

if __name__ == "__main__":    
    yaml_file = 'config.yaml'
    with open(yaml_file, 'r') as f: args = yaml.load(f, Loader=yaml.FullLoader)

    main(float(args['temp']), int(args['num_epochs']), int(args['hidden_nf']), int(args['n_iter']), femtosecond_to_lj(float(args['dt'])), ang_to_lj(float(args['r_cut'])),\
         float(args['softening']), int(args['batch_size']), float(args['lr']), args['checkpoint_path'], int(args['log_interval']))
    
