import os
import sys
import yaml
import numpy as np
from datetime import timedelta
import time

import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from enflow.flow.dynamics import LeapFrogIntegrator
from enflow.flow.loss import Alchemical_NLL
from enflow.nn.egcl import EGCL
from enflow.data.sdf import SDFDataset
from enflow.data.base import DataLoader
from enflow.data import transforms
from enflow.utils.conversion import ang_to_lj, kelvin_to_lj, picosecond_to_lj, femtosecond_to_lj
from enflow.utils.helpers import get_box
from enflow.utils.constants import sigma

def write_xyz(out, file):
    with open(file, 'a') as f:
        f.write("%d\n%s\n" % (10, ' '))
        for x in out.pos:
            #x = x*sigma*1e10
            f.write("%s %.18g %.18g %.18g\n" % ('Ar', x[0].item(), x[1].item(), x[2].item()))

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def init_ddp():

    world_size = int(os.environ['SLURM_NTASKS'])
    world_rank = int(os.environ['SLURM_PROCID'])
    local_rank = int(os.environ['SLURM_LOCALID'])

    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(world_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)

    os.environ["NCCL_SOCKET_IFNAME"] = "hsn0"
    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()

    dist.init_process_group('nccl', timeout=timedelta(seconds=7200000), init_method="env://", rank=world_rank, world_size=world_size)
    
    return world_size, world_rank, local_rank, int(os.environ["SLURM_CPUS_PER_TASK"])

def main(temp, num_epochs, hidden_nf, n_iter, dt, r_cut, softening, batch_size, lr, checkpoint_path, log_interval, box_pad):
    start_epoch = 0
    kBT = kelvin_to_lj(temp)
    checkpoint = None
    
    world_size, world_rank, local_rank, num_cpus_per_task = init_ddp()

    if world_rank == 0:
        eprint(f"DDP initialized? {dist.is_initialized()}", flush=True)

    dataset = SDFDataset(raw_file="data/qm9/raw.sdf", processed_file="data/qm9/processed.pt", transform=transforms.Compose([transforms.ConvertPositionsFrom('ang'), transforms.Center(), transforms.RandomizeVelocity(temp)]))
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=world_rank, shuffle=False)
    train_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_cpus_per_task, pin_memory=True, shuffle=False, sampler=sampler, drop_last=False)

    node_nf=dataset.node_nf
    box = get_box(dataset) + box_pad
    
    if world_rank == 0:
         eprint(f"Box size: {box}", flush=True)

    model = LeapFrogIntegrator(network=EGCL(node_nf, node_nf, hidden_nf), n_iter=n_iter, dt=dt, r_cut=r_cut, box=box).to(local_rank)
    
    if os.path.exists(checkpoint_path):
        if world_rank == 0: print("Loading from saved state", flush=True)
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        #dist.barrier(device_ids=[local_rank])
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']+1
        
    model = DDP(model, device_ids=[local_rank])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if checkpoint: optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    nll = Alchemical_NLL(kBT=kBT, softening=softening)
    #scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
    
    if world_rank == 0:
        print('Epoch \tTraining Loss \t   TGPU (s)', flush=True)
    
    for epoch in range(start_epoch, num_epochs):
        losses = []
    
        sampler.set_epoch(epoch)
        model.train()
        
        if world_rank == 0:
            torch.cuda.synchronize()
            start_time = time.time()
            
            eprint(f"###### Starting epoch {epoch} ######", flush=True)
        
        for i, data in enumerate(train_loader):
            if world_rank == 0:
                eprint(f'*** Batch Number {i} out of {len(train_loader)} batches ***', flush=True)
                eprint('GPU \tTraining Loss', flush=True)
                
            data = data.to(local_rank)
            optimizer.zero_grad()
            out, ldj = model(data)
            loss = nll(data, ldj)
            loss.backward()
            optimizer.step()
            #scheduler.step()
            losses.append(loss.item())
            
            eprint('%.5i \t    %.2f' % (world_rank, loss.item()), flush=True)
            
            if torch.any(torch.gt(out.pos, box[0]*0.5)) or torch.any(torch.lt(out.pos, -box[0]*0.5)):
                eprint(f"Crossing over boundary detected in {world_rank}", flush=True)
                write_xyz(out, f'traj_gpu{world_rank}.xyz')
        
        epoch_loss = np.mean(losses)
        
        if world_rank == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, checkpoint_path)

            eprint("State saved", flush=True)
            
            eprint(f"###### Ending epoch {epoch} ###### ")
            
            torch.cuda.synchronize()
            end_time = time.time()
            print('%.5i \t    %.2f \t    %.2f' % (epoch, epoch_loss, end_time - start_time), flush=True)
        
        dist.barrier()
        
    dist.destroy_process_group()

if __name__ == "__main__":    
    yaml_file = 'config.yaml'
    with open(yaml_file, 'r') as f: args = yaml.load(f, Loader=yaml.FullLoader)
    
    if args['time_units'] == 'pico':
        dt = picosecond_to_lj(float(args['dt']))
    else:
        dt = femtosecond_to_lj(float(args['dt']))
    
    main(float(args['temp']), int(args['num_epochs']), int(args['hidden_nf']), int(args['n_iter']), dt, ang_to_lj(float(args['r_cut'])),\
         float(args['softening']), int(args['batch_size']), float(args['lr']), args['checkpoint_path'], int(args['log_interval']), int(args['box_pad']))
