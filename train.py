import os
import sys
import yaml
import numpy as np
from datetime import timedelta
import time

import torch

from enflow.flow.dynamics import LeapFrogIntegrator, VelocityVerletIntegrator
from enflow.flow.loss import Alchemical_NLL
from enflow.nn.egcl import EGCL
from enflow.data.sdf import SDFDataset
from enflow.data.base import DataLoader
from enflow.data import transforms
from enflow.utils.conversion import ang_to_lj, kelvin_to_lj, picosecond_to_lj, femtosecond_to_lj
from enflow.utils.helpers import get_box
from enflow.utils.constants import sigma

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

class Trainer:

    def __init__(self, world_size=None, world_rank=None, local_rank=None, num_cpus_per_task=None):
        if world_size and world_rank and local_rank:
            self.ddp = True
        else:
            self.ddp = False
            
        if self.ddp:
            self.world_size = int(world_size)
            self.world_rank = int(world_rank)
            self.local_rank = int(local_rank)

            os.environ["WORLD_SIZE"] = str(self.world_size)
            os.environ["RANK"] = str(self.world_rank)
            os.environ["LOCAL_RANK"] = str(self.local_rank)

            os.environ["NCCL_SOCKET_IFNAME"] = "hsn0"
            torch.cuda.set_device(self.local_rank)
            device = torch.cuda.current_device()

            dist.init_process_group('nccl', timeout=timedelta(seconds=7200000), init_method="env://", rank=self.world_rank, world_size=self.world_size)
    
            self.num_cpus_per_task = int(num_cpus_per_task)
            
            if world_rank == 0:
                eprint(f"DDP initialized? {dist.is_initialized()}", flush=True)
        else:
            self.world_rank = 0
            self.local_rank = 'cpu'
    
    def setup(self, input):
        self.start_epoch = 0
        checkpoint = None
    
        with open(input, 'r') as f: args = yaml.load(f, Loader=yaml.FullLoader)

        if args['dynamics']['time_units'] == 'pico':
            dt = picosecond_to_lj(float(args['dynamics']['dt']))
        else:
            dt = femtosecond_to_lj(float(args['dynamics']['dt']))

        if args['training']['scheduler']:
            scheduler_step = float(args['training']['scheduler_step'])
            gamma = float(args['training']['gamma'])
        else:
            scheduler_step = 0
            gamma = 0
    
        self.checkpoint_path = args['io']['checkpoint_path']
        self.log_interval = int(args['io']['log_interval'])
        self.batch_size = int(args['dataset']['batch_size'])
        self.num_epochs = int(args['training']['num_epochs'])
        
        self.dataset = SDFDataset(raw_file="data/qm9/raw.sdf", processed_file="data/qm9/processed.pt", transform=transforms.Compose([transforms.ConvertPositionsFrom('ang'), transforms.Center(), transforms.RandomizeVelocity(kelvin_to_lj(float(args['dataset']['temp'])))]))
    
        if self.ddp:
            self.sampler = DistributedSampler(self.dataset, num_replicas=self.world_size, rank=self.world_rank, shuffle=False)
            self.train_loader = DataLoader(self.dataset, batch_size=self.batch_size, num_workers=self.num_cpus_per_task, pin_memory=True, shuffle=False, sampler=self.sampler, drop_last=False)
        else:
            self.train_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
    
        node_nf=self.dataset.node_nf

        box = get_box(self.dataset) + int(args['dynamics']['box_pad'])
    
        if self.world_rank == 0:
             eprint(f"Box size: {box}", flush=True)
         
        integrator = args['dynamics']['integrator'].lower()
        network=EGCL(node_nf, node_nf, int(args['dynamics']['network']['hidden_nf']))
        if integrator == 'lf':
            self.model = LeapFrogIntegrator(network=network, n_iter=int(args['dynamics']['n_iter']), dt=dt, r_cut=ang_to_lj(float(args['dynamics']['r_cut'])), box=box).to(self.local_rank)
        elif integrator == 'vv':
            self.model = VelocityVerletIntegrator(network=network, n_iter=int(args['dynamics']['n_iter']), dt=dt, r_cut=ang_to_lj(float(args['dynamics']['r_cut'])), box=box).to(self.local_rank)
        else:
            print("error")
    
        if os.path.exists(self.checkpoint_path):
            if self.world_rank == 0: print("Loading from saved state", flush=True)
            checkpoint = torch.load(self.checkpoint_path, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.start_epoch = checkpoint['epoch']+1
            
        if self.ddp: self.model = DDP(self.model, device_ids=[self.local_rank])
    
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(args['training']['lr']))
        self.scheduler = None
    
        if args['training']['scheduler']:
            scheduler_step = float(args['training']['scheduler_step'])
            gamma = float(args['training']['gamma'])
        else:
            scheduler_step = 0
            gamma = 0
        
        if scheduler_step != 0 and gamma != 0:
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=scheduler_step, gamma=gamma)
        
        self.nll = Alchemical_NLL(kBT=kelvin_to_lj(float(args['lj']['temp'])), softening=float(args['lj']['softening']))

        if checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if self.scheduler: self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        
    def __call__(self, input):
        self.setup(input)
        
        if self.world_rank == 0:
            print('Epoch \tTraining Loss \t   TGPU (s)', flush=True)
    
        for epoch in range(self.start_epoch, self.num_epochs):
            losses = []
    
            if self.ddp: self.sampler.set_epoch(epoch)
            self.model.train()
        
            if self.world_rank == 0:
                eprint(f"###### Starting epoch {epoch} ######", flush=True)
                if self.ddp: torch.cuda.synchronize()
                start_time = time.time()
                
            for i, data in enumerate(self.train_loader):
                if self.world_rank == 0:
                    eprint(f'*** Batch Number {i} out of {len(self.train_loader)} batches ***', flush=True)
                    eprint('GPU \tTraining Loss', flush=True)
                
                data = data.to(self.local_rank)
                self.optimizer.zero_grad()
                out, ldj = self.model(data)
                loss = self.nll(data, ldj)
                loss.backward()
                self.optimizer.step()
                if self.scheduler: self.scheduler.step()
                losses.append(loss.item())
            
                eprint('%.5i \t    %.2f' % (self.world_rank, loss.item()), flush=True)

            epoch_loss = np.mean(losses)
        
            if self.world_rank == 0:
                if self.scheduler: 
                   torch.save({
                       'epoch': epoch,
                       'model_state_dict': self.model.module.state_dict(),
                       'optimizer_state_dict': self.optimizer.state_dict(),
                       'scheduler_state_dict': self.scheduler.state_dict()
                   }, self.checkpoint_path)
                else:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.module.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict()
                    }, self.checkpoint_path)


                eprint("State saved", flush=True)
            
                eprint(f"###### Ending epoch {epoch} ###### ")
            
                if self.ddp: torch.cuda.synchronize()
                end_time = time.time()
                if epoch % self.log_interval == 0: print('%.5i \t    %.2f \t    %.2f' % (epoch, epoch_loss, end_time - start_time), flush=True)
        
            if self.ddp: dist.barrier()
        
        if self.ddp: dist.destroy_process_group()   
        
if __name__ == "__main__":
    trainer = Trainer(world_size=os.environ.get('SLURM_NTASKS'), world_rank=os.environ.get('SLURM_PROCID'), local_rank=os.environ.get('SLURM_LOCALID'), num_cpus_per_task=os.environ["SLURM_CPUS_PER_TASK"])
    trainer(sys.argv[1]) 
