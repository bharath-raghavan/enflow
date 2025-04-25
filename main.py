import os
import sys
import yaml
import numpy as np
from datetime import timedelta
import time
import importlib

import torch

from enflow.flow.loss import Alchemical_NLL
from enflow.nn.egcl import EGCL
from enflow.data.sdf import SDFDataset
from enflow.data.base import DataLoader
from enflow.data import transforms
from enflow.utils.conversion import dist_to_lj, kelvin_to_lj, time_to_lj, lj_to_dist, lj_to_kelvin
from enflow.utils.constants import sigma

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def write_xyz(out, file):
    with open(file, 'w') as f:
        f.write("%d\n%s\n" % (out.N.item(), ' '))
        for x in out.pos:
            x = x*sigma*1e10
            f.write("%s %.18g %.18g %.18g\n" % ('Ar', x[0].item(), x[1].item(), x[2].item()))
                
class Main:

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
                eprint(f"Running DDP\nInitialized? {dist.is_initialized()}", flush=True)
        else:
            print("Running serially", flush=True)
            self.world_rank = 0
            self.local_rank = 'cpu'
    
    def setup(self, input):
        self.start_epoch = 0
        checkpoint = None
    
        with open(input, 'r') as f: args = yaml.load(f, Loader=yaml.FullLoader)
        
        self.mode = 'train'
        if args['mode'] == 'generate': self.mode = 'gen'
        elif args['mode'] == 'dataset': self.mode = 'data'
        elif args['mode'] != 'train': eprint("error")
        
        if 'dynamics' in args and 'checkpoint_path' in args['dynamics']:
             self.checkpoint_path = args['dynamics']['checkpoint_path']
        else:
            self.checkpoint_path = '' 
        
        if os.path.exists(self.checkpoint_path):
            if self.world_rank == 0: print("Loading from saved state", flush=True)
            checkpoint = torch.load(self.checkpoint_path, weights_only=False)
            node_nf = checkpoint['node_nf']
            self.hidden_nf = checkpoint['hidden_nf']
            box = checkpoint['box']
            n_iter = checkpoint['n_iter']
            dt = checkpoint['dt']
            r_cut = checkpoint['r_cut']
            self.integrator = checkpoint['integrator']
            lj_kBT = checkpoint['lj_kBT']
            softening = checkpoint['softening']
        elif self.mode != 'data':
            self.hidden_nf = int(args['dynamics']['network']['hidden_nf'])
            n_iter = int(args['dynamics']['n_iter'])
            dt = time_to_lj(float(args['dynamics']['dt']), unit=args['units']['time'])
            r_cut = dist_to_lj(float(args['dynamics']['r_cut']), unit=args['units']['dist'])
            self.integrator = args['dynamics']['integrator'].lower()
            lj_kBT = kelvin_to_lj(float(args['training']['loss']['temp']))
            softening = float(args['training']['loss']['softening'])
        
        dataset_type = args['dataset']['type']
        dataset_class = getattr(importlib.import_module(f"enflow.data.{dataset_type}"), f"{dataset_type.upper()}Dataset")
        dataset_args = {i:args['dataset'][i] for i in args['dataset'] if (i!='temp' and i!='batch_size' and i!='type')}
        
        T = [transforms.ConvertPositionsFrom(args['units']['dist']), transforms.Center()]
        
        if 'temp' in args['dataset']:
            T.append(transforms.RandomizeVelocity(kelvin_to_lj(float(args['dataset']['temp']))))
        else:
            T.append(transforms.ConvertVelocitiesFrom(args['units']['dist'], args['units']['time']))
        
        dataset_args['dist_unit'] = args['units']['dist']
        dataset_args['time_unit'] = args['units']['time']
        
        if self.mode == 'gen':
            dataset_args['node_nf'] = node_nf
            dataset_args['softening'] = softening
            dataset_args['temp'] = lj_to_kelvin(lj_kBT)
            dataset_args['box'] = [lj_to_dist(i, unit=args['units']['dist']) for i in box.tolist()]
            dataset_args['n_atoms'] = checkpoint['N']    
            batch_size = 1
        elif self.mode == 'train':
            batch_size = int(args['training']['batch_size'])
        
        self.dataset = dataset_class(**dataset_args, transform=transforms.Compose(T))
        
        if self.mode == 'data':
            return
        
        if self.ddp:
            self.sampler = DistributedSampler(self.dataset, num_replicas=self.world_size, rank=self.world_rank, shuffle=False)
            self.train_loader = DataLoader(self.dataset, batch_size=batch_size, num_workers=self.num_cpus_per_task, pin_memory=True, shuffle=False, sampler=self.sampler, drop_last=False)
        else:
            self.train_loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False)
        
        if not checkpoint:
            node_nf = self.dataset.node_nf
            box_native_units = self.dataset.box + float(args['dynamics']['box_pad'])
            box = dist_to_lj(box_native_units, unit=args['units']['dist'])
            if self.world_rank == 0: eprint(f"Using box of size {box_native_units[0]} x {box_native_units[1]} x {box_native_units[2]}")
        
        network=EGCL(node_nf, node_nf, self.hidden_nf)
        integrator_class = getattr(importlib.import_module(f"enflow.flow.dynamics"), f"{self.integrator.upper()}Integrator")
        self.model = integrator_class(network=network, n_iter=n_iter, dt=dt, r_cut=r_cut, box=box).to(self.local_rank)
        
        if checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.start_epoch = checkpoint['epoch']+1
            
        if self.ddp: self.model = DDP(self.model, device_ids=[self.local_rank])
        
        if self.mode == 'gen':
            if self.world_rank == 0: eprint("In generate mode", flush=True)
            return    
        
        if self.world_rank == 0: eprint("In training mode", flush=True)
        
        if args['training']['scheduler']:
            scheduler_step = float(args['training']['scheduler_step'])
            gamma = float(args['training']['gamma'])
        else:
            scheduler_step = 0
            gamma = 0
        
        self.log_interval = int(args['training']['log_interval'])
        self.num_epochs = int(args['training']['num_epochs'])
        
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
        
        self.nll = Alchemical_NLL(kBT=lj_kBT, softening=softening)

        if checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if self.scheduler: self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    def train(self):
        if self.world_rank == 0:
            print('Epoch \tTraining Loss \t   TGPU (s)', flush=True)

        for epoch in range(self.start_epoch, self.start_epoch+self.num_epochs):
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
                to_save = {
                       'epoch': epoch,
                       'model_state_dict': self.model.module.state_dict(),
                       'optimizer_state_dict': self.optimizer.state_dict(),
                       'node_nf': self.dataset.node_nf,
                       'hidden_nf': self.hidden_nf,
                       'softening': self.nll.softening,
                       'lj_kBT': self.nll.kBT,
                       'box': self.model.module.box,
                       'integrator': self.integrator,
                       'n_iter': self.model.module.n_iter,
                       'dt': self.model.module.dt,
                       'r_cut': self.model.module.r_cut,
                       'N': self.dataset.num_atoms_per_mol
                   }
                if self.scheduler: to_save['scheduler_state_dict'] = self.scheduler.state_dict()
                
                torch.save(to_save, self.checkpoint_path)

                eprint("State saved", flush=True)
        
                eprint(f"###### Ending epoch {epoch} ###### ")
        
                if self.ddp: torch.cuda.synchronize()
                end_time = time.time()
                if epoch % self.log_interval == 0: print('%.5i \t    %.2f \t    %.2f' % (epoch, epoch_loss, end_time - start_time), flush=True)
    
            if self.ddp: dist.barrier()
    
    def generate(self):
        
        for i, data in enumerate(self.train_loader): 
            if i==0:
                break
        
        out = self.model.reverse(data)

        np.savetxt('h.out', out.h.detach().numpy(), delimiter=' ')

        write_xyz(out, 'test_out.xyz')

        data_, _ = self.model(out)

        print(torch.allclose(data_.pos, data.pos, atol=1e-8))
       
    def __call__(self, input):
        self.setup(input)
        
        if self.mode == 'train':
            self.train()
        elif self.mode == 'gen':
            self.generate()
        
        if self.ddp: dist.destroy_process_group()

        
if __name__ == "__main__":
    main_hndl = Main(world_size=os.environ.get('SLURM_NTASKS'), world_rank=os.environ.get('SLURM_PROCID'), local_rank=os.environ.get('SLURM_LOCALID'), num_cpus_per_task=os.environ.get("SLURM_CPUS_PER_TASK"))
    main_hndl(sys.argv[1]) 
