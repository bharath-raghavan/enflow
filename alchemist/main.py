import os
import sys
import yaml
import numpy as np
from datetime import timedelta
import time
import importlib

import torch

from .config import TrainConfig, GenConfig, NetworkSetup
from .flow.loss import Alchemical_NLL
from .nn.egcl import EGCL
from .data.sdf import SDFDataset
from .data.base import DataLoader
from .data import transforms
from .utils.conversion import dist_to_lj, kelvin_to_lj, time_to_lj, lj_to_dist, lj_to_kelvin
from .utils.constants import sigma

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, flush=True, **kwargs)

def write_xyz(out, file):
    with open(file, 'w') as f:
        f.write("%d\n%s\n" % (out.N.item(), ' '))
        for x in out.pos:
            x = x*sigma*1e10
            f.write("%s %.18g %.18g %.18g\n" % ('Ar', x[0].item(), x[1].item(), x[2].item()))

class Parallel:
    def __init__(self, world_size=None, world_rank=None, local_rank=None, num_cpus_per_task=None):
        if world_size and world_rank and local_rank and num_cpus_per_task:
            self.ddp = True
        else:
            self.ddp = False
            
        if self.ddp:
            self.world_size = int(world_size)
            self.world_rank = int(world_rank)
            self.local_rank = int(local_rank)
            self.num_cpus_per_task = int(num_cpus_per_task)
        else:
            self.world_size = 1
            self.world_rank = 0
            self.local_rank = 'cpu'
            self.num_cpus_per_task = 4

    def eprint(self, *args, **kwargs):
        if self.world_rank == 0:
            eprint(*args, **kwargs)

    def sync(self):
        if self.ddp: torch.cuda.synchronize()

    def __enter__(self):
        if self.ddp:
            torch.cuda.set_device(self.local_rank)
            device = torch.cuda.current_device()
            os.environ["WORLD_SIZE"] = str(self.world_size)
            os.environ["RANK"] = str(self.world_rank)
            os.environ["LOCAL_RANK"] = str(self.local_rank)
            os.environ["NCCL_SOCKET_IFNAME"] = "hsn0"
            dist.init_process_group('nccl', timeout=timedelta(seconds=7200000), init_method="env://", rank=self.world_rank, world_size=self.world_size)
            self.eprint(f"Running DDP.  Initialized = {dist.is_initialized()}")
        else:
            self.eprint("Running serially")

    def __exit__(self, type, value, traceback):
        #self.file_obj.close()
        if self.ddp: dist.destroy_process_group()
        
    def setup_sampler(self, dataset, batch_size):
        if self.ddp:
            sampler = DistributedSampler(dataset, num_replicas=self.world_size, rank=self.world_rank, shuffle=False)
            train_loader = DataLoader(dataset, batch_size=batch_size, num_workers=self.num_cpus_per_task, pin_memory=True, shuffle=False, sampler=sampler, drop_last=False)
        else:
            sampler = None
            train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        return sampler, train_loader

    def setup_model(self, model):
        if self.ddp:
            return DDP(model, device_ids=[self.local_rank])
        return model

class Main:
    def setup(self, args: TrainConfig, checkpoint: Path):
        self.start_epoch = 0
        checkpoint = None
        
        self.mode = 'train'
        
        if 'dynamics' in args and 'checkpoint_path' in args['dynamics']:
             self.checkpoint_path = args['dynamics']['checkpoint_path']
        else:
            self.checkpoint_path = '' 
        self.load(checkpoint)
    
    def load(self, checkpoint: Path) -> None:
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)

        if self.world_rank == 0:
            print("Loading from saved state", flush=True)
        checkpoint = torch.load(checkpoint, weights_only=False)
        node_nf = checkpoint['node_nf']
        self.hidden_nf = checkpoint['hidden_nf']
        n_iter = checkpoint['n_iter']
        dt = checkpoint['dt']
        self.integrator = checkpoint['integrator']
        lj_kBT = checkpoint['lj_kBT']
        softening = checkpoint['softening']

    def new_model(self, net: NetworkSetup, loss: LossSetup):
        self.hidden_nf = net.hidden_nf
        n_iter = net.n_iter
        dt = net.dt
        temp = loss.temp
        softening = loss.softening
        
    def setup_dataset(self, dataset: DatasetSetup, parallel: Parallel):
        dset = ComposeDataset(dataset.traj_file)
        sampler, train_loader = parallel.setup_sampler(
                                        dset, dataset.batch_size)
        return dset, sampler, train_loader
        
    def setup_model(self, node_nf, checkpoint, parallel: Parallel):
        #node_nf = self.dataset.node_nf
        network = EGCL(node_nf, node_nf, self.hidden_nf)
        integrator_class = getattr(importlib.import_module(f"alchemist.flow.dynamics"), f"{self.integrator.upper()}Integrator")
        model = integrator_class(network=network, n_iter=n_iter, dt=dt).to(parallel.local_rank)

        if checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            self.start_epoch = checkpoint['epoch']+1
        
        return parallel.setup_model(model)

    def train(self, config, checkpoint, parallel):
        log_interval = config.training.log_interval
        num_epochs = config.training.num_epochs

        dataset, sampler, train_loader \
                    = self.setup_dataset(config.dataset, parallel)
        model = self.setup_model(dataset.node_nf, checkpoint, parallel)
        
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=config.training.lr)

        sch = config.training.scheduler
        if sch.step_size != 0 and sch.gamma > 0:
            scheduler = torch.optim.lr_scheduler.StepLR(
                        optimizer,
                        step_size=sch.step_size,
                        gamma=sch.gamma)
        else:
            scheduler = None
        
        nll = Alchemical_NLL(kBT=config.loss.temp*BOLTZMANN, softening=config.loss.softening)

        if checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if scheduler: scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        parallel.eprint('Epoch \tTraining Loss \t   TGPU (s)')

        for epoch in range(self.start_epoch, self.start_epoch+num_epochs):
            losses = []

            if sampler: sampler.set_epoch(epoch)
            model.train()
    
            if parallel.world_rank == 0:
                eprint(f"###### Starting epoch {epoch} ######")
                if parallel.ddp: torch.cuda.synchronize()
                start_time = time.time()
            
            for i, data in enumerate(train_loader):
                if parallel.world_rank == 0:
                    eprint(f'*** Batch Number {i} out of {len(train_loader)} batches ***')
                    eprint('GPU \tTraining Loss')
            
                data = data.to(parallel.local_rank)
                optimizer.zero_grad()
                out, ldj = model(data)
                loss = nll(out, ldj)
                loss.backward()
                optimizer.step()
                if scheduler: scheduler.step()
                losses.append(loss.item())
        
                eprint('%.5i \t    %.2f' % (parallel.world_rank, loss.item()))

            epoch_loss = np.mean(losses)
    
            if parallel.world_rank == 0:
                to_save = {
                       'epoch': epoch,
                       'model_state_dict': model.module.state_dict(),
                       'optimizer_state_dict': optimizer.state_dict(),
                       'node_nf': dataset.node_nf,
                       'hidden_nf': dataset.hidden_nf,
                       'softening': nll.softening,
                       'lj_kBT': nll.kBT,
                       'integrator': integrator,
                       'n_iter': model.module.n_iter,
                       'dt': model.module.dt
                   }
                if scheduler: to_save['scheduler_state_dict'] = scheduler.state_dict()
                
                torch.save(to_save, checkpoint_path)
                parallel.eprint("State saved")
                parallel.eprint(f"###### Ending epoch {epoch} ###### ")
        
                parallel.sync()
                end_time = time.time()
                if epoch % self.log_interval == 0: print('%.5i \t    %.2f \t    %.2f' % (epoch, epoch_loss, end_time - start_time), flush=True)
    
            if self.ddp: dist.barrier()
    
    def generate(self, parallel):
        # FIXME: use separate main() function for gen vs. train.
        # One class should hold the model and be able to do both.
        args['dataset']['node_nf'] = node_nf
        args['dataset']['softening'] = softening
        args['dataset']['temp'] = lj_to_kelvin(lj_kBT)
        args['dataset']['box'] = [float(i) for i in args['dataset']['box']]
        args['dataset']['n_atoms'] = int(args['dataset']['n_atoms'])
        batch_size = 1
        
        for i, data in enumerate(self.train_loader): 
            if i==0:
                break
        
        out = model.reverse(data)
        np.savetxt('h.out', out.h.detach().numpy(), delimiter=' ')
        write_xyz(out, 'test_out.xyz')
        data_, _ = self.model(out)
        eprint(torch.allclose(data_.pos, data.pos, atol=1e-8))
