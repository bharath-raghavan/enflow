import os
import sys
import yaml
import numpy as np
from datetime import timedelta
from pathlib import Path
import time
import importlib

import torch

from .config import TrainConfig, SchedulerSetup, NetworkSetup, TrainingSetup, DatasetSetup
from .flow.loss import Alchemical_NLL
from .flow.dynamics import LFIntegrator
from .nn.egcl import EGCL
from .nn.argmax import ArgMax
from .data.sdf import SDFDataset
from .data.base import DataLoader, ComposeInMemoryDatasets
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
    def barrier(self):
        if self.ddp: dist.barrier()

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

        return self

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

class GenNN:
    def __init__(self,
                 parallel: Parallel,
                 config: TrainConfig,
                 model = None,
                 epoch: int = 0,
                 ) -> None:
        self.parallel = parallel
        self.config = config
        self.epoch = epoch
        if model is None:
            self.model = new_model(config.network, parallel)
        else:
            self.model = model
        self.scheduler = None
        self.optimizer = None
    
    @classmethod
    def load(cls, checkpoint_path: Path, parallel: Parallel):
        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)

        parallel.eprint("Loading from saved state")
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        config = TrainConfig.model_validate(**checkpoint['config'])
        epoch = checkpoint['epoch']

        model = new_model(config.network, parallel)
        model.load_state_dict(checkpoint['model_state_dict'])

        nn = cls(parallel, config, model, epoch)
        nn.init_optimizer(config)
        nn.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        nn.init_scheduler(config.scheduler)
        if nn.scheduler:
            nn.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return nn

    def setup_dataset(self, dataset: DatasetSetup):
        dset = ComposeInMemoryDatasets(dataset.traj_file)
        sampler, train_loader = self.parallel.setup_sampler(
                                        dset, dataset.batch_size)
        return dset, sampler, train_loader
        
    def init_optimizer(self, training: TrainingSetup):
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=training.lr)

    def init_scheduler(self, sch: SchedulerSetup):
        if sch.step_size != 0 and sch.gamma > 0:
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                        optimizer,
                        step_size=sch.step_size,
                        gamma=sch.gamma)

    def train(self, dataset: DatasetSetup, training: TrainingSetup,
              checkpoint_path: Path):
        log_interval = training.log_interval
        num_epochs = training.num_epochs

        parallel = self.parallel
        dataset, sampler, train_loader = self.setup_dataset(dataset)
        if self.optimizer is None:
            self.init_optimizer(training)
        if self.scheduler is None:
            self.init_scheduler(training.scheduler)
        
        nll = Alchemical_NLL(kBT=loss.temp*BOLTZMANN, softening=loss.softening)

        parallel.eprint('Epoch \tTraining Loss \t   TGPU (s)')

        while self.epoch < train.num_epochs:
            self.epoch += 1
            losses = []

            if sampler: sampler.set_epoch(self.epoch)
            self.model.train()
    
            if parallel.world_rank == 0:
                parallel.eprint(f"###### Starting epoch {self.epoch} ######")
                parallel.synchronize()
                start_time = time.time()
            
            for i, data in enumerate(train_loader):
                parallel.eprint(f'*** Batch Number {i} out of {len(train_loader)} batches ***\nGPU \tTraining Loss')
            
                data = data.to(parallel.local_rank)
                self.optimizer.zero_grad()
                out, ldj = self.model(data)
                loss = nll(out, ldj)
                loss.backward()
                self.optimizer.step()
                if self.scheduler: self.scheduler.step()
                losses.append(loss.item())
        
                parallel.eprint('%.5i \t    %.2f' % (parallel.world_rank, loss.item()))

            epoch_loss = np.mean(losses)

            if parallel.world_rank == 0:
                parallel.sync()
                end_time = time.time()
                if epoch % self.log_interval == 0: print('%.5i \t    %.2f \t    %.2f' % (epoch, epoch_loss, end_time - start_time), flush=True)
                self.save(checkpoint_path)
    
            parallel.barrier()
    
    def save(self, checkpoint_path: Path):
        if self.parallel.world_rank != 0:
            return
        eprint("State saved")
        eprint(f"###### Ending epoch {epoch} ###### ")
        to_save = {
            'epoch': epoch,
            'model_state_dict': self.model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': self.config.model_dump(),
        }
        if self.scheduler:
            to_save['scheduler_state_dict'] = scheduler.state_dict()
        torch.save(to_save, checkpoint_path)
    
    def generate(self, data):
        parallel = self.parallel
        data = data.to(parallel.local_rank)
        out = self.model.reverse(data)
        yield out
        #np.savetxt('h.out', out.h.detach().numpy(), delimiter=' ')
        #write_xyz(out, 'test_out.xyz')
        #data_, _ = self.model(out)
        #eprint(torch.allclose(data_.pos, data.pos, atol=1e-8))

def new_model(net: NetworkSetup, parallel: Parallel) -> EGCL:
    network = EGCL(net.node_nf, net.node_nf, net.hidden_nf)
    #integrator_class = getattr(importlib.import_module(f"alchemist.flow.dynamics"), f"{self.integrator.upper()}Integrator")
    networks = [network]*net.n_iter

    dequant = ArgMax(net.node_nf, net.hidden_nf)
    model = LFIntegrator(networks, dequant, net.dt).to(parallel.local_rank)

    return parallel.setup_model(model)

