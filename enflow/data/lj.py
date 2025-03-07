import math
import numpy as np
import torch
from .base import BaseDataset, Data
from ..utils.conversion import kelvin_to_lj, femtosecond_to_lj

class LJMDEngine:
    def __init__(self, softening, target_kBT, dt, nu, box):
        self.softening = softening
        self.target_kBT = target_kBT
        self.dt = dt
        self.nu = nu
        self.box = box
        
    def compute_force(self, pos):
        dr = (pos.unsqueeze(1) - pos)
        r_sq = dr.pow(2).sum(dim=2)
        r = r_sq.pow(0.5)
        r_ = r.pow(-1)
        r_13 = (r+math.sqrt(self.softening)).pow(-13)
        r_7 = (r+math.sqrt(self.softening)).pow(-7)
        f = 48*r_+(r_13 - 0.5*r_7)
        f[f == float("Inf")] = 0
        dx = dr[:,:,0]
        fx = (f*dx).sum(dim=1)
        dy = dr[:,:,1]
        fy = (f*dy).sum(dim=1)
        dz = dr[:,:,2]
        fz = (f*dz).sum(dim=1)
        return torch.stack((fx, fy, fz), dim=1)

    def verlet_andersen(self, x, v, f, N):
        x += self.dt*v + 0.5*self.dt*self.dt*f
        v += 0.5*self.dt*f

        f = self.compute_force(x)

        v += 0.5*self.dt*f
        kBT = v.pow(2).sum()/(3*N)
    
        sigma = math.sqrt(self.target_kBT)

        randf = torch.rand(N) < self.nu*self.dt
        v[randf] = torch.normal(0, sigma, size=v[randf].shape, dtype=torch.float64)

        return x,v,kBT.item(),f

    def run(self, N, nT):
        x = (self.box[1] - self.box[0])*torch.rand(N, 3, dtype=torch.float64) + self.box[0]
        v = torch.normal(0, math.sqrt(self.target_kBT), size=x.shape, dtype=torch.float64)
    
        f = self.compute_force(x)
        
        kBTs = []
        
        for i in range(nT):
            x, v, kBT, f = self.verlet_andersen(x, v, f, N)
            kBTs.append(kBT)
        
        return x, v, np.array(kBTs)
        
def prepare_dataset(md, params):
    data_list = []
    for param in params:
        N = param[0]
        box = param[1]
        nT = param[2]
        discard = param[3]
        
        pos, vel, kBT = md.run(N, box, nT, discard)

        data = Data(
            h=torch.rand(N, 5, dtype=torch.float64),
            g=torch.rand(N, 5, dtype=torch.float64),
            pos=pos,
            vel=vel,
            N=N,
            kBT=kBT 
        )

        data_list.append(data)

    return data_list
    
class LJDataset(BaseDataset):       
    def process(self, **input_params):
        
        md = LJMDEngine(input_params['softening'], input_params['target_kBT'], input_params['dt'], input_params['nu'], input_params['box'])
        log = input_params['log_file']
        discard = input_params['discard']
        node_nf = input_params['node_nf']
        
        log_txt = ''
        
        for i,simulation in enumerate(input_params['simulations']):
            N = simulation[0]
            nT = simulation[1]
        
            pos, vel, kBT = md.run(N, nT)
            
            start = int(nT*discard/100)
            
            mean = kBT[start:].mean()
            var = kBT[start:].var()
            
            log_txt += f'**********************\nSimulation {i}:\nN={N}\nNumber of timesteps:{nT}\n'
            log_txt += f'Temperature Mean:{mean} and Variance:{var}\nTemperature log:\n'
            log_txt += '\n'.join(["%.2f" % number for number in kBT])
            log_txt += '\n'
            self.append(
                z=['Ar']*N,
                h=torch.rand(N, node_nf, dtype=torch.float64),
                g=torch.rand(N, node_nf, dtype=torch.float64),
                pos=pos,
                vel=vel,
                N=N,
                label=f'Temperature Mean:{mean} and Variance:{var}\n'
            )
            
        with open(log, 'w') as f:
            f.write(log_txt)
