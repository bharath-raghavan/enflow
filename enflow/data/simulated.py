from sys import stdout
from abc import ABC, abstractmethod
import torch
from .base import BaseDataset, Data
from ..utils.helpers import apply_pbc

import openmm as mm
import openmm.app as app
import openmm.unit as unit

class SimulatedDatasetReporter(object):
    def __init__(self, node_nf, transform, report_interval, report_from, desc, dist_units, time_units):
        self.data_list = []
        self.transform = transform
        self.node_nf = node_nf
        self.report_interval = report_interval
        self.report_from = report_from
        self.desc = desc
        self.dist_units = dist_units
        self.time_units = time_units

    def describeNextReport(self, simulation):
        steps = self.report_interval - simulation.currentStep%self.report_interval
        return {'steps': steps, 'periodic': False, 'include':['positions', 'velocities']} # OpenMM's PBC application is not great, we will do it ourselves

    def report(self, simulation, state):
        if simulation.currentStep < self.report_from: return
        
        pos = state.getPositions().value_in_unit(self.dist_units)
        vel = state.getVelocities().value_in_unit(self.dist_units/self.time_units)
        N = len(pos)
        
        box_vec3 = simulation.topology.getUnitCellDimensions().value_in_unit(self.dist_units)
        box = torch.tensor([box_vec3[0], box_vec3[1], box_vec3[2]])
        
        pos = apply_pbc(pos, box)
        
        data = Data(
            z=[a.element.symbol for a in simulation.topology.atoms()]*N,
            h=torch.normal(0, 1, size=(N, self.node_nf), dtype=torch.float64),
            g=torch.normal(0, 1, size=(N, self.node_nf), dtype=torch.float64),
            pos=torch.tensor(pos, dtype=torch.float64),
            vel=torch.tensor(vel, dtype=torch.float64),
            N=N,
            box=box.repeat(N, 1),
            label=f'Simulated dataset: {self.desc} Frame: {simulation.currentStep}'
        )

        if self.transform:
            self.data_list.append(self.transform(data))
        else:
            self.data_list.append(data)

class SimulatedDataset(BaseDataset, ABC):
    @abstractmethod
    def setup(self, **input_params):
        pass
        
    def process(self, **input_params):
        temp = input_params['temp']
        node_nf = input_params['node_nf']
        report_interval = input_params['interval']
        report_from = input_params['discard']
        if report_from == -1: report_from = report_interval
        log = input_params['log']
        traj = input_params['traj']
        n_iter = input_params['n_iter']
        dist_units = input_params['dist_unit']
        time_units = input_params['time_unit']
        dt = input_params['dt']
        friction = input_params['friction']
        
        if dist_units == 'ang':
            dist_units = unit.angstrom
        elif dist_units == 'nm':
            dist_units = unit.nanometers
    
        scale = 1
        if time_units == 'pico':
            time_units = unit.picoseconds
        elif time_units == 'femto':
            time_units = unit.femtoseconds
            scale = 1e-3
            
        self.integrator = mm.LangevinMiddleIntegrator(temp*unit.kelvin, friction/(scale*unit.picosecond), dt*scale*unit.picoseconds)
        simulation, desc = self.setup(**input_params)

        simulation.context.setVelocitiesToTemperature(temp*unit.kelvin)
        
        # Add reporters to get data
        rep = SimulatedDatasetReporter(node_nf, self.transform, report_interval, report_from, desc, dist_units, time_units, input_params['box'])
        simulation.reporters.append(rep)
        
        # Add reporters to output log and traj
        simulation.reporters.append(app.PDBReporter(traj, report_interval))
        simulation.reporters.append(app.StateDataReporter(log, report_interval, step=True, potentialEnergy=True, temperature=True))
        simulation.reporters.append(app.StateDataReporter(stdout, report_interval, step=True, potentialEnergy=True, temperature=True))
        print("Running MD simulation")
        simulation.step(n_iter)
        self.data_list = rep.data_list # capture data list from rep