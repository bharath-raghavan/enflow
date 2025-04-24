import numpy as np
from .simulated import SimulatedDataset
from ..utils.constants import sigma as sig, eps

import openmm as mm
import openmm.app as app
import openmm.unit as unit

def arrange_points_on_grid(n, box, space):
    """Arranges n points on a 3D grid.

    Args:
        n: The number of points to arrange.

    Returns:
        A NumPy array of shape (num_x * num_y * num_z, 3) representing the 3D grid coordinates.
    """
    num_z = int(np.ceil(n**(1/3)))
    num_y = int(np.ceil((n/num_z)**(1/2)))
    num_x = int(np.ceil(n/(num_y*num_z)))

    x = np.linspace(space, box[0]-space, num_x)
    y = np.linspace(space, box[1]-space, num_y)
    z = np.linspace(space, box[2]-space, num_z)

    xv, yv, zv = np.meshgrid(x, y, z)
    
    points = np.stack((xv.flatten(), yv.flatten(), zv.flatten()), axis=-1)
    
    return points[:n]
        
class LJDataset(SimulatedDataset):       
    def setup(self, **input_params):
        N = input_params['n_atoms']
        scale = 1
        dist_units = input_params['dist_unit']
        if dist_units == 'ang': scale = 0.1
        L = np.array(input_params['box'])*scale*unit.nanometers # assume square box for now
        softening = input_params['softening']
        cutoff = input_params.get('cutoff', 3.0)
        
        # Lennard-Jones parameters
        sigma = sig * 1e10 * unit.angstrom
        epsilon = eps * 1e-3 * unit.kilojoules_per_mole

        # Make an empty topology
        topology = app.Topology()

        # Add a single chain
        chain = topology.addChain()

        # Add atoms to the topology
        for i in range(N):
            residue = topology.addResidue(name='Ar', chain=chain)
            atom = topology.addAtom(name='Ar', element=app.element.get_by_symbol('Ar'), residue=residue)

        topology.setPeriodicBoxVectors(np.eye(3)*L)

        # Generate positions on a grid
        positions = arrange_points_on_grid(N, L._value, 0.5)*unit.nanometers
        
        custom_lj = '4*epsilon*((sigma/(scale*sigma + r))^12-(sigma/(scale*sigma + r))^6); sigma=0.5*(sigma1+sigma2); epsilon=sqrt(epsilon1*epsilon2)'
        custom_nb_force = mm.CustomNonbondedForce(custom_lj)

        custom_nb_force.addPerParticleParameter('sigma')
        custom_nb_force.addPerParticleParameter('epsilon')
        custom_nb_force.addGlobalParameter('scale', softening)

        for i in range(N):
            custom_nb_force.addParticle([sigma, epsilon])
    
        custom_nb_force.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
        custom_nb_force.setCutoffDistance(cutoff*sigma)
        
        return topology, positions, custom_nb_force, "LJ"
