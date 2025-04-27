from .simulated import SimulatedDataset

import openmm
import openmm.app
import openmm.unit
from openmm.vec3 import Vec3
from openmmforcefields.generators import SMIRNOFFTemplateGenerator
        
class LIGDataset(SimulatedDataset):       
    def setup(self, **input_params):
        # TODO: accept inputs
        # TODO: OpenMM doesn't fix PBC correctly, so enforcePeriodicBox should be False in reporter and correct by hand
        # TODO: make sure that molecule is not broken in the box when doing that
        molecule = Molecule.from_smiles(smiles)
        topology = molecule.to_topology().to_openmm()

        smirnoff = SMIRNOFFTemplateGenerator(molecules=molecule)
        ff = openmm.app.ForceField('amber/protein.ff14SB.xml', 'amber/tip3p_standard.xml', 'amber/tip3p_HFE_multivalent.xml')
        ff.registerTemplateGenerator(smirnoff.generator)

        molecule.generate_conformers(n_conformers=5)
        positions = to_openmm(molecule.conformers[0])

        modeller = openmm.app.Modeller(topology, positions)
        modeller.addSolvent(ff, boxSize=Vec3(3.0, 3.0, 3.0)*openmm.unit.nanometers)

        system = ff.createSystem(modeller.topology, nonbondedMethod=openmm.app.PME,
                nonbondedCutoff=1*openmm.unit.nanometer, constraints=openmm.app.HBonds)
        
        return openmm.app.Simulation(modeller.topology, system, self.integrator), smiles
