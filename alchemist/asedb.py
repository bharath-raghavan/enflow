from collections.abc import AsyncIterable, AsyncIterator
from typing import Optional, List

from ase.atoms import Atoms
from ase.db import connect
from ase.units import Ha, Bohr
from ase.io.extxyz import read_xyz
from ase.calculators.singlepoint import SinglePointCalculator
from ase.calculators.calculator import PropertyNotImplementedError

import numpy as np

def to_ase(names:  List[str],
           crd:    np.ndarray,
           cell:   np.ndarray,
           energy: Optional[float] = None,
           forces: Optional[np.ndarray] = None) -> Atoms:
    """ Convert the given structure to an ase.atoms.Atoms object
    """
    magmoms = [0]*len(names)
    charges = [0]*len(names)
    atoms = Atoms(symbols = names, positions = crd, cell = cell,
                  pbc = [True]*3)

    if energy is not None or forces is not None:
        atoms.calc = SinglePointCalculator(atoms,
                                energy=energy,
                                magmom=sum(magmoms),
                                forces=forces)
                                #magmoms=magmoms,
                                #charges=charges)
    #data = { 'dipoles': dipoles, 'ratios': ratios }
    return atoms

async def add_mols(molecules: AsyncIterator[Atoms], fname) -> AsyncIterator[int]:
    """ Add a list of molecules to the ASE database.

        Yields: list of row-ids
    """
    with connect(fname) as db:
        async for mol in molecules:
            yield db.write(mol)
                                   #basis=str(basis),
                                   #functional=functional,
                                   #data=data) )

"""
def show_db(fname):
    with connect(fname) as db:
        #for row in db.select():
        #   print_row(row)
        for a in added:
            print_row(db[a])
"""
