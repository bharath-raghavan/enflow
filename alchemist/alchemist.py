from pathlib import Path
from collections.abc import AsyncIterator
from typing import Optional, List, Union, Dict
from typing_extensions import Annotated

import asyncio
import os
import logging
_logger = logging.getLogger(__name__)

import numpy as np
import typer
app = typer.Typer()

from .config import load_dict, DynamicConfig, TrainConfig, GenConfig
from .data.lj import LJDataset
from .dynamic import simulate_system
from .asedb import to_ase, add_mols
from .main import Main, Parallel

Model = Annotated[Path,
                typer.Argument(help="NN Parameters for generation.")]

@app.command()
def train(config: Annotated[Path, typer.Argument(help="Training parameter yaml file.")],
          model: Model
         ):
    """ Train (or continue training of) a neural network
        flow model for generating structures.
    """

    trainConfig = TrainConfig.model_validate(load_dict(config))
    with Parallel(world_size=os.environ.get("SLURM_NTASKS", None),
                  world_rank=os.environ.get("SLURM_PROCID", None),
                  local_rank=os.environ.get("SLURM_LOCALID", None),
                  num_cpus_per_task=os.environ.get("SLURM_CPUS_PER_TASK", None)) \
            as ann:
        ann.train(trainConfig, model)

@app.command()
def generate(config: GenConfig,
             model: Model,
             db: Annotated[Path, typer.Argument(help="Input structures for generator.")],
             out: Annotated[Path, typer.Argument(help="ASE DB to store results.")],
             s: Annotated[int, typer.Option(help="Number of structures to generate.")] = 0):
    """ Use a trained model to generate structures.
    """
    # FIXME: this is basically pseudocode, since the functions aren't
    # setup correctly yet.
    genConfig = GenConfig.model_validate(load_dict(config))
    with Parallel(world_size=os.environ.get("SLURM_NTASKS", None),
                  world_rank=os.environ.get("SLURM_PROCID", None),
                  local_rank=os.environ.get("SLURM_LOCALID", None),
                  num_cpus_per_task=os.environ.get("SLURM_CPUS_PER_TASK", None)) \
            as ann:
        i = 0
        for row in db:
            i += 1
            if s > 0 and i > s:
                break
            y = ann.generate(genConfig, model, row.atoms())
            out.add(y)

@app.command()
def dynamics(config: Annotated[Path, typer.Argument(help="DynamicConfig parameters.")],
             s: Annotated[int, typer.Argument(help="Number of structures to generate.")],
             out: Annotated[Path, typer.Argument(help="Output file.")]
            ):
    """ Run dynamics on a model to generate structures.
    """

    #config = DynamicConfig.model_validate(**load_dict(config))
    config = DynamicConfig.model_validate(load_dict(config))
    #data = LJDataset(config, s, out)
    #data.process()

    volume = config.nparticles*(config.sigma**3)/config.reduced_density
    box_edge = volume**(1.0/3.0)

    names = ['Ar']*config.nparticles
    cell  = np.eye(3)*box_edge
    def get_ase(x, e):
        return to_ase(names, x, cell, energy=e)

    async def map_fn(it, fn):
        i = 0
        for z in it:
            yield fn(*z)
            i += 1
            if i >= s:
                break

    def c2(inp, out, *args):
        return out(inp, *args)

    async def consume(stream):
        async for i in stream:
            pass

    asyncio.run(
            consume( add_mols(map_fn(simulate_system(config), get_ase), out) )
        )
