from typing import List, Optional
from pathlib import Path
import json

from pydantic import BaseModel
import yaml

class Units(BaseModel):
    time: str
    dist: str

class DatasetSetup(BaseModel):
    batch_size: int = 100
    #top_file: List[str]
    traj_file: List[str]
    #processed_file: str
    #units: Units

class DynamicConfig(BaseModel):
    nparticles: int = 216
    substeps: int = 100

    reduced_density: float = 0.85 # reduced density rho*sigma^3
    temperature: float = 120 # Kelvin
    collision_rate: int = 5 # 1/ps
    timestep: float = 2.5 # fs
    sigma: float = 3.4 # angstrom
    pressure: Optional[float] = None # bar | None

class SchedulerSetup(BaseModel):
    step_size: int = 0
    gamma: float = 0

class TrainingSetup(BaseModel):
    num_epochs: int
    lr: float
    log_interval: int
    scheduler: SchedulerSetup = SchedulerSetup()

class NetworkSetup(BaseModel):
    """ Configuration Options for the NN flow model,
        including its energy function.
    """
    hidden_nf: int # number of hidden dimensions
    r_cut: float # distance cutoff for NN interactions
    n_iter: int # Number of iterations of the flow
    dt: float # timestep for integrator
    node_nf: int # number of features per node

    #report_interval: int = 10
    #report_from: int = 100
    #traj_file: str
    #integrator: str
    #box_pad: float = 1

class LossSetup(BaseModel):
    temp: float = 298.15
    softening: float

class TrainConfig(BaseModel):
    training: TrainingSetup
    network:  NetworkSetup
    loss:     LossSetup
    dataset:  DatasetSetup

def load_dict(fname: Path) -> dict:
    """ Read a dict from a yaml or json-formatted file.
    """
    with open(fname, "r", encoding="utf-8") as f:
        if fname.suffix in [".yaml", ".yml"]:
            data = yaml.safe_load(f)
        else:
            data = json.load(f)
    return data

#def load_config(fname: Path) -> ConfigFile:
#    """ Read a ConfigFile from a yaml or json-formatted
#    file.
#    """
#    return ConfigFile.model_validate(**load_dict(fname))
