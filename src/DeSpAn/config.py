__all__ = ["RunConfig"]

import argparse
from dataclasses import dataclass, field
from pathlib import Path

from hydra import compose, initialize_config_module, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


@dataclass(frozen=True)
class _ProjectMeta:
    name: str
    epoch1_name: str
    epoch2_name: str


@dataclass(frozen=True)
class _AppSettings:
    retain_intensities: bool
    filter_ground_points: bool
    save_intermediate_results: bool
    greedy_directory_search: bool
    greedy_file_types: list[str]

    def __post_init__(self):
        object.__setattr__(self, 'greedy_file_types',
                           [ft if ft[0] == "." else f".{ft}" for ft in self.greedy_file_types])


@dataclass(init=False, frozen=True)
class _Paths:
    pcd_e1: Path
    pcd_e2: Path
    CC_exe: Path = field(compare=False)
    intermediate_results: Path

    def __init__(self, pcd_e1: str, pcd_e2: str, CC_exe: str, intermediate_results: str) -> None:
        object.__setattr__(self, 'pcd_e1', Path(pcd_e1).absolute())
        object.__setattr__(self, 'pcd_e2', Path(pcd_e2).absolute())
        object.__setattr__(self, 'CC_exe', Path(CC_exe).absolute())
        object.__setattr__(self, 'intermediate_results', Path(intermediate_results).absolute())


@dataclass(init=False, frozen=True)
class RunConfig(metaclass=Singleton):
    project_meta: _ProjectMeta = None
    app_settings: _AppSettings = None
    paths: _Paths = None

    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-cf', '--config_file', type=str,
                            help='Location of a configuration file to use',
                            # TODO: REMOVE DEFAULT VALUE
                            default=".\\specific_config.yaml")
        # TODO: Add the additional configuration arguments
        args = parser.parse_args()

        with initialize_config_module(version_base=None, config_module="DeSpAn"):
            default_cfg = compose(config_name="default_config")

        file_config_path = Path(args.config_file).absolute()

        with initialize_config_dir(version_base=None, config_dir=f"{file_config_path.parent}"):
            file_cfg = compose(f"{file_config_path.name}")


        run_cfg_dict = OmegaConf.merge(default_cfg, file_cfg)
        for key, value in run_cfg_dict.items():
            object.__setattr__(self, key, instantiate(value))
