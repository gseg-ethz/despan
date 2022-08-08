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
    greedy_directory_search: bool
    greedy_file_types: list[str]

    def __post_init__(self):
        object.__setattr__(self, 'greedy_file_types',
                           [ft if ft[0] == "." else f".{ft}" for ft in self.greedy_file_types])


@dataclass(init=False, frozen=True)
class _Paths:
    pcd_e1: Path
    pcd_e2: Path
    intermediate_results: Path
    CC_exe: Path = field(compare=False)
    m3c2_settings: Path
    hsv_settings: Path

    def __init__(self, pcd_e1: str, pcd_e2: str, CC_exe: str, intermediate_results: str, m3c2_settings: str,
                 hsv_settings: str) -> None:
        object.__setattr__(self, 'pcd_e1', Path(pcd_e1).absolute())
        object.__setattr__(self, 'pcd_e2', Path(pcd_e2).absolute())
        object.__setattr__(self, 'CC_exe', Path(CC_exe).absolute())
        object.__setattr__(self, 'intermediate_results', Path(intermediate_results).absolute())
        object.__setattr__(self, 'm3c2_settings', Path(m3c2_settings).absolute())
        object.__setattr__(self, 'hsv_settings', Path(hsv_settings).absolute())


@dataclass(init=False, frozen=True)
class RunConfig(metaclass=Singleton):
    project_meta: _ProjectMeta = None
    app_settings: _AppSettings = None
    paths: _Paths = None

    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("-cf", "--config_file", type=str,
                            help="Path of a configuration file to use",
                            default="")
        parser.add_argument("-e1", "--epoch1", type=str,
                            help="Path of the first epoch's point cloud file (or folder)",
                            default=argparse.SUPPRESS)
        parser.add_argument("-e2", "--epoch2", type=str,
                            help="Path of the second epoch's point cloud file (or folder)",
                            default=argparse.SUPPRESS)
        parser.add_argument("-gd", "--greedy_dir_search", type=int,
                            choices=[0, 1],
                            help="Should subdirectories be included in search (0: fal",
                            default=argparse.SUPPRESS)
        # TODO: Add the additional configuration arguments
        args = parser.parse_args()

        with initialize_config_module(version_base=None, config_module="DeSpAn.conf"):
            default_cfg = compose(config_name="default_config")

        file_config_path = Path(args.config_file).absolute()

        if file_config_path.is_file():
            with initialize_config_dir(version_base=None, config_dir=f"{file_config_path.parent}"):
                file_cfg = compose(f"{file_config_path.name}")

            run_cfg_dict = OmegaConf.merge(default_cfg, file_cfg)
        else:
            run_cfg_dict = default_cfg

        for key, value in args.__dict__.items():
            if key == "epoch1":
                run_cfg_dict.project_meta = value
            if key == "epoch2":
                run_cfg_dict.project_meta = value
            if key == "greedy_dir_search":
                run_cfg_dict.app_settings.greedy_directory_search = bool(value)

        for key, value in run_cfg_dict.items():
            object.__setattr__(self, key, instantiate(value))
