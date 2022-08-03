"""Console script for DeSpAn"""

import os
import sys
import errno
import yaml
from pathlib import Path
from pprint import pprint as print

import numpy as np

from DeSpAn.data_io import get_point_cloud_data, save_ply, load_ply


DEFAULT_CONFIG: Path = Path(__file__).resolve().with_name('default_config.yaml')


def main():
    """

    Returns
    -------

    """
    run_config = config_builder()

    pcd_e1 = get_point_cloud_data(data_path=run_config["pcd_e1_path"],
                                  data_name=run_config["e1_name"],
                                  **run_config)

    pcd_e2 = get_point_cloud_data(data_path=run_config["pcd_e2_path"],
                                  data_name=run_config["e2_name"],
                                  **run_config)


    return 0


def config_builder():
    """
    Build the run config based on the `default_config.yaml` and  specific *run_config* yaml files, and any command line
    arguments.
    The priority is defined as:
        1. command line arguments
        2. run config yaml (file defined in the command line arguments)
        3. `default_config.yaml`

    Returns
    -------
    run_config: dict
        (Single-level) dictionary with the necessary config key-value pairs.

    """
    run_config = dict()

    # TODO: Add `argparse`
    default_config = read_yaml(DEFAULT_CONFIG)
    if not isinstance(default_config, dict):
        raise ImportError

    run_config["name"] = default_config["project_meta"]["name"]
    run_config["e1_name"] = default_config["project_meta"]["epoch1_name"]
    run_config["e2_name"] = default_config["project_meta"]["epoch2_name"]

    run_config["retain_intensities"] = default_config["app_settings"]["retain_intensities"]
    run_config["filter_ground_points"] = default_config["app_settings"]["filter_ground_points"]
    run_config["greedy_directory_search"] = default_config["app_settings"]["greedy_directory_search"]
    run_config["greedy_file_types"] = [ft if ft[0] == "." else f".{ft}" for ft in
                                       default_config["app_settings"]["greedy_file_types"]]
    run_config["save_intermediate_results"] = default_config["app_settings"]["save_intermediate_results"]

    run_config["CC_exe_path"] = Path(default_config["paths"]["CC_exe"]).resolve()
    run_config["pcd_e1_path"] = Path(default_config["paths"]["pcd_e1"]).resolve()
    run_config["pcd_e2_path"] = Path(default_config["paths"]["pcd_e2"]).resolve()
    run_config["intermediate_results_path"] = Path(default_config["paths"]["intermediate_results"]).resolve()

    # TODO: Add log of settings

    return run_config


def read_yaml(file_path):
    """

    Parameters
    ----------
    file_path

    Returns
    -------

    """
    try:
        with open(file_path, "r") as f:
            return yaml.safe_load(f)
    except OSError as e:
        print(e)
        return errno.ENOENT
    except yaml.YAMLError as e:
        if hasattr(e, 'problem_mark'):
            mark = e.problem_mark
            print(f"Error position: ({mark.line + 1}:{mark.column + 1})")
            return errno.EIO
        else:
            print(e)
            return errno.EIO


if __name__ == "__main__":
    sys.exit(main())
