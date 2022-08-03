"""Console script for DeSpAn"""

import os
import sys
import errno
import yaml
from datetime import datetime

from pathlib import Path
from pprint import pprint as print

import numpy as np

from DeSpAn.config import RunConfig
from DeSpAn.data_io import get_point_cloud_data, save_ply, load_ply

RUN_CFG = RunConfig()


def main() -> int:
    print(f"{__name__}")
    print(RUN_CFG)

    # Load pcd
    pcd = load_ply(Path("E:\\13_REASSESS\\00_test_data\\Flamatt\\02_intermediate\\99_old\\20220727-165822_border_cut_M3C2.ply"))
    # Save intermediate
    if RUN_CFG.app_settings.save_intermediate_results and RUN_CFG.paths.intermediate_results is not None:
        pass

    return 0


if __name__ == "__main__":
    sys.exit(main())