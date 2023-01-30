"""Console script for DeSpAn"""

import sys
import subprocess
from pathlib import Path

from shapely.geometry import Polygon
import numpy as np

from DeSpAn.config import RunConfig
from DeSpAn.core import get_point_cloud_data, border_extraction, cut_to_common_box
from DeSpAn.data_io import save_ply, load_ply


RUN_CFG = RunConfig()


def main() -> int:

    filter_functions = (
        [("classification", lambda npa: npa == 2)]
        if RUN_CFG.app_settings.filter_ground_points
        else None
    )
    scalar_fields = []
    if RUN_CFG.app_settings.retain_intensities:
        scalar_fields.append("intensity")
    if RUN_CFG.app_settings.filter_ground_points:
        scalar_fields.append("classification")
    pcd_e1 = get_point_cloud_data(
        RUN_CFG.paths.pcd_e1,
        pcd_file_types=RUN_CFG.app_settings.greedy_file_types,
        greedy=RUN_CFG.app_settings.greedy_directory_search,
        scalar_fields=scalar_fields,
        filter_functions=filter_functions,
    )

    pcd_e2 = get_point_cloud_data(
        RUN_CFG.paths.pcd_e2,
        pcd_file_types=RUN_CFG.app_settings.greedy_file_types,
        greedy=RUN_CFG.app_settings.greedy_directory_search,
        scalar_fields=scalar_fields,
        filter_functions=filter_functions,
    )
    #

    save_ply(
        RUN_CFG.paths.intermediate_results
        / f"01a_{RUN_CFG.project_meta.epoch1_name}_merged.ply",
        pcd_e1,
    )

    save_ply(
        RUN_CFG.paths.intermediate_results
        / f"01b_{RUN_CFG.project_meta.epoch2_name}_merged.ply",
        pcd_e2,
    )

    cut_to_common_box((pcd_e1, pcd_e2))

    pcd_e1_path_boxcut = (
        RUN_CFG.paths.intermediate_results
        / f"02a_{RUN_CFG.project_meta.epoch1_name}_boxcut.ply"
    )
    pcd_e2_path_boxcut = (
        RUN_CFG.paths.intermediate_results
        / f"02b_{RUN_CFG.project_meta.epoch2_name}_boxcut.ply"
    )

    save_ply(pcd_e1_path_boxcut, pcd_e1)
    save_ply(pcd_e2_path_boxcut, pcd_e2)

    border_e1_xy = border_extraction(pcd_e1)
    border_e2_xy = border_extraction(pcd_e2)
    #
    border_e1 = Polygon(border_e1_xy)
    border_e2 = Polygon(border_e2_xy)

    border_common = border_e1.intersection(border_e2)
    # TODO: CHECK FOR MULTIPOLYGON

    border_xy = np.array(border_common.exterior.coords.xy).T
    offset_xy = -np.round(
        np.array(border_common.exterior.centroid.coords.xy).T.squeeze()
    )

    pcd_e1_path_bordercut = (
        RUN_CFG.paths.intermediate_results
        / f"03a_{RUN_CFG.project_meta.epoch1_name}_bordercut.ply"
    )
    pcd_e2_path_bordercut = (
        RUN_CFG.paths.intermediate_results
        / f"03b_{RUN_CFG.project_meta.epoch2_name}_bordercut.ply"
    )

    print("Running border cut on first point cloud")
    cc_call_e1 = subprocess.run(
        [
            RUN_CFG.paths.CC_exe,
            "-SILENT",
            "-LOG_FILE",
            f"{RUN_CFG.paths.intermediate_results / 'boxcut_e1.log'}",
            "-C_EXPORT_FMT",
            "PLY",
            "-AUTO_SAVE",
            "OFF",
            "-O",
            "-GLOBAL_SHIFT",
            *[f"{x:0.3f}" for x in offset_xy],
            "0.0",
            f"{pcd_e1_path_boxcut}",
            "-CROP2D",
            "Z",
            str(border_xy.shape[0]),
            *[f"{x:0.3f}" for x in (border_xy + offset_xy).flatten()],
            "-SAVE_CLOUDS",
            "FILE",
            f"{pcd_e1_path_bordercut}",
        ],
        shell=False,
        check=True,
        text=True,
        capture_output=True,
    )

    print("Running border cut on second point cloud")
    cc_call_e2 = subprocess.run(
        [
            RUN_CFG.paths.CC_exe,
            "-SILENT",
            "-LOG_FILE",
            f"{RUN_CFG.paths.intermediate_results / 'boxcut_e2.log'}",
            "-C_EXPORT_FMT",
            "PLY",
            "-AUTO_SAVE",
            "OFF",
            "-O",
            "-GLOBAL_SHIFT",
            *[f"{x:0.3f}" for x in offset_xy],
            "0.0",
            f"{pcd_e2_path_boxcut}",
            "-CROP2D",
            "Z",
            str(border_xy.shape[0]),
            *[f"{x:0.3f}" for x in (border_xy + offset_xy).flatten()],
            "-SAVE_CLOUDS",
            "FILE",
            f"{pcd_e2_path_bordercut}",
        ],
        shell=False,
        check=True,
        text=True,
        capture_output=True,
    )

    print("Running M3C2")

    cc_m3c2 = subprocess.run(
        [
            RUN_CFG.paths.CC_exe,
            "-SILENT",
            "-NO_TIMESTAMP",
            "-LOG_FILE",
            f"{RUN_CFG.paths.intermediate_results / 'log_m3c2.log'}",
            "-C_EXPORT_FMT",
            "PLY",
            "-AUTO_SAVE",
            "OFF",
            "-O",
            "-GLOBAL_SHIFT",
            *[f"{x:.3f}" for x in offset_xy],
            "0.0",
            f"{pcd_e1_path_bordercut}",
            "-O",
            "-GLOBAL_SHIFT",
            *[f"{x:.3f}" for x in offset_xy],
            "0.0",
            f"{pcd_e2_path_bordercut}",
            "-M3C2",
            f"{RUN_CFG.paths.m3c2_settings}",
            "-SET_ACTIVE_SF",
            "8",
            "-SF_COLOR_SCALE",
            f"{RUN_CFG.paths.hsv_settings}",
            "-SF_CONVERT_TO_RGB",
            "FALSE",
            "-SAVE_CLOUDS",
        ],
        shell=False,
        check=True,
        text=True,
        capture_output=True,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
