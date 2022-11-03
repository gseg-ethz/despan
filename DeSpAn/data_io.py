from datetime import datetime
from itertools import compress
from pathlib import Path
from typing import Any, Callable

import numpy as np
import laspy
from plyfile import PlyElement, PlyData

from DeSpAn.geometry import PointCloudData, merge_pcd


def find_pcd_in_directory(directory_path, pcd_file_types: list[str], greedy: bool = True) -> list[Path]:
    file_list = [file_path for file_path in directory_path.iterdir() if file_path.suffix.lower() in pcd_file_types]

    if greedy:
        for file_path in directory_path.iterdir():
            if file_path.is_dir():
                file_list.extend(find_pcd_in_directory(file_path, pcd_file_types, greedy))
    return file_list


def save_ply(pcd_path: Path, pcd: PointCloudData, retain_colors: bool = True, retain_normals: bool = True,
             scalar_fields: list[str] = None) -> None:
    """

    Parameters
    ----------
    pcd_path
    pcd
    retain_colors
    retain_normals
    scalar_fields
    """
    nb_points = pcd.xyz.shape[0]

    dtype_list = [("x", "f8"), ("y", "f8"), ("z", "f8"), ]

    if retain_colors and pcd.color is not None:
        assert pcd.color.shape == (nb_points, 3)
        dtype_list.extend([("red", "u1"), ("green", "u1"), ("blue", "u1"), ])

    if retain_normals and pcd.normals is not None:
        assert pcd.normals.shape == (nb_points, 3)
        dtype_list.extend([("nx", "f8"), ("ny", "f8"), ("nz", "f8"), ])

    pcd_scalar_fields = pcd.scalar_fields.keys()
    common_scalar_fields = pcd_scalar_fields if scalar_fields is None else list(set(scalar_fields) &
                                                                                set(pcd_scalar_fields))

    for sf in common_scalar_fields:
        assert pcd.scalar_fields[sf].shape == (nb_points, )
        dtype_list.append((sf, pcd.scalar_fields[sf].dtype.str))

    pcd_np_st = np.empty((nb_points,), dtype=dtype_list)

    pcd_np_st["x"] = pcd.xyz[:, 0]
    pcd_np_st["y"] = pcd.xyz[:, 1]
    pcd_np_st["z"] = pcd.xyz[:, 2]

    if retain_colors and pcd.color is not None:
        pcd_np_st["red"] = pcd.color[:, 0]
        pcd_np_st["green"] = pcd.color[:, 1]
        pcd_np_st["blue"] = pcd.color[:, 2]

    if retain_normals and pcd.normals is not None:
        pcd_np_st["nx"] = pcd.normals[:, 0]
        pcd_np_st["ny"] = pcd.normals[:, 1]
        pcd_np_st["nz"] = pcd.normals[:, 2]

    for sf in common_scalar_fields:
        pcd_np_st[sf] = pcd.scalar_fields[sf]

    # TODO: Rename program in comment
    el = PlyElement.describe(pcd_np_st, "vertex", comments=["Created with dranjan/python-plyfile in REASSESS program",
                                                            f"Created {datetime.now():%Y-%m-%dT%H:%M:%S}"])

    if not pcd_path.parent.exists():
        pcd_path.parent.mkdir(parents=True, exist_ok=True)

    PlyData([el]).write(f"{pcd_path}")


def load_ply(pcd_path: Path, retain_colors: bool = True, retain_normals: bool = True, scalar_fields: list[str] = None
             ) -> PointCloudData:
    """
    Loads a ply file using *dranjan/python-plyfile*.

    Parameters
    ----------
    pcd_path : pathlib.Path
    scalar_fields : list[str], optional
                    List of scalar fields to keep (will be intersected against the available scalar fields from the
                    *ply-file*). `None` retains all available scalar fields.
    retain_colors : bool, default=True
    retain_normals : bool, default=True

    Returns
    -------
    pcd : DeSpAn.geometry.PointCloudData
    """
    with open(pcd_path, "rb") as f:
        plydata = PlyData.read(f)
    xyz = np.empty((plydata['vertex'].count, 3,), dtype=float)
    xyz[:, 0] = plydata["vertex"]["x"]
    xyz[:, 1] = plydata["vertex"]["y"]
    xyz[:, 2] = plydata["vertex"]["z"]

    ply_scalar_fields = [pe.name for pe in plydata["vertex"].properties]

    # ply_scalar_fields_lower = [ply_sf.lower() for ply_sf in ply_scalar_fields]
    scalar_fields = None if scalar_fields is None else [sf.lower() for sf in scalar_fields]

    colors = None
    if retain_colors and len(set(ply_scalar_fields) & set(["r", "g", "b", "red", "green", "blue"])) == 3:
        colors = np.empty((plydata['vertex'].count, 3,), dtype=np.uint8)
        colors[:, 0] = plydata["vertex"]["r"] if "r" in ply_scalar_fields else plydata["vertex"]["red"]
        colors[:, 1] = plydata["vertex"]["g"] if "g" in ply_scalar_fields else plydata["vertex"]["green"]
        colors[:, 2] = plydata["vertex"]["b"] if "b" in ply_scalar_fields else plydata["vertex"]["blue"]

    normals = None
    if retain_normals and len(set(ply_scalar_fields) & set(["nx", "ny", "nz"])) == 3:
        normals = np.empty((plydata['vertex'].count, 3,), dtype=float)
        normals[:, 0] = plydata["vertex"]["nx"]
        normals[:, 1] = plydata["vertex"]["ny"]
        normals[:, 2] = plydata["vertex"]["nz"]

    common_scalar_fields = ply_scalar_fields if scalar_fields is None else list(set(scalar_fields) &
                                                                                set(ply_scalar_fields))

    scalar_fields_dict = dict()
    for sf in common_scalar_fields:
        if sf.lower() not in ["x", "y", "z", "r", "g", "b", "red", "green", "blue", "nx", "ny", "nz"]:
            scalar_fields_dict[sf] = np.array(plydata["vertex"][sf]).squeeze()

    return PointCloudData(xyz, color=colors, normals=normals, scalar_fields=scalar_fields_dict)


def load_laz(pcd_path, retain_colors: bool = True, scalar_fields: list[str] = None):
    """
    Loads a ply file using *laspy*.

    TODO: Extend usage from `dimension_names` to `extra_dimension_names`

    Parameters
    ----------
    pcd_path : pathlib.Path
    scalar_fields : list[str], optional
                    List of scalar fields to keep (will be intersected against the available scalar fields from the
                    *ply-file*). `None` retains all available scalar fields.
    retain_colors : bool, default=True
    retain_normals : bool, default=True

    Returns
    -------
    pcd : DeSpAn.geometry.PointCloudData
    """
    pcd = laspy.read(pcd_path)
    laz_scalar_fields = list(pcd.point_format.dimension_names)


    colors = None
    if retain_colors and len(set(laz_scalar_fields) & set(["red", "green", "blue"])) == 3:
        colors = np.empty((pcd.header.point_count, 3,), dtype=np.uint8)
        colors[:, 0] = (pcd["red"] / 256).astype(np.uint8)
        colors[:, 1] = (pcd["green"] / 256).astype(np.uint8)
        colors[:, 2] = (pcd["blue"] / 256).astype(np.uint8)

    common_scalar_fields = laz_scalar_fields if scalar_fields is None else list(set(scalar_fields) &
                                                                                set(laz_scalar_fields))

    scalar_fields_dict = dict()
    for sf in common_scalar_fields:
        if sf.lower() not in ["x", "y", "z", "red", "green", "blue"]:
            if isinstance(pcd[sf], np.ndarray):
                scalar_fields_dict[sf] = pcd[sf]
            elif isinstance(pcd[sf], laspy.point.dims.SubFieldView):
                if sf in ["scan_direction_flag", "edge_of_flight_line", "synthetic", "key_point", "withheld",
                          "overlap"]:
                    scalar_fields_dict[sf] = np.array(pcd[sf], dtype=bool)
                else:
                    scalar_fields_dict[sf] = np.array(pcd[sf])

            else:
                raise NotImplementedError

    print(f"{pcd.header.point_count:,d} points added for file '{pcd_path.name:s}'.")

    return PointCloudData(xyz=pcd.xyz, color=colors, scalar_fields=scalar_fields_dict)
