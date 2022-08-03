from datetime import datetime
from pathlib import Path
import numpy as np

import laspy
from plyfile import PlyElement, PlyData

from DeSpAn.config import RunConfig
from DeSpAn.core import PointCloudData

RUN_CFG = RunConfig()


def get_point_cloud_data(data_path, data_name=None, *args, **kwargs):
    if data_path.is_dir():
        return read_pcd_directory(data_path, data_name, *args, **kwargs)
    elif data_path.is_file():
        if data_path.suffix in [".laz", ".las"]:
            return load_laz(data_path, *args, **kwargs)
        elif data_path.suffix == ".ply":
            return load_ply(data_path, *args, **kwargs)
        elif data_path.suffix == ".npy":
            return np.load(data_path)
        else:
            raise RuntimeError


def read_pcd_directory(directory_path, *args, **kwargs):
    file_list = find_pcd_in_directory(directory_path, *args, **kwargs)

    pcd_xyz_tuple = tuple(get_point_cloud_data(pcd_path, *args, **kwargs) for pcd_path in file_list)

    # Check if all pcd_xyz have the same shape, if not use only geometries
    if len(np.unique(np.array([xyz.shape[1] for xyz in pcd_xyz_tuple]))) > 1:
        raise RuntimeError

    xyz = np.vstack(pcd_xyz_tuple)

    # if save_intermediate_results and intermediate_results_path is not None:
    #     if data_name is not None:
    #         np.save(intermediate_results_path / f"01_{data_name}_merged", xyz)
    #     else:
    #         np.save(intermediate_results_path / f"01_{datetime.now():%Y%m%d-%H%M%S}_merged", xyz)
    return xyz


def find_pcd_in_directory(directory_path, *args, **kwargs):

    greedy_directory_search = RUN_CFG.app_settings.greedy_directory_search
    greedy_file_types = RUN_CFG.app_settings.greedy_file_types

    file_list = [file_path for file_path in directory_path.iterdir() if file_path.suffix.lower() in greedy_file_types]

    if greedy_directory_search:
        for file_path in directory_path.iterdir():
            if file_path.is_dir():
                file_list.extend(find_pcd_in_directory(file_path, greedy_directory_search, greedy_file_types, *args,
                                                       **kwargs))
    return file_list


def load_laz(pcd_path, filter_ground_points=False, retain_intensities=False, *args, **kwargs):
    pcd = laspy.read(pcd_path)

    pt_mask = [True] * len(pcd) if not filter_ground_points or "classification" not in list(
        pcd.point_format.dimension_names) else pcd.classification == 2

    xyz = pcd[pt_mask].xyz
    if retain_intensities and "intensity" in list(pcd.point_format.dimension_names):
        intensities = pcd[pt_mask]["intensity"]
        xyz = np.hstack((xyz, intensities[:, np.newaxis]))

    print(f"{xyz.shape[0]:,d} points added for file '{pcd_path.name:s}'.")

    return xyz


def save_ply(file_path, pcd, retain_intensities=False, *args, **kwargs):
    if isinstance(pcd, np.ndarray) and pcd.ndim == 1:
        # TODO: Rename program in comment
        el = PlyElement.describe(pcd, "vertex", comments=["Created with dranjan/python-plyfile in REASSESS program",
                                                          f"Created {datetime.now():%Y-%m-%dT%H:%M:%S}"])
        PlyData([el]).write(str(file_path))

    if isinstance(pcd, np.ndarray) and pcd.ndim == 2:
        if retain_intensities and pcd.shape[1] >= 4:
            pcd_np_st = np.empty((pcd.shape[0],), dtype=[('x', 'f8'), ('y', 'f8'), ('z', 'f8'), ('intensity', 'f4')])
            pcd_np_st['intensity'] = pcd[:, 3]
        else:
            pcd_np_st = np.empty((pcd.shape[0],), dtype=[('x', 'f8'), ('y', 'f8'), ('z', 'f8')])
        pcd_np_st['x'] = pcd[:, 0]
        pcd_np_st['y'] = pcd[:, 1]
        pcd_np_st['z'] = pcd[:, 2]
        save_ply(file_path, pcd_np_st, retain_intensities, *args, **kwargs)

    # elif isinstance(pcd, o3d.geometry.PointCloud):
    #     pcd_np = np.hstack((np.asarray(pcd.points), np.asarray(pcd.colors)[:, 0, np.newaxis])) \
    #         if (retain_intensities and len(pcd.colors) > 0) else np.asarray(pcd.points)
    #     save_ply(file_path, pcd_np, retain_intensities, *args, **kwargs)


def load_ply(pcd_path: Path, retain_colors: bool = True, retain_normals: bool = True, scalar_fields: list[str] = None,
             *args, **kwargs) -> PointCloudData:
    """

    Parameters
    ----------
    pcd_path: `pathlib.Path`
    scalar_fields: `list[str]` (optional, default=`None`)
                    List of scalar fields to keep (will be intersected against the available scalar fields from the
                    *ply-file*). `None` retains all available scalar fields.
    retain_colors: `bool` (optional, default=`True`)
    retain_normals: `bool` (optional, default=`True`)


    Returns
    -------
    pcd: `DeSpAn.core.PointCloudData`
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
