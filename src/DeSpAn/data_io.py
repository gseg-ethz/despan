# import datetime
from datetime import datetime
import numpy as np

import laspy
from plyfile import PlyElement, PlyData

from DeSpAn.config import RunConfig

# run_cfg = RunConfig()


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


def read_pcd_directory(directory_path, data_name=None, intermediate_results_path=None, save_intermediate_results=False, *args,
                       **kwargs):
    file_list = find_pcd_in_directory(directory_path, *args, **kwargs)

    pcd_xyz_tuple = tuple(get_point_cloud_data(pcd_path, *args, **kwargs) for pcd_path in file_list)

    # Check if all pcd_xyz have the same shape, if not use only geometries
    if len(np.unique(np.array([xyz.shape[1] for xyz in pcd_xyz_tuple]))) > 1:
        raise RuntimeError

    xyz = np.vstack(pcd_xyz_tuple)

    if save_intermediate_results and intermediate_results_path is not None:
        if data_name is not None:
            np.save(intermediate_results_path / f"01_{data_name}_merged", xyz)
        else:
            np.save(intermediate_results_path / f"01_{datetime.now():%Y%m%d-%H%M%S}_merged", xyz)
    return xyz


def find_pcd_in_directory(directory_path, greedy_directory_search=False, greedy_file_types=[".ply", ".laz", ".las"],
                          *args, **kwargs):
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


def load_ply(pcd_path, scalar_fields=None, *args, **kwargs):
    with open(pcd_path, "rb") as f:
        plydata = PlyData.read(f)
    pcd = np.empty((plydata['vertex'].count, 3,), dtype=float)
    pcd[:, 0] = plydata["vertex"]["x"]
    pcd[:, 1] = plydata["vertex"]["y"]
    pcd[:, 2] = plydata["vertex"]["z"]

    ply_names = [pe.name for pe in plydata["vertex"].properties]

    if scalar_fields is not None:
        sfd_list = list()
        for sf in scalar_fields:
            if sf in ply_names:
                sfd_list.append(np.array(plydata["vertex"][sf])[:, np.newaxis])
        if len(sfd_list) > 0:
            pcd = np.hstack(tuple([pcd, *sfd_list]))

    return pcd
