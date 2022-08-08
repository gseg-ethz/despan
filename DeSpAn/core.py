from pathlib import Path
from typing import Any, Callable, Iterable, Tuple

import alphashape
import numpy as np
from shapely.geometry import Polygon

from DeSpAn.geometry import PointCloudData, merge_pcd
from DeSpAn.data_io import find_pcd_in_directory, load_laz, load_ply


def get_point_cloud_data(data_path: Path,
                         pcd_file_types: list[str] = None,
                         greedy: bool = False,
                         scalar_fields: list[str] = None,
                         filter_functions: Iterable[Tuple[str, Callable[[np.ndarray],
                                                                    np.ndarray[Any, np.dtype[bool]]]]] = None
                         ) -> PointCloudData:
    """
    Load *point cloud data*  from either a file or directory (possible inclusion of subdirectories).



    Parameters
    ----------
    data_path : pathlib.Path
        Path to point cloud file or directory. In case of directory all files of type `pcd_file_types` will be loaded
        and merged.
    pcd_file_types : list[str], optional
        List of file type endings to load when `data_path` points to directory. In case of
    greedy : bool, default=False
        Additional search in subdirectories.
    scalar_fields : list[str], optional
        Scalar fields to keep.
    filter_functions : Iterable[tuple[str, func]]
        List of filter functions to run on the data. Each filter function is represented by the scalar field string and
        a function which takes one value and returns a boolean.

    Returns
    -------
    pcd : DeSpAn.geometry.PointCloudData

    """
    if data_path.is_dir() and pcd_file_types is not None:
        pcd_path_list = find_pcd_in_directory(data_path, pcd_file_types, greedy)

        pcds = tuple(get_point_cloud_data(pcd_path, pcd_file_types, greedy, scalar_fields, filter_functions)
                     for pcd_path in pcd_path_list)
        return merge_pcd(pcds)
    elif data_path.is_file():
        if data_path.suffix in [".laz", ".las"]:
            pcd = load_laz(data_path, scalar_fields=scalar_fields)
        elif data_path.suffix == ".ply":
            pcd = load_ply(data_path, scalar_fields=scalar_fields)
        # elif data_path.suffix == ".npy":
        #     return np.load(data_path)
        else:
            raise NotImplementedError
        if filter_functions is not None:
            for f in filter_functions:
                pcd.filter(*f)

        print(f"{pcd.xyz.shape[0]:,d}")
        return pcd
    else:
        raise NotImplementedError


# def cut_to_common_outline_border(pcds: Iterable[PointCloudData]) -> None:
#     cut_to_common_box


def border_extraction(pcd: PointCloudData, alpha_value: float = 20.0, nb_points: int = 10000,
                      show_plot: bool = False) -> np.ndarray:
    # xrange = (np.floor(pcd[:, 0].min()), np.ceil(pcd[:, 0].max()) + 1)
    # xedges = np.arange(*xrange, step=raster_size)
    #
    # yrange = (np.floor(pcd[:, 1].min()), np.ceil(pcd[:, 1].max()) + 1)
    # yedges = np.arange(*yrange, step=raster_size)
    #
    # H, xedges, yedges = np.histogram2d(pcd[:, 0], pcd[:, 1], bins=(xedges, yedges))
    #
    # H[H > 0] = 1
    #
    # # Get x-gradient in "sx"
    # sx = ndimage.sobel(H, axis=0, mode='constant')
    # # Get y-gradient in "sy"
    # sy = ndimage.sobel(H, axis=1, mode='constant')
    # # Get square root of sum of squares
    # sobel = np.hypot(sx, sy)
    #
    # if show_plot:
    #     sobel_binary = sobel.copy()
    #     sobel_binary[sobel_binary > 0] = 1
    #     plt.figure()
    #     plt.imshow(sobel_binary.T, origin="lower", extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap=plt.cm.gray, aspect="equal")
    #     plt.show()
    #
    # x_i, y_i = np.nonzero(sobel)
    # borderish_points = np.array([xedges[x_i], yedges[y_i]]).T

    borderish_points = pcd.xyz[:, 0:2]

    bp_mean = np.mean(borderish_points, axis=0)
    bp_scale = np.max(borderish_points, axis=0) - np.min(borderish_points, axis=0)
    bp_norm = (borderish_points - bp_mean) / bp_scale
    bp_norm_ds = bp_norm[np.random.permutation(bp_norm.shape[0])[:nb_points], :]

    als200 = alphashape.alphashape(bp_norm_ds, alpha=alpha_value)

    if isinstance(als200, Polygon):
        return np.array(als200.exterior.xy).T * bp_scale + bp_mean
        # return als200
    else:
        raise NotImplementedError


def cut_to_common_box(pcds: Iterable[PointCloudData], margin: float = 0.0) -> None:
    """
    Determines the minimum common bounding box and reduces all point clouds to the points within.

    Parameters
    ----------
    pcds : Iterable[DeSpAn.geometry.PointCloudData]
    margin : float, default=0.0
        Additional margin to extend the bounding box (Factor by which the diagonal gets expanded).
    """
    minimum_corner = np.ones((3,), dtype=float) * -np.inf
    maximum_corner = np.ones((3,), dtype=float) * np.inf

    for pcd in pcds:
        minimum_corner = np.maximum(minimum_corner, np.amin(pcd.xyz, axis=0))
        maximum_corner = np.minimum(maximum_corner, np.amax(pcd.xyz, axis=0))

    if margin:
        span = maximum_corner - minimum_corner
        minimum_corner = minimum_corner - margin * span
        maximum_corner = maximum_corner + margin * span

    minimum_corner = tuple(minimum_corner)
    maximum_corner = tuple(maximum_corner)

    for pcd in pcds:
        pcd.box_cut(minimum_corner, maximum_corner)

