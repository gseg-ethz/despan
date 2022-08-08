from collections import defaultdict
from dataclasses import dataclass, field
from itertools import compress
import gc
from typing import Iterable, Callable, Any, Tuple

import numpy as np


@dataclass(frozen=True)
class PointCloudData:
    """
    Array with associated photographic information.


    Attributes
    ----------
    xyz : np.ndarray
        nx3 float array with the *x*, *y* and *z* coordinates of the cloud.
    color : np.ndarray
        nx3 uint8 array of *r*, *g* and *b* colors.
    normals : np.ndarray

    """

    xyz: np.ndarray
    color: np.ndarray = None
    normals: np.ndarray = None
    scalar_fields: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """

        """
        # Check types
        assert isinstance(self.xyz, np.ndarray)
        assert self.color is None or isinstance(self.color, np.ndarray)
        assert self.normals is None or isinstance(self.normals, np.ndarray)
        assert isinstance(self.scalar_fields, dict)
        for key, value in self.scalar_fields.items():
            assert isinstance(key, str)
            assert isinstance(value, np.ndarray)

        # Check dimensions
        nb_pts = self.xyz.shape[0]
        assert self.xyz.shape[1] == 3
        assert self.color is None or self.color.shape == (nb_pts, 3,)
        assert self.normals is None or self.normals.shape == (nb_pts, 3,)
        for sf in self.scalar_fields.values():
            assert sf.shape == (nb_pts,)

    def __repr__(self) -> str:
        return f"Point cloud with {self.xyz.shape[0]:,d} point(s)"

    def _reduce_points_to(self, mask: np.ndarray) -> None:
        object.__setattr__(self, "xyz", self.xyz[mask])
        if self.color is not None:
            object.__setattr__(self, "color", self.color[mask])
        if self.normals is not None:
            object.__setattr__(self, "normals", self.normals[mask])
        for sf_key in self.scalar_fields.keys():
            self.scalar_fields[sf_key] = self.scalar_fields[sf_key][mask]

    # def copy(self) -> PointCloudData:
    #     xyz = self.xyz.copy()

        return PointCloudData(self.xyz.copy())

    def filter(self, sf_filter: str, truth_func: Callable[[np.ndarray], np.ndarray[Any, np.dtype[bool]]]) -> None:
        """
        Filters the point cloud based on the function.

        Parameters
        ----------
        sf_filter : str
            Scalar field identifier.
        truth_func
            Callable that takes a number and returns a `bool`
        """
        if sf_filter in self.scalar_fields.keys():
            filter_mask = truth_func(self.scalar_fields[sf_filter])
            self._reduce_points_to(filter_mask)

    def box_cut(self, minimum_corner: Tuple[float, float, float], maximum_corner: Tuple[float, float, float]) -> None:
        # If the dimension difference in an axis is 0, then it should be ignored
        minimum_corner = np.array(minimum_corner, dtype=float)
        maximum_corner = np.array(maximum_corner, dtype=float)

        span = maximum_corner - minimum_corner
        minimum_corner[span == 0] = -np.inf
        maximum_corner[span == 0] = np.inf

        mask = np.logical_and(np.all(self.xyz >= minimum_corner, axis=1),
                              np.all(self.xyz <= maximum_corner, axis=1))
        self._reduce_points_to(mask)


def merge_pcd(pcds: Iterable[PointCloudData]) -> PointCloudData:
    """
    Merge multiple point clouds.

    Merges two or more point clouds. The new point cloud will only retain scalar fields (and colors and normals) if
    TODO: Check for dtype (uint8 vs uint16 -> needs to be scaled)

    Parameters
    ----------
    pcds : iterable[DeSpAn.geometry.PointCloudData]

    Returns
    -------
    pcd : DeSpAn.geometry.PointCloudData
    """
    xyz = []
    color = []
    normals = []
    scalar_fields = defaultdict(list)

    for i, pcd in enumerate(pcds):
        xyz.append(pcd.xyz)
        color.append(pcd.color)
        normals.append(pcd.normals)
        for sf, pcd_sf in pcd.scalar_fields.items():
            scalar_fields[sf].append(pcd_sf)
        scalar_fields["point_cloud_merge"].append(np.ones((pcd.xyz.shape[0],), dtype=np.uint8) * (i + 1))

    # Find empty pcd
    # empty_mask = [False if val is None else True for val in xyz]
    # xyz = list(compress(xyz, empty_mask))
    # color = list(compress(color, empty_mask))
    # normals = list(compress(normals, empty_mask))
    # scalar_fields = {sf_key: list(compress(sf_filter, empty_mask)) for sf_key, sf_filter in scalar_fields.items()}

    nb_pcds = len(xyz)

    if any(val is None for val in color):
        color = None

    if any(val is None for val in normals):
        normals = None

    remove_keys = []
    for sf_key, sf in scalar_fields.items():
        if any(val is None for val in sf) or len(sf) < nb_pcds:
            remove_keys.append(sf_key)

    for rk in remove_keys:
        del (scalar_fields[rk])

    xyz_np = np.vstack(tuple(xyz))
    del xyz
    gc.collect()

    color_np = np.vstack(tuple(color)) if color is not None else None #
    del color
    gc.collect()

    normals_np = np.vstack(tuple(normals)) if normals is not None else None
    del normals
    gc.collect()

    # sf_keys = scalar_fields.keys()
    # scalar_fields_np = {}
    # for sf_key in sf_keys:
    #     scalar_fields_np[sf_key] = np.hstack(tuple(scalar_fields[sf_key]))
    #     del scalar_fields[sf_key]
    #     gc.collect()

    scalar_fields = {sf_key: np.hstack(tuple(sf)) for sf_key, sf in scalar_fields.items()}

    return PointCloudData(xyz_np, color=color_np, normals=normals_np, scalar_fields=scalar_fields)


