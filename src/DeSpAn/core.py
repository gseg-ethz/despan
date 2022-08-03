from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class PointCloudData:
    xyz: np.ndarray
    color: np.ndarray = None
    normals: np.ndarray = None
    scalar_fields: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self):
        # Check types
        assert isinstance(self.xyz, np.ndarray)
        assert self.color is None or isinstance(self.color, np.ndarray)
        assert self.normals is None or isinstance(self.normals, np.ndarray)
        assert isinstance(self.scalar_fields, dict)
        for key, value in self.scalar_fields.items():
            assert isinstance(key, str)
            assert isinstance(value, np.ndarray)

        # Check dimensions
        assert self.xyz.shape[1] == 3
        assert self.color is None or self.color.shape[1] == 3
        assert self.normals is None or self.normals.shape[1] == 3
        for sf in self.scalar_fields.values():
            assert len(sf.squeeze().shape) == 1

        # Check if nn of points matches
        nb_pts = self.xyz.shape[0]
        assert self.color is None or self.color.shape[0] == nb_pts
        assert self.normals is None or self.color.shape[0] == nb_pts
        for sf in self.scalar_fields.values():
            assert sf.squeeze().shape[0] == nb_pts

    def __repr__(self):
        return f"Point cloud with {self.xyz.shape[0]:,d} point(s)"

