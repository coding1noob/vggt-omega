# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

try:
    import pycolmap
except ImportError as exc:  # pragma: no cover - handled by CLI caller
    pycolmap = None
    PYCOLMAP_IMPORT_ERROR = exc
else:
    PYCOLMAP_IMPORT_ERROR = None


def require_pycolmap():
    if pycolmap is None:
        raise ImportError(
            "pycolmap is required for COLMAP export. Install it with `pip install pycolmap` "
            "or `pip install 'vggt-omega[export]'`."
        ) from PYCOLMAP_IMPORT_ERROR


def randomly_limit_trues(mask: np.ndarray, max_trues: int) -> np.ndarray:
    if max_trues <= 0:
        return np.zeros_like(mask, dtype=bool)

    true_indices = np.flatnonzero(mask)
    if true_indices.size <= max_trues:
        return mask

    sampled_indices = np.random.choice(true_indices, size=max_trues, replace=False)
    limited_flat_mask = np.zeros(mask.size, dtype=bool)
    limited_flat_mask[sampled_indices] = True
    return limited_flat_mask.reshape(mask.shape)


def create_pixel_coordinate_grid(num_frames: int, height: int, width: int) -> np.ndarray:
    y_grid, x_grid = np.indices((height, width), dtype=np.float32)
    x_grid = x_grid[np.newaxis, :, :]
    y_grid = y_grid[np.newaxis, :, :]

    x_coords = np.broadcast_to(x_grid, (num_frames, height, width))
    y_coords = np.broadcast_to(y_grid, (num_frames, height, width))

    frame_idx = np.arange(num_frames, dtype=np.float32)[:, np.newaxis, np.newaxis]
    f_coords = np.broadcast_to(frame_idx, (num_frames, height, width))

    return np.stack((x_coords, y_coords, f_coords), axis=-1)


def _build_pycolmap_intri(fidx: int, intrinsics: np.ndarray, camera_type: str) -> np.ndarray:
    if camera_type == "PINHOLE":
        return np.array(
            [intrinsics[fidx][0, 0], intrinsics[fidx][1, 1], intrinsics[fidx][0, 2], intrinsics[fidx][1, 2]],
            dtype=np.float64,
        )
    if camera_type == "SIMPLE_PINHOLE":
        focal = (intrinsics[fidx][0, 0] + intrinsics[fidx][1, 1]) / 2
        return np.array([focal, intrinsics[fidx][0, 2], intrinsics[fidx][1, 2]], dtype=np.float64)
    raise ValueError(f"Camera type {camera_type} is not supported yet")


def build_colmap_reconstruction_wo_track(
    points3d: np.ndarray,
    points_xyf: np.ndarray,
    points_rgb: np.ndarray,
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    image_size: tuple[int, int],
    image_names: list[str],
    shared_camera: bool = False,
    camera_type: str = "PINHOLE",
):
    require_pycolmap()

    num_frames = len(extrinsics)
    num_points = len(points3d)
    width, height = map(int, image_size)

    if len(image_names) != num_frames:
        raise ValueError(f"Expected {num_frames} image names, got {len(image_names)}")

    reconstruction = pycolmap.Reconstruction()

    for point_idx in range(num_points):
        reconstruction.add_point3D(points3d[point_idx], pycolmap.Track(), points_rgb[point_idx])

    camera = None
    for frame_idx in range(num_frames):
        if camera is None or not shared_camera:
            pycolmap_intri = _build_pycolmap_intri(frame_idx, intrinsics, camera_type)
            camera = pycolmap.Camera(
                model=camera_type,
                width=width,
                height=height,
                params=pycolmap_intri,
                camera_id=frame_idx + 1,
            )
            reconstruction.add_camera_with_trivial_rig(camera)

        cam_from_world = pycolmap.Rigid3d(
            pycolmap.Rotation3d(extrinsics[frame_idx][:3, :3]),
            extrinsics[frame_idx][:3, 3],
        )

        points2d_list = []
        point2d_idx = 0
        image_id = frame_idx + 1
        points_belong_to_frame = np.nonzero(points_xyf[:, 2].astype(np.int32) == frame_idx)[0]

        for point3d_batch_idx in points_belong_to_frame:
            point3d_id = point3d_batch_idx + 1
            point2d_xy = points_xyf[point3d_batch_idx][:2]
            points2d_list.append(pycolmap.Point2D(point2d_xy, point3d_id))

            track = reconstruction.points3D[point3d_id].track
            track.add_element(image_id, point2d_idx)
            point2d_idx += 1

        image = pycolmap.Image(
            name=image_names[frame_idx],
            camera_id=camera.camera_id,
            image_id=image_id,
            points2D=pycolmap.Point2DList(points2d_list),
        )
        reconstruction.add_image_with_trivial_frame(image, cam_from_world)

    return reconstruction
