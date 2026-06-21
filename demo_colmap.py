# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import glob
import os

import numpy as np
import torch

from visual_util import depth_edge
from vggt_omega.models import VGGTOmega
from vggt_omega.utils.colmap_export import (
    build_colmap_reconstruction_wo_track,
    create_pixel_coordinate_grid,
    randomly_limit_trues,
    require_pycolmap,
)
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera

try:
    import trimesh
except ImportError:
    trimesh = None


def parse_args():
    parser = argparse.ArgumentParser(description="VGGT-Omega COLMAP demo")
    parser.add_argument("--input_path", type=str, required=True, help="Directory containing scene images in <input_path>/images")
    parser.add_argument("--output_path", type=str, required=True, help="Directory where outputs will be written")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the VGGTOmega checkpoint")
    parser.add_argument("--image_resolution", type=int, default=512, help="Image resolution used for preprocessing")
    parser.add_argument(
        "--conf_thres_value",
        type=float,
        default=5.0,
        help="Absolute depth confidence threshold for exporting 3D points",
    )
    parser.add_argument("--max_points", type=int, default=100000, help="Maximum number of 3D points to export")
    parser.add_argument(
        "--camera_type",
        type=str,
        default="PINHOLE",
        choices=["PINHOLE", "SIMPLE_PINHOLE"],
        help="COLMAP camera model",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--save_npz", action="store_true", default=False, help="Save predictions.npz alongside outputs")
    parser.add_argument(
        "--filter_depth_edges",
        action="store_true",
        default=False,
        help="Filter depth discontinuities before exporting points",
    )
    parser.add_argument(
        "--depth_edge_rtol",
        type=float,
        default=0.03,
        help="Relative depth jump threshold used with --filter_depth_edges",
    )
    parser.add_argument("--use_ba", action="store_true", default=False, help="Reserved for future BA support")
    parser.add_argument("--max_reproj_error", type=float, default=8.0, help="Reserved for future BA support")
    parser.add_argument("--shared_camera", action="store_true", default=False, help="Reserved for future BA support")
    parser.add_argument("--vis_thresh", type=float, default=0.2, help="Reserved for future BA support")
    parser.add_argument("--query_frame_num", type=int, default=8, help="Reserved for future BA support")
    parser.add_argument("--max_query_pts", type=int, default=4096, help="Reserved for future BA support")
    parser.add_argument(
        "--fine_tracking",
        action="store_true",
        default=True,
        help="Reserved for future BA support",
    )
    return parser.parse_args()


def load_model(checkpoint_path: str) -> VGGTOmega:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run VGGT-Omega COLMAP export.")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = VGGTOmega().eval()
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    return model.to("cuda")


def unproject_depth_map_to_point_map(depth_map: np.ndarray, extrinsic: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    depth = depth_map[..., 0]
    num_frames, height, width = depth.shape

    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    x = np.broadcast_to(x[None], (num_frames, height, width))
    y = np.broadcast_to(y[None], (num_frames, height, width))

    fx = intrinsic[:, 0, 0][:, None, None]
    fy = intrinsic[:, 1, 1][:, None, None]
    cx = intrinsic[:, 0, 2][:, None, None]
    cy = intrinsic[:, 1, 2][:, None, None]

    camera_points = np.stack(
        [
            (x - cx) / fx * depth,
            (y - cy) / fy * depth,
            depth,
        ],
        axis=-1,
    )

    rotation = extrinsic[:, :3, :3]
    translation = extrinsic[:, :3, 3]
    return np.einsum(
        "sij,shwj->shwi",
        np.transpose(rotation, (0, 2, 1)),
        camera_points - translation[:, None, None, :],
    )


def run_model(input_path: str, model: VGGTOmega, image_resolution: int) -> tuple[dict, list[str]]:
    image_names = sorted(glob.glob(os.path.join(input_path, "images", "*")))
    if len(image_names) == 0:
        raise ValueError(f"No images found in {os.path.join(input_path, 'images')}")

    print(f"Processing {len(image_names)} images from {input_path}")
    images = load_and_preprocess_images(image_names, image_resolution=image_resolution).to("cuda")
    print(f"Preprocessed images shape: {tuple(images.shape)}")

    with torch.inference_mode():
        predictions = model(images)

    extrinsic, intrinsic = encoding_to_camera(predictions["pose_enc"], predictions["images"].shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    predictions_np = {}
    for key, value in predictions.items():
        if isinstance(value, torch.Tensor):
            value = value.detach().float().cpu().numpy()
            if value.shape[0] == 1:
                value = value[0]
            predictions_np[key] = value

    predictions_np["world_points_from_depth"] = unproject_depth_map_to_point_map(
        predictions_np["depth"],
        predictions_np["extrinsic"],
        predictions_np["intrinsic"],
    )

    torch.cuda.empty_cache()
    return predictions_np, image_names


def run_ba_export(args, predictions_np: dict, image_names: list[str]):
    raise NotImplementedError(
        "--use_ba is not implemented in vggt-omega yet. Upstream BA support depends on tracker and "
        "pycolmap helper modules that are not present in this repository. Use the default feed-forward "
        "export path for now."
    )


def run_feedforward_export(args, predictions_np: dict, image_names: list[str]):
    world_points = predictions_np["world_points_from_depth"]
    depth_conf = predictions_np["depth_conf"]
    extrinsic = predictions_np["extrinsic"]
    intrinsic = predictions_np["intrinsic"]
    images = predictions_np["images"]
    depth = predictions_np["depth"]

    num_frames, height, width, _ = world_points.shape
    image_size = (width, height)

    points_rgb = np.transpose(images, (0, 2, 3, 1))
    points_rgb = (points_rgb * 255).clip(0, 255).astype(np.uint8)

    conf_mask = depth_conf >= args.conf_thres_value
    if args.filter_depth_edges:
        conf_mask &= ~depth_edge(depth[..., 0], rtol=args.depth_edge_rtol)
    conf_mask &= np.isfinite(world_points).all(axis=-1)
    conf_mask = randomly_limit_trues(conf_mask, args.max_points)

    if not np.any(conf_mask):
        raise ValueError("No valid 3D points remained after confidence and finite-value filtering.")

    points_3d = world_points[conf_mask]
    points_rgb = points_rgb[conf_mask]
    points_xyf = create_pixel_coordinate_grid(num_frames, height, width)[conf_mask]
    base_image_names = [os.path.basename(path) for path in image_names]

    print(f"Exporting {len(points_3d)} 3D points to COLMAP")
    reconstruction = build_colmap_reconstruction_wo_track(
        points_3d,
        points_xyf,
        points_rgb,
        extrinsic,
        intrinsic,
        image_size,
        image_names=base_image_names,
        shared_camera=args.shared_camera,
        camera_type=args.camera_type,
    )

    sparse_dir = os.path.join(args.output_path, "sparse", "0")
    os.makedirs(sparse_dir, exist_ok=True)
    reconstruction.write(sparse_dir)
    print(f"Saved COLMAP reconstruction to {sparse_dir}")

    if trimesh is None:
        print("Skipping points.ply export because trimesh is not installed.")
    else:
        trimesh.PointCloud(points_3d, colors=points_rgb).export(os.path.join(args.output_path, "points.ply"))
        print(f"Saved point cloud to {os.path.join(args.output_path, 'points.ply')}")


def main():
    args = parse_args()
    require_pycolmap()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.output_path, exist_ok=True)

    model = load_model(args.checkpoint)
    predictions_np, image_names = run_model(args.input_path, model, args.image_resolution)

    if args.save_npz:
        prediction_save_path = os.path.join(args.output_path, "predictions.npz")
        np.savez(prediction_save_path, **predictions_np)
        print(f"Saved predictions to {prediction_save_path}")

    if args.use_ba:
        run_ba_export(args, predictions_np, image_names)
    else:
        run_feedforward_export(args, predictions_np, image_names)


if __name__ == "__main__":
    main()
