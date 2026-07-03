import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Dict

sys.path.append(str(Path(__file__).resolve().parent.parent / "third_party" / "gaussian-splatting"))

from scene.cameras import Camera
from gaussian_renderer import render
from argparse import ArgumentParser
from arguments import PipelineParams

from src.utils.pose_io import read_test_poses_csv, quaternion_to_rotation_matrix


def render_test_views(
    scene_name: str,
    scene_path: Path,
    model_path: Path,
    submission_root: Path,
    silent: bool = False,
):
    output_path = submission_root / scene_name
    output_path.mkdir(parents=True, exist_ok=True)

    csv_path = scene_path / "test" / "test_poses.csv"
    test_poses = read_test_poses_csv(csv_path)

    from scene.gaussian_model import GaussianModel
    from utils.general_utils import safe_state

    safe_state(silent)

    gaussians = GaussianModel(3)
    ckpt = _find_checkpoint(model_path)
    gaussians.load_ply(str(ckpt / "point_cloud.ply"))
    gaussians.active_sh_degree = 3

    if not silent:
        print(f"  Loaded {gaussians._xyz.shape[0]} Gaussians")

    pipe = PipelineParams(ArgumentParser())
    bg = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")

    for i, pose in enumerate(test_poses):
        cam = _make_camera(pose)
        result = render(cam, gaussians, pipe, bg)
        img = result["render"]
        img_np = (img.detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        out_file = output_path / pose["image_name"]
        img_np = img_np[..., :3]  # drop alpha if RGBA
        Image.fromarray(img_np).save(out_file)
        if not silent and (i + 1) % 10 == 0:
            print(f"    Rendered {i + 1}/{len(test_poses)}")

    if not silent:
        print(f"  Rendered {len(test_poses)} views -> {output_path}")


def _find_checkpoint(model_path: Path) -> Path:
    pc_dir = model_path / "point_cloud"
    if not pc_dir.exists():
        raise RuntimeError(
            f"No point_cloud directory at {pc_dir}. "
            f"Training may have been interrupted or skipped. "
            f"Delete {model_path} and re-run training."
        )
    dirs = sorted(pc_dir.iterdir())
    if not dirs:
        raise RuntimeError(f"No checkpoints in {pc_dir}")
    return dirs[-1]


def _make_camera(pose: Dict) -> Camera:
    from PIL import Image as PILImage

    R = quaternion_to_rotation_matrix(pose["qvec"]).T
    T = pose["tvec"]

    W, H = pose["width"], pose["height"]
    fx, fy = pose["fx"], pose["fy"]
    FoVx = 2 * np.arctan2(W / 2, fx)
    FoVy = 2 * np.arctan2(H / 2, fy)

    dummy_image = PILImage.fromarray(np.zeros((H, W, 3), dtype=np.uint8))

    cam = Camera(
        resolution=(W, H),
        colmap_id=0,
        R=R,
        T=T,
        FoVx=FoVx,
        FoVy=FoVy,
        depth_params=None,
        image=dummy_image,
        invdepthmap=None,
        image_name=Path(pose["image_name"]).stem,
        uid=0,
    )
    return cam
