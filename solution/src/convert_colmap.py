import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import shutil
from src.utils.colmap_bin_reader import read_cameras_bin, read_images_bin, read_points3D_bin


def convert_scene(scene_path: Path, output_path: Path):
    sparse_in = scene_path / "train" / "sparse" / "0"
    images_dir = scene_path / "train" / "images"

    sparse_out = output_path / "sparse" / "0"
    input_out = output_path / "images"

    sparse_out.mkdir(parents=True, exist_ok=True)
    input_out.mkdir(parents=True, exist_ok=True)

    train_names = set(p.name for p in images_dir.iterdir())

    all_cameras = read_cameras_bin(sparse_in / "cameras.bin")
    all_images = read_images_bin(sparse_in / "images.bin")

    train_images = [img for img in all_images if img["name"] in train_names]
    train_ids = {img["id"] for img in train_images}
    used_cam_ids = {img["camera_id"] for img in train_images}
    used_cameras = [c for c in all_cameras if c["id"] in used_cam_ids]

    train_pt_ids = set(
        int(pid) for img in train_images
        for pid in img["point3D_ids"]
        if pid >= 0
    )

    points3D = []
    points_path = sparse_in / "points3D.bin"
    if points_path.exists():
        all_points = read_points3D_bin(points_path)
        for pt in all_points:
            if pt["id"] in train_pt_ids:
                track_img_ids = pt["image_ids"]
                track_pt_idxs = pt["point2D_idxs"]
                mask = np.isin(track_img_ids, list(train_ids))
                if mask.any():
                    pt["image_ids"] = track_img_ids[mask]
                    pt["point2D_idxs"] = track_pt_idxs[mask]
                    points3D.append(pt)

    convert_camera_models(used_cameras)
    write_cameras_txt(used_cameras, sparse_out / "cameras.txt")
    write_images_txt(train_images, sparse_out / "images.txt")
    write_points3D_txt(points3D, sparse_out / "points3D.txt")

    for img in train_images:
        src = images_dir / img["name"]
        if src.exists():
            shutil.copy2(src, input_out / img["name"])

    print(f"  Converted {len(used_cameras)} cameras, {len(train_images)} images, {len(points3D)} points")


COLMAP_MODEL_TO_PINHOLE = {
    "SIMPLE_RADIAL":  {"num_params": 4, "extract": lambda p: [p[0], p[0], p[1], p[2]]},
    "RADIAL":         {"num_params": 4, "extract": lambda p: [p[0], p[1], p[2], p[3]]},
    "SIMPLE_PINHOLE": {"num_params": 4, "extract": lambda p: [p[0], p[0], p[1], p[2]]},
    "PINHOLE":        {"num_params": 4, "extract": lambda p: list(p)},
}

def convert_camera_models(cameras):
    for cam in cameras:
        model = cam["model_name"]
        if model in COLMAP_MODEL_TO_PINHOLE:
            entry = COLMAP_MODEL_TO_PINHOLE[model]
            cam["params"] = entry["extract"](cam["params"])
            cam["model_name"] = "PINHOLE"
        else:
            raise ValueError(f"Unknown camera model: {model}")

def write_cameras_txt(cameras, path: Path):
    with open(path, "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: {len(cameras)}\n")
        for cam in cameras:
            params_str = " ".join(f"{p:.6f}" for p in cam["params"])
            f.write(f"{cam['id']} {cam['model_name']} {cam['width']} {cam['height']} {params_str}\n")


def write_images_txt(images, path: Path):
    with open(path, "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(images)}\n")
        for img in images:
            q = img["qvec"]
            t = img["tvec"]
            f.write(f"{img['id']} {q[0]:.16f} {q[1]:.16f} {q[2]:.16f} {q[3]:.16f} "
                    f"{t[0]:.16f} {t[1]:.16f} {t[2]:.16f} {img['camera_id']} {img['name']}\n")
            points_str = []
            for i in range(len(img["xys"])):
                x, y = img["xys"][i]
                pt3d_id = img["point3D_ids"][i]
                if pt3d_id >= 0:
                    points_str.append(f"{x:.6f} {y:.6f} {pt3d_id}")
            f.write(" ".join(points_str) + "\n" if points_str else "\n")


def write_points3D_txt(points3D, path: Path):
    with open(path, "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write(f"# Number of points: {len(points3D)}\n")
        for pt in points3D:
            track_str = " ".join(
                f"{img_id} {pt2d_idx}"
                for img_id, pt2d_idx in zip(pt["image_ids"], pt["point2D_idxs"])
            )
            x, y, z = pt["xyz"]
            r, g, b = pt["rgb"]
            f.write(f"{pt['id']} {x:.6f} {y:.6f} {z:.6f} {r} {g} {b} {pt['error']:.6f} {track_str}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_name", required=True)
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--output_root", default=None)
    args = parser.parse_args()

    if args.data_root is None:
        import yaml
        config_path = Path(__file__).resolve().parent.parent / "config.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)
        data_root = Path(config["data_root"])
        output_root = Path(config["output_root"])
    else:
        data_root = Path(args.data_root)
        output_root = Path(args.output_root)

    set_name = "public_set"
    scene_path = data_root / set_name / args.scene_name
    out = output_root / "converted" / args.scene_name

    print(f"Converting {scene_path} -> {out}")
    convert_scene(scene_path, out)
