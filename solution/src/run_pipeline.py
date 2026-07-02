import argparse
import sys
import yaml
from pathlib import Path

from src.convert_colmap import convert_scene
from src.render_test import render_test_views


def load_config():
    path = Path(__file__).resolve().parent.parent / "config.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def train_scene_subprocess(scene_name, converted_root, model_root, gsplat_root, iterations=30000):
    import subprocess

    source = converted_root / scene_name
    model_dir = model_root / scene_name

    if model_dir.exists() and any(model_dir.iterdir()):
        return

    cmd = [
        sys.executable,
        str(gsplat_root / "train.py"),
        "-s", str(source),
        "-m", str(model_dir),
        "--iterations", str(iterations),
        "--images", "input",
    ]
    print(f"  Running 3DGS training...")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-3000:])
        raise RuntimeError(f"Training failed for {scene_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["public", "private", "all"], default="public")
    parser.add_argument("--stage", choices=["all", "convert", "train", "render", "evaluate"], default="all")
    parser.add_argument("--scene", type=str, default=None)
    args = parser.parse_args()

    config = load_config()
    data_root = Path(config["data_root"])
    output_root = Path(config["output_root"])
    gsplat_root = Path(config["gaussian_splatting"])

    converted_root = output_root / "converted"
    model_root = output_root / "models"
    submission_root = output_root / "submissions"

    converted_root.mkdir(parents=True, exist_ok=True)
    model_root.mkdir(parents=True, exist_ok=True)
    submission_root.mkdir(parents=True, exist_ok=True)

    if args.split == "all":
        scenes = config["scenes"]["public"] + config["scenes"]["private"]
    else:
        scenes = config["scenes"][args.split]

    if args.scene:
        scenes = [s for s in scenes if s == args.scene]
        if not scenes:
            print(f"Scene '{args.scene}' not found in split '{args.split}'")
            sys.exit(1)

    for scene_name in scenes:
        set_name = "public_set" if scene_name in config["scenes"]["public"] else "private_set1"
        scene_path = data_root / set_name / scene_name

        print(f"\n{'='*60}")
        print(f"[{scene_name}] Processing...")
        print(f"{'='*60}")

        if args.stage in ("all", "convert"):
            out = converted_root / scene_name
            if not out.exists():
                print("[convert] Converting COLMAP data...")
                convert_scene(scene_path, out)
            else:
                print("[convert] Already converted, skipping")

        if args.stage in ("all", "train"):
            print("[train] Training 3DGS...")
            train_cfg = config["train"]
            train_scene_subprocess(
                scene_name, converted_root, model_root, gsplat_root,
                iterations=train_cfg["iterations"],
            )

        if args.stage in ("all", "render"):
            print("[render] Rendering test views...")
            render_test_views(
                scene_name, scene_path, model_root / scene_name, submission_root,
            )

        if args.stage in ("all", "evaluate") and set_name == "public_set":
            print("[evaluate] Computing metrics...")
            results = evaluate_public_scene(scene_path, submission_root, scene_name)
            if results:
                print(f"  PSNR: {results['psnr_mean']:.2f}  SSIM: {results['ssim_mean']:.4f}")
                if "lpips_mean" in results:
                    print(f"  LPIPS: {results['lpips_mean']:.4f}")

    print(f"\nDone!")


def evaluate_public_scene(scene_path, submission_root, scene_name):
    import numpy as np
    from PIL import Image
    from src.utils.metrics import psnr, ssim
    from skimage.metrics import structural_similarity as ssim_skimage

    gt_dir = scene_path / "test" / "images"
    pred_dir = submission_root / scene_name

    if not pred_dir.exists():
        return None

    psnr_list, ssim_list = [], []

    for img_path in sorted(gt_dir.iterdir()):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        pred = _find_pred(pred_dir, img_path.name)
        if pred is None:
            continue

        gt = np.array(Image.open(img_path).convert("RGB"))
        pd = np.array(Image.open(pred).convert("RGB"))
        if gt.shape != pd.shape:
            pd = np.array(Image.fromarray(pd).resize((gt.shape[1], gt.shape[0])))

        mse = np.mean((gt.astype(np.float64) - pd.astype(np.float64)) ** 2)
        psnr_list.append(20 * np.log10(255.0 / np.sqrt(mse + 1e-10)))
        ssim_list.append(ssim_skimage(gt, pd, channel_axis=-1, data_range=255))

    return {
        "scene": scene_name,
        "num_images": len(psnr_list),
        "psnr_mean": float(np.mean(psnr_list)),
        "ssim_mean": float(np.mean(ssim_list)),
    } if psnr_list else None


def _find_pred(pred_dir, gt_name):
    stems = [pred_dir / gt_name, pred_dir / Path(gt_name).with_suffix(".png")]
    for p in stems:
        if p.exists():
            return p
    return None


if __name__ == "__main__":
    main()
