import argparse
import sys
import yaml
from pathlib import Path
import shutil
import torch

from src.convert_colmap import convert_scene
from src.render_test import render_test_views
from src.utils.seed import set_seed

set_seed()


def load_config():
    path = Path(__file__).resolve().parent.parent / "config.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def _find_resume_checkpoint(model_dir: Path) -> str:
    chkpnts = sorted(model_dir.glob("chkpnt*.pth"))
    return str(chkpnts[-1]) if chkpnts else None


def train_scene(scene_name, converted_root, model_root, gsplat_root, train_cfg, seed=42):
    import os
    import sys

    source = converted_root / scene_name
    model_dir = model_root / scene_name
    iterations = train_cfg["iterations"]
    test_iterations = train_cfg.get("test_iterations", [7_000, 30_000])
    save_iterations = train_cfg.get("save_iterations", test_iterations)
    checkpoint_iterations = train_cfg.get("checkpoint_iterations", [])

    final_ply = model_dir / "point_cloud" / f"iteration_{iterations}" / "point_cloud.ply"
    if final_ply.exists():
        print(f"  Final checkpoint ({iterations} iters) found, skipping training")
        return

    start_checkpoint = _find_resume_checkpoint(model_dir)
    if start_checkpoint:
        print(f"  Resuming from checkpoint: {start_checkpoint}")

    set_seed()

    gsplat_dir = str(gsplat_root.resolve())
    orig_cwd = os.getcwd()
    orig_path = sys.path.copy()

    try:
        os.chdir(gsplat_dir)
        sys.path.insert(0, gsplat_dir)

        from arguments import ModelParams, PipelineParams, OptimizationParams

        parser = argparse.ArgumentParser()
        lp = ModelParams(parser)
        op = OptimizationParams(parser)
        pp = PipelineParams(parser)
        parser.add_argument('--detect_anomaly', action='store_true', default=False)
        parser.add_argument("--test_iterations", nargs="+", type=int, default=test_iterations)
        parser.add_argument("--save_iterations", nargs="+", type=int, default=save_iterations)
        parser.add_argument("--quiet", action="store_true")
        parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=checkpoint_iterations)
        parser.add_argument("--start_checkpoint", type=str, default=start_checkpoint)

        args = parser.parse_args([
            "-s", str(source),
            "-m", str(model_dir),
            "--iterations", str(iterations),
        ])
        args.save_iterations = save_iterations
        if iterations not in args.save_iterations:
            args.save_iterations.append(iterations)

        from train import training
        from utils.general_utils import safe_state

        import torch
        original_torch_load = torch.load
        torch.load = lambda f, **kw: original_torch_load(f, weights_only=False, **kw)

        print(f"  Running 3DGS training (in-process)...")
        safe_state(args.quiet)
        set_seed(seed)
        try:
            training(lp.extract(args), op.extract(args), pp.extract(args),
                     args.test_iterations, args.save_iterations,
                     args.checkpoint_iterations, args.start_checkpoint, -1)
        finally:
            torch.load = original_torch_load
    finally:
        os.chdir(orig_cwd)
        sys.path = orig_path


def _train_one_scene_worker(worker_id: int, scene_name, converted_path, model_path, gsplat_root, train_cfg, seed, n_gpus=0):
    import os
    if n_gpus > 0:
        gpu_id = worker_id % n_gpus
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print(f"[worker {worker_id}] {scene_name} -> GPU {gpu_id}")
    else:
        print(f"[worker {worker_id}] {scene_name} -> CPU / single GPU")
    train_scene(scene_name, converted_path, model_path, gsplat_root, train_cfg, seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["public", "private", "all"], default="public")
    parser.add_argument("--stage", choices=["all", "convert", "train", "render", "evaluate"], default="all")
    parser.add_argument("--scene", type=str, default=None)
    parser.add_argument("--force", action="store_true", default=False,
                        help="Delete existing model dir and retrain from scratch")
    parser.add_argument("--continue", action="store_true", default=False, dest="cont",
                        help="Allow re-running existing exp_name (skip config snapshot error)")
    parser.add_argument("--parallel", type=int, default=1,
                        help="Number of scenes to train/render in parallel")
    args = parser.parse_args()

    config = load_config()
    data_root = Path(config["data_root"])
    output_root = Path(config["output_root"])
    gsplat_root = Path(config["gaussian_splatting"])

    seed = config.get("seed", 42)
    set_seed(seed)

    exp_name = config.get("exp_name", "default")
    converted_root = output_root / "converted"
    model_root = output_root / "models" / exp_name
    submission_root = Path(config.get("submission_root", output_root / "submissions" / exp_name))

    converted_root.mkdir(parents=True, exist_ok=True)
    model_root.mkdir(parents=True, exist_ok=True)
    submission_root.mkdir(parents=True, exist_ok=True)

    config_dir = output_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_dst = config_dir / f"{exp_name}.yaml"
    if config_dst.exists():
        if args.cont:
            print(f"  --continue: using existing config snapshot -> {config_dst}")
        else:
            print(f"  Config snapshot already exists -> {config_dst}")
            print(f"  Use --continue to re-run, or a different exp_name.")
            sys.exit(1)
    else:
        shutil.copy2(Path(__file__).resolve().parent.parent / "config.yaml", config_dst)
        print(f"  Saved config snapshot -> {config_dst}")

    if args.split == "all":
        scenes = config["scenes"]["public"] + config["scenes"]["private"]
    else:
        scenes = config["scenes"][args.split]

    if args.scene:
        scenes = [s for s in scenes if s == args.scene]
        if not scenes:
            print(f"Scene '{args.scene}' not found in split '{args.split}'")
            sys.exit(1)

    train_tasks = []

    for scene_name in scenes:
        set_name = "public_set" if scene_name in config["scenes"]["public"] else "private_set1"
        split_name = "public" if scene_name in config["scenes"]["public"] else "private"
        scene_path = data_root / set_name / scene_name

        print(f"\n{'='*60}")
        print(f"[{scene_name}] Processing...")
        print(f"{'='*60}")

        if args.stage in ("all", "convert"):
            out = converted_root / split_name / scene_name
            if not out.exists():
                print("[convert] Converting COLMAP data...")
                convert_scene(scene_path, out)
            else:
                print("[convert] Already converted, skipping")

        if args.stage in ("all", "train"):
            train_cfg = config["train"]
            model_path = model_root / split_name / scene_name
            if args.force and model_path.exists():
                shutil.rmtree(model_path)
                print(f"  --force: deleted {model_path}")
            train_tasks.append(
                (scene_name, converted_root / split_name, model_root / split_name, gsplat_root, train_cfg, seed)
            )

        if args.stage in ("all", "render"):
            print("[render] Rendering test views...")
            render_test_views(
                scene_name, scene_path, model_root / split_name / scene_name, submission_root / split_name,
            )

        if args.stage in ("all", "evaluate") and set_name == "public_set":
            print("[evaluate] Computing metrics...")
            results = evaluate_public_scene(scene_path, submission_root / split_name, scene_name)
            if results:
                print(f"  PSNR: {results['psnr_mean']:.2f}  SSIM: {results['ssim_mean']:.4f}")
                if "lpips_mean" in results:
                    print(f"  LPIPS: {results['lpips_mean']:.4f}")

    if train_tasks:
        parallel = max(1, min(args.parallel, len(train_tasks)))
        if parallel > 1:
            from concurrent.futures import ProcessPoolExecutor, as_completed
            n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
            gpu_names = [torch.cuda.get_device_name(i) for i in range(n_gpus)] if n_gpus else []
            print(f"\n[parallel] {len(train_tasks)} scenes, {parallel} workers, {n_gpus} GPU(s): {gpu_names}")
            with ProcessPoolExecutor(max_workers=parallel) as executor:
                futures = {
                    executor.submit(_train_one_scene_worker, i, *task, n_gpus):
                    task[0] for i, task in enumerate(train_tasks)
                }
                for f in as_completed(futures):
                    name = futures[f]
                    try:
                        f.result()
                    except Exception as e:
                        print(f"\n[ERROR] {name} failed: {e}")
        else:
            for task in train_tasks:
                _train_one_scene_worker(0, *task, n_gpus=0)

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
