import csv
import numpy as np
from pathlib import Path
from typing import List, Dict


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = q
    R = np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2],
    ])
    return R


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    qw = np.sqrt(1 + R[0,0] + R[1,1] + R[2,2]) / 2
    qx = (R[2,1] - R[1,2]) / (4 * qw)
    qy = (R[0,2] - R[2,0]) / (4 * qw)
    qz = (R[1,0] - R[0,1]) / (4 * qw)
    return np.array([qw, qx, qy, qz])


def build_camera_to_world(qvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    R = quaternion_to_rotation_matrix(qvec)
    C = tvec
    c2w = np.eye(4)
    c2w[:3, :3] = R
    c2w[:3, 3] = C
    return c2w


def build_world_to_camera(qvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    R = quaternion_to_rotation_matrix(qvec)
    C = tvec
    w2c = np.eye(4)
    w2c[:3, :3] = R.T
    w2c[:3, 3] = -R.T @ C
    return w2c


def read_test_poses_csv(path: Path) -> List[Dict]:
    poses = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            poses.append({
                "image_name": row["image_name"],
                "qvec": np.array([
                    float(row["qw"]),
                    float(row["qx"]),
                    float(row["qy"]),
                    float(row["qz"]),
                ], dtype=np.float64),
                "tvec": np.array([
                    float(row["tx"]),
                    float(row["ty"]),
                    float(row["tz"]),
                ], dtype=np.float64),
                "fx": float(row["fx"]),
                "fy": float(row["fy"]),
                "cx": float(row["cx"]),
                "cy": float(row["cy"]),
                "width": int(row["width"]),
                "height": int(row["height"]),
            })
    return poses
