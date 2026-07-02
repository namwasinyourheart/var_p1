import struct
import numpy as np
from pathlib import Path
from typing import Dict, List


CameraModels = {
    0: "SIMPLE_PINHOLE",
    1: "PINHOLE",
    2: "SIMPLE_RADIAL",
    3: "RADIAL",
    4: "OPENCV",
    5: "OPENCV_FISHEYE",
    6: "FULL_OPENCV",
    7: "FOV",
    8: "SIMPLE_RADIAL_FISHEYE",
    9: "RADIAL_FISHEYE",
    10: "THIN_PRISM_FISHEYE",
}

# number of params per camera model (in order: fx/f, fy, cx, cy, k1, k2, p1, p2, ...)
CameraModelNumParams = {
    0: 3,  # SIMPLE_PINHOLE: f, cx, cy
    1: 4,  # PINHOLE: fx, fy, cx, cy
    2: 4,  # SIMPLE_RADIAL: f, cx, cy, k
    3: 5,  # RADIAL: f, cx, cy, k1, k2
    4: 8,  # OPENCV: fx, fy, cx, cy, k1, k2, p1, p2
    5: 8,  # OPENCV_FISHEYE: ...
    6: 12,
    7: 5,
    8: 4,
    9: 5,
    10: 12,
}


def read_cameras_bin(path: Path) -> List[Dict]:
    cameras = []
    with open(path, "rb") as f:
        num = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num):
            cam_id = struct.unpack("<I", f.read(4))[0]
            model_id = struct.unpack("<i", f.read(4))[0]
            width = struct.unpack("<Q", f.read(8))[0]
            height = struct.unpack("<Q", f.read(8))[0]
            num_params = CameraModelNumParams.get(model_id, 4)
            raw = f.read(8 * num_params)
            params = struct.unpack("<" + "d" * num_params, raw)
            cameras.append({
                "id": cam_id,
                "model": model_id,
                "model_name": CameraModels.get(model_id, "UNKNOWN"),
                "width": width,
                "height": height,
                "params": np.array(params, dtype=np.float64),
            })
    return cameras


def read_images_bin(path: Path) -> List[Dict]:
    images = []
    with open(path, "rb") as f:
        num = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num):
            buf = f.read(64)
            parts = struct.unpack("<I4d3dI", buf)
            img_id = parts[0]
            qvec = np.array(parts[1:5], dtype=np.float64)
            tvec = np.array(parts[5:8], dtype=np.float64)
            cam_id = parts[8]

            name_buf = []
            while True:
                c = f.read(1)
                if not c or c == b"\x00":
                    break
                name_buf.append(c)
            name = b"".join(name_buf).decode()

            npts = struct.unpack("<Q", f.read(8))[0]

            if npts > 0:
                raw = f.read(24 * npts)
                dt = np.dtype([("x", np.float64), ("y", np.float64), ("id", np.int64)])
                parsed = np.frombuffer(raw, dtype=dt)
                xys = np.column_stack([parsed["x"], parsed["y"]])
                pt_ids = parsed["id"].copy()
            else:
                xys = np.empty((0, 2), dtype=np.float64)
                pt_ids = np.empty(0, dtype=np.int64)

            images.append({
                "id": img_id,
                "qvec": qvec,
                "tvec": tvec,
                "camera_id": cam_id,
                "name": name,
                "xys": xys,
                "point3D_ids": pt_ids,
            })
    return images


def read_points3D_bin(path: Path) -> List[Dict]:
    points = []
    with open(path, "rb") as f:
        num = struct.unpack("<Q", f.read(8))[0]
        data = f.read()

    offset = 0
    for _ in range(num):
        pt_id = struct.unpack_from("<Q", data, offset)[0]; offset += 8
        xyz = struct.unpack_from("<3d", data, offset); offset += 24
        rgb = struct.unpack_from("<3B", data, offset); offset += 3
        error = struct.unpack_from("<d", data, offset)[0]; offset += 8
        track_len = struct.unpack_from("<Q", data, offset)[0]; offset += 8

        img_ids = np.empty(track_len, dtype=np.uint32)
        pt2d_idxs = np.empty(track_len, dtype=np.uint32)
        for j in range(track_len):
            img_ids[j] = struct.unpack_from("<I", data, offset)[0]; offset += 4
            pt2d_idxs[j] = struct.unpack_from("<I", data, offset)[0]; offset += 4

        points.append({
            "id": pt_id,
            "xyz": np.array(xyz, dtype=np.float64),
            "rgb": np.array(rgb, dtype=np.uint8),
            "error": error,
            "image_ids": img_ids,
            "point2D_idxs": pt2d_idxs,
        })

    return points
