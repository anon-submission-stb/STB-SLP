#!/usr/bin/env python3
"""Convert raw SOKE SMPL-X fits to Sign-IDD-origin's 50-joint layout.

Uses the SMPL-X forward pass underlying SOKE's feats2joints/get_coord,
but preserves the raw fits' root/body rotations and per-frame shape.
"""
import argparse
from collections import Counter, OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import gzip
import hashlib
import json
import os
from pathlib import Path
import pickle
import time

import numpy as np
import smplx
from smplx.joint_names import JOINT_NAMES
import torch


FIELDS = {
    "global_orient": ("smplx_root_pose", 3),
    "body_pose": ("smplx_body_pose", 63),
    "left_hand_pose": ("smplx_lhand_pose", 45),
    "right_hand_pose": ("smplx_rhand_pose", 45),
    "jaw_pose": ("smplx_jaw_pose", 3),
    "betas": ("smplx_shape", 10),
    "expression": ("smplx_expr", 10),
}
# Body slots 2:5 connect to hand root 29; slots 5:8 connect to root 8.
# Finger order: thumb, index, middle, ring, pinky (three joints + tip).
JOINT_IDS = [
    55, 12, 16, 18, 20, 17, 19, 21,
    21, 52, 53, 54, 71, 40, 41, 42, 72, 43, 44, 45, 73,
    49, 50, 51, 74, 46, 47, 48, 75,
    20, 37, 38, 39, 66, 25, 26, 27, 67, 28, 29, 30, 68,
    34, 35, 36, 69, 31, 32, 33, 70,
]


class DataUnpickler(pickle.Unpickler):
    """Read arrays and metadata without executing arbitrary pickle globals."""
    def find_class(self, module, name):
        if (module, name) == ("torch._utils", "_rebuild_tensor_v2"):
            return lambda storage, offset, size, stride, *args: {"shape": size}
        if (module, name) == ("torch.storage", "_load_from_bytes"):
            return lambda data: None  # Metadata's unused (1, 1) placeholder.
        if (module, name) == ("collections", "OrderedDict"):
            return OrderedDict
        if module == "numpy" and name in ("ndarray", "dtype"):
            return getattr(np, name)
        if module in ("numpy.core.multiarray", "numpy._core.multiarray") and name == "_reconstruct":
            return np._core.multiarray._reconstruct
        raise pickle.UnpicklingError(f"Unsupported pickle global: {module}.{name}")


def read_sequence(directory, cached_xyz=None):
    if not directory.is_dir():
        return None
    paths = sorted(directory.glob("*.pkl"), key=lambda p: int(p.stem))
    if not paths:
        return None
    frame_ids = [int(p.stem) for p in paths]
    if len(set(frame_ids)) != len(frame_ids):
        raise ValueError(f"Duplicate frame IDs: {directory}")
    if cached_xyz is not None and cached_xyz.is_file():
        return frame_ids, None
    arrays = {key: np.empty((len(paths), size), np.float32)
              for key, (_, size) in FIELDS.items()}
    for i, path in enumerate(paths):
        with path.open("rb") as handle:
            record = DataUnpickler(handle).load()
        for key, (source_key, size) in FIELDS.items():
            value = np.asarray(record[source_key])
            if value.shape != (size,) or not np.isfinite(value).all():
                raise ValueError(f"Invalid {source_key}: {path}")
            arrays[key][i] = value
    return frame_ids, arrays


@torch.inference_mode()
def convert(model, arrays, device, batch_size):
    count = len(arrays["body_pose"])
    chunks = []
    for start in range(0, count, batch_size):
        inputs = {k: torch.as_tensor(v[start:start + batch_size], device=device)
                  for k, v in arrays.items()}
        zeros = torch.zeros((len(inputs["body_pose"]), 3), device=device)
        output = model(**inputs, leye_pose=zeros, reye_pose=zeros,
                       transl=zeros, return_verts=False)
        # Original model coordinates, centered on the anatomical neck.
        xyz = output.joints[:, JOINT_IDS] - output.joints[:, 12:13]
        chunks.append(xyz.cpu().numpy())
    xyz = np.concatenate(chunks).astype(np.float32, copy=False)
    if xyz.shape != (count, 50, 3) or not np.isfinite(xyz).all():
        raise ValueError("Invalid XYZ output")
    if not np.all(xyz[:, 1] == 0):
        raise ValueError("Neck centering failed")
    return xyz


def sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true", help="Reuse existing XYZ; source poses must be unchanged")
    parser.add_argument("--limit", type=int, help="Smoke test: first N samples per split")
    args = parser.parse_args()
    output_dir = args.output or args.data / "sign_idd_xyz"
    output_dir.mkdir(parents=True, exist_ok=True)
    # A fresh output prevents accidental mixing of coordinate conventions.
    previous = None
    if any(output_dir.iterdir()):
        if not args.resume:
            raise FileExistsError(f"Output must be empty, or pass --resume: {output_dir}")
        previous = json.loads((output_dir / "metadata.json").read_text())
    torch.set_num_threads(4)
    model = smplx.SMPLX(str(args.model), gender="neutral", use_pca=False,
                       flat_hand_mean=False, use_face_contour=True).to(args.device).eval()
    metadata = {
        "status": "running",
        "source": str(args.data.resolve()),
        "method": "raw SMPL-X forward pass; all source rotations, shape and expression retained",
        "reference": "https://github.com/2000ZRL/SOKE/blob/main/vis_mesh.py",
        "model": str(args.model.resolve()), "model_sha256": sha256(args.model),
        "torch_version": torch.__version__, "numpy_version": np.__version__,
        "smplx_module": smplx.__file__,
        "layout": "Sign-IDD-origin: body(8), right hand(21), left hand(21)",
        "shape": ["T", 50, 3], "dtype": "float32", "units": "meters",
        "origin": "anatomical neck, SMPL-X joint 12, separately for each frame",
        "axes": "SMPL-X output axes with source global_orient retained; no axis flip or rescaling",
        "camera_translation": "not applied; cancels under neck centering",
        "normalization": "no mean/std normalization or denormalization; inputs are raw parameters",
        "head_slot": "nose landmark (SMPL-X joint 55)",
        "hand_mean": "flat_hand_mean=False, matching SOKE",
        "joint_mapping": [{"index": i, "smplx_index": j, "name": JOINT_NAMES[j]}
                          for i, j in enumerate(JOINT_IDS)],
        "counter": "i / actual_frame_count, 0 <= i < actual_frame_count",
        "missing_frames": "retain existing frames in numeric order, no interpolation; IDs in manifest.jsonl",
        "limit_per_split": args.limit,
    }
    if previous is not None:
        for key in ("source", "method", "model_sha256", "joint_mapping", "origin", "axes", "hand_mean", "limit_per_split"):
            if previous[key] != metadata[key]:
                raise ValueError(f"Resume setting differs: {key}")
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output_dir / "xyz").mkdir(exist_ok=True)
    stats = {}
    vocabulary = Counter()
    started = time.monotonic()
    last_report = started
    write_mode = "w" if args.resume else "x"
    with (output_dir / "manifest.jsonl").open(write_mode, encoding="utf-8") as manifest:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for split, destination in (("train", "train"), ("val", "dev"), ("test", "test")):
                with gzip.open(args.data / f"csl_clean.{split}", "rb") as handle:
                    records = DataUnpickler(handle).load()
                if args.limit is not None:
                    records = records[:args.limit]
                stats[destination] = {"metadata_samples": len(records), "converted": 0,
                                      "missing": [], "frame_count_mismatches": 0, "frames": 0}
                current = stats[destination]
                pending = deque()
                iterator = iter(records)

                def enqueue():
                    record = next(iterator, None)
                    if record is not None:
                        name = record["name"]
                        if Path(name).name != name:
                            raise ValueError(f"Invalid sequence name: {name}")
                        pending.append((record, pool.submit(read_sequence, args.data / "csl-daily_pose" / name, output_dir / "xyz" / f"{name}.npy" if args.resume else None)))

                for _ in range(args.workers * 2):
                    enqueue()
                with ExitStack() as stack:
                    handles = {ext: stack.enter_context((output_dir / f"{destination}.{ext}.partial").open(write_mode, encoding="utf-8"))
                               for ext in ("skels", "gloss", "text", "files")}
                    while pending:
                        record, future = pending.popleft()
                        data = future.result()
                        enqueue()
                        name = record["name"]
                        info = {k: record[k] for k in ("name", "signer", "gloss", "text")}
                        info.update(split=destination, metadata_num_frames=record["num_frames"])
                        if data is None:
                            current["missing"].append(name)
                            info.update(status="missing_pose", num_frames=0)
                        else:
                            frame_ids, arrays = data
                            npy_path = output_dir / "xyz" / f"{name}.npy"
                            if arrays is None:
                                xyz = np.load(npy_path, allow_pickle=False)
                                if xyz.shape != (len(frame_ids), 50, 3) or xyz.dtype != np.float32 or not np.isfinite(xyz).all():
                                    raise ValueError(f"Invalid cached XYZ: {npy_path}")
                            else:
                                xyz = convert(model, arrays, args.device, args.batch_size)
                                with npy_path.with_suffix(".npy.partial").open("wb" if args.resume else "xb") as handle:
                                    np.save(handle, xyz, allow_pickle=False)
                                os.replace(npy_path.with_suffix(".npy.partial"), npy_path)
                            count = len(xyz)
                            frame = np.empty((count, 151), np.float32)
                            frame[:, :150] = xyz.reshape(count, 150)
                            frame[:, 150] = np.arange(count, dtype=np.float32) / count
                            handles["skels"].write(" ".join(f"{value:.6f}" for value in frame.ravel()) + "\n")
                            for ext, key in (("gloss", "gloss"), ("text", "text"), ("files", "name")):
                                value = record[key]
                                if "\n" in value or "\r" in value:
                                    raise ValueError(f"Multiline {key}: {name}")
                                handles[ext].write(value + "\n")
                            if destination == "train":
                                vocabulary.update(record["gloss"].split())
                            current["converted"] += 1
                            current["frames"] += count
                            current["frame_count_mismatches"] += count != record["num_frames"]
                            info.update(status="ok", num_frames=count, frame_ids=frame_ids,
                                        xyz_file=str(npy_path.relative_to(output_dir)))
                        manifest.write(json.dumps(info, ensure_ascii=False) + "\n")
                        now = time.monotonic()
                        if now - last_report >= 30:
                            print(f'{destination}: {current["converted"]:,}/{len(records):,} samples; '
                                  f'{current["frames"]:,} frames; elapsed {now-started:.0f}s', flush=True)
                            last_report = now
                for ext in handles:
                    os.replace(output_dir / f"{destination}.{ext}.partial", output_dir / f"{destination}.{ext}")
                print(destination, json.dumps(current), flush=True)
    (output_dir / "gloss_vocab.txt").write_text("\n".join(sorted(vocabulary, key=lambda x: (-vocabulary[x], x))) + "\n", encoding="utf-8")
    metadata.update(status="complete", splits=stats, elapsed_seconds=round(time.monotonic()-started, 2))
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("COMPLETE", output_dir, flush=True)


if __name__ == "__main__":
    main()
