#!/usr/bin/env python3
"""Extract a downsampled NRRD from an OME-Zarr group, suitable for `brainreg`.

Usage:
    python zarr_to_nrrd.py /data/sample.zarr out.nrrd \
        --level 2 --factor 9

The output NRRD has spacing in millimeters, LPS frame, gzip-encoded — the
combination `brainreg` / NiftyReg / ITK-SNAP all expect.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import dask.array as da
import numpy as np
import nrrd
import zarr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("zarr_path", type=Path, help="Input OME-Zarr group.")
    ap.add_argument("out_nrrd", type=Path, help="Output NRRD path.")
    ap.add_argument("--level", type=int, default=2,
                    help="Pyramid level to read (default: 2).")
    ap.add_argument("--factor", type=int, default=9,
                    help="Strided decimation factor (default: 9 ≈ 25 µm "
                         "for a 2.76 µm level 2).")
    ap.add_argument("--dataset", type=str, default=None,
                    help="Dataset path inside the zarr group "
                         "(default: '<level>' string).")
    ap.add_argument("--voxel-um", type=float, default=None,
                    help="Override the source voxel size (µm) if the zarr's "
                         "metadata is wrong/missing. Default: read from "
                         "OME-Zarr coordinateTransformations.")
    args = ap.parse_args()

    root = zarr.open(str(args.zarr_path), mode="r")
    dataset_path = args.dataset if args.dataset else str(args.level)
    arr = root[dataset_path]

    # voxel size: from OME-Zarr metadata, or override
    if args.voxel_um is not None:
        src_um = (args.voxel_um, args.voxel_um, args.voxel_um)
    else:
        try:
            ms = root.attrs["multiscales"][0]
            ds = ms["datasets"][args.level]
            scale = next(t["scale"] for t in ds.get("coordinateTransformations", [])
                         if t["type"] == "scale")
            # assume last 3 axes are spatial (z,y,x) in µm
            src_um = tuple(float(s) for s in scale[-3:])
        except Exception as e:
            sys.exit(f"could not read voxel size from OME-Zarr: {e}\n"
                     f"specify --voxel-um explicitly.")

    print(f"input:  level {args.level} {arr.shape} {arr.dtype} @ {src_um} µm")
    out_um = tuple(s * args.factor for s in src_um)
    print(f"output: stride x{args.factor} → "
          f"~{tuple(s // args.factor for s in arr.shape)} @ {out_um} µm")

    # strided read (lazy → compute at the end)
    sub = da.from_zarr(arr)[..., ::args.factor, ::args.factor, ::args.factor]
    print(f"reading {sub.nbytes / 1e6:.1f} MB output (will read {arr.nbytes / 1e9:.1f} "
          f"GB from disk because strided)...")
    data = sub.compute()
    if data.ndim == 4:
        data = data[0]   # drop channel if present
    print(f"loaded: shape={data.shape}, dtype={data.dtype}, "
          f"size={data.nbytes / 1e6:.1f} MB")

    # write NRRD: LPS frame, mm spacing, (x,y,z) order on disk
    sz_mm, sy_mm, sx_mm = (v * 1e-3 for v in out_um)
    header = {
        "type": str(data.dtype),
        "dimension": 3,
        "space": "left-posterior-superior",
        "sizes": list(data.shape[::-1]),    # NRRD uses (x, y, z)
        "space directions": [
            [sx_mm, 0.0, 0.0],
            [0.0, sy_mm, 0.0],
            [0.0, 0.0, sz_mm],
        ],
        "kinds": ["domain", "domain", "domain"],
        "endian": "little",
        "encoding": "gzip",
        "space origin": [0.0, 0.0, 0.0],
    }
    args.out_nrrd.parent.mkdir(parents=True, exist_ok=True)
    nrrd.write(str(args.out_nrrd), np.transpose(data, (2, 1, 0)), header)
    print(f"wrote: {args.out_nrrd}  ({args.out_nrrd.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
