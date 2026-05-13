#!/usr/bin/env python3
"""Apply brainreg's NiftyReg transforms to an original OME-Zarr,
producing a multiscale OME-Zarr in atlas (CCF) coordinate space.

What it does
------------
brainreg writes its transforms into `<brainreg_out>/niftyreg/`:
  - one or more affine matrix text files (rigid + affine stages)
  - a control-point grid (CPP, .nii.gz) for the B-spline stage

This script:
  1. Reads brainreg's transforms.
  2. For each output chunk in CCF voxel grid, computes the corresponding
     positions in the input zarr by composing all transforms.
  3. Pulls only the needed sub-region from the input zarr.
  4. Resamples via scipy.ndimage.map_coordinates.
  5. Writes the result as a multiscale OME-Zarr (level 0 at full input
     resolution interpolated onto the CCF grid; coarser levels by 2× block-mean).

Limitations
-----------
- The B-spline (deformable) stage is applied via NiftyReg's `reg_resample`
  CLI per-chunk if available; otherwise the script applies only the affine
  composite and warns. Install niftyreg from conda-forge:
      conda install -c conda-forge niftyreg
  to get B-spline.
- Output spacing matches the atlas (e.g. 25 µm). If you want finer output,
  pass --output-um.

Usage
-----
    python apply_brainreg_to_zarr.py \
        /data/sample.zarr  /path/to/brainreg_out  out_warped.zarr \
        --atlas allen_mouse_25um \
        --input-level 0 \
        --output-um 25 \
        --chunk 128
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import dask
import dask.array as da
import numpy as np
import nibabel as nib
import zarr
from scipy.ndimage import affine_transform, map_coordinates


# ---------------------------------------------------------------------------
# Brainreg / NiftyReg transform loading
# ---------------------------------------------------------------------------
def _find_transform_files(niftyreg_dir: Path) -> dict:
    """Discover what NiftyReg wrote. Returns dict with keys:
       'affine': Path or None
       'cpp':    Path or None  (B-spline control-point grid, .nii or .nii.gz)
       'orientation_matrix': Path or None
    """
    out = {"affine": None, "cpp": None, "orientation_matrix": None}
    if not niftyreg_dir.exists():
        return out
    for p in sorted(niftyreg_dir.iterdir()):
        n = p.name.lower()
        if n.endswith(".txt"):
            # NiftyReg affine outputs are text 4x4 matrices
            if "affine" in n or "rigid" in n:
                out["affine"] = p
        elif n.endswith((".nii", ".nii.gz")):
            if "cpp" in n or "bspline" in n or "control" in n:
                out["cpp"] = p
    return out


def _load_affine_matrix(path: Path) -> np.ndarray:
    """Load a NiftyReg 4x4 affine matrix from a text file."""
    M = np.loadtxt(str(path))
    if M.shape != (4, 4):
        raise ValueError(f"expected 4x4 matrix in {path}, got {M.shape}")
    return M


def _have_reg_resample() -> bool:
    return shutil.which("reg_resample") is not None


# ---------------------------------------------------------------------------
# Coordinate conventions
# ---------------------------------------------------------------------------
# brainreg / NiftyReg / NIfTI conventions:
#   - Coordinates are in millimetres, in physical space (LPS or RAS — RAS for
#     NIfTI by default, LPS for the NRRD we wrote).
#   - The 4x4 affine maps voxel-index (i, j, k, 1) -> physical-mm (x, y, z, 1).
# zrot / OME-Zarr / numpy conventions:
#   - Array indexing is (z, y, x).
#   - Voxel sizes are in µm.
#
# To compose, we keep everything in physical millimetres and convert
# numpy-(z,y,x) <-> physical-(x,y,z) at the boundaries.

_PERM_ZYX_TO_XYZ = np.array([[0, 0, 1, 0],
                              [0, 1, 0, 0],
                              [1, 0, 0, 0],
                              [0, 0, 0, 1]], dtype=float)


def _voxel_to_mm(voxel_um: tuple[float, float, float]) -> np.ndarray:
    """4x4 mapping numpy-(z,y,x) voxel index -> physical-(x,y,z) mm."""
    sz, sy, sx = (v * 1e-3 for v in voxel_um)
    A = np.diag([sx, sy, sz, 1.0])  # acts on (x, y, z, 1)
    return A @ _PERM_ZYX_TO_XYZ


def _mm_to_voxel(voxel_um: tuple[float, float, float]) -> np.ndarray:
    return np.linalg.inv(_voxel_to_mm(voxel_um))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("zarr_in", type=Path, help="Original OME-Zarr group.")
    ap.add_argument("brainreg_out", type=Path,
                    help="Brainreg output directory (must contain niftyreg/).")
    ap.add_argument("zarr_out", type=Path,
                    help="Output OME-Zarr (CCF coords, multiscale).")
    ap.add_argument("--atlas", default="allen_mouse_25um",
                    help="Brainglobe atlas name (default: allen_mouse_25um).")
    ap.add_argument("--input-level", type=int, default=0,
                    help="Input pyramid level to use (default: 0 = full res).")
    ap.add_argument("--input-voxel-um", type=float, default=None,
                    help="Override input voxel size (µm) if zarr metadata "
                         "is wrong/missing.")
    ap.add_argument("--output-um", type=float, default=None,
                    help="Output voxel size in µm. Default: match the atlas.")
    ap.add_argument("--chunk", type=int, default=128, help="Output chunk (cubic).")
    ap.add_argument("--n-pyramid-levels", type=int, default=4,
                    help="How many pyramid levels to build in the output zarr.")
    ap.add_argument("--order", type=int, default=1,
                    help="Interpolation order (0=nearest, 1=linear, 3=cubic).")
    args = ap.parse_args()

    # 1. Load atlas (just for shape + voxel size)
    from brainglobe_atlasapi import BrainGlobeAtlas
    atlas = BrainGlobeAtlas(args.atlas)
    atlas_shape = tuple(atlas.reference.shape)            # (z, y, x)
    atlas_um_native = tuple(float(r) for r in atlas.resolution)
    out_um = (args.output_um,) * 3 if args.output_um else atlas_um_native
    if args.output_um is not None:
        # rescale output shape proportionally
        ratio = atlas_um_native[0] / args.output_um
        out_shape = tuple(int(round(s * ratio)) for s in atlas_shape)
    else:
        out_shape = atlas_shape
    print(f"atlas:   {args.atlas}  shape={atlas_shape}  voxel={atlas_um_native} µm")
    print(f"output:  shape={out_shape}  voxel={out_um} µm")

    # 2. Open input zarr, get voxel size
    root_in = zarr.open(str(args.zarr_in), mode="r")
    arr_in = root_in[str(args.input_level)]
    if args.input_voxel_um is not None:
        in_um = (args.input_voxel_um,) * 3
    else:
        ms = root_in.attrs["multiscales"][0]
        ds = ms["datasets"][args.input_level]
        scale = next(t["scale"] for t in ds.get("coordinateTransformations", [])
                     if t["type"] == "scale")
        in_um = tuple(float(s) for s in scale[-3:])
    print(f"input:   level {args.input_level} {arr_in.shape} {arr_in.dtype} @ {in_um} µm")

    # 3. Load brainreg transforms
    nifty_dir = args.brainreg_out / "niftyreg"
    transforms = _find_transform_files(nifty_dir)
    if transforms["affine"] is None:
        sys.exit(f"no affine .txt found in {nifty_dir}; is this a brainreg output?")
    affine_mm_to_mm = _load_affine_matrix(transforms["affine"])
    print(f"affine:  {transforms['affine'].name}")
    has_bspline = transforms["cpp"] is not None and _have_reg_resample()
    if transforms["cpp"] and not _have_reg_resample():
        print("WARNING: B-spline CPP found but `reg_resample` not on PATH.\n"
              "         Install with:  conda install -c conda-forge niftyreg\n"
              "         Proceeding with affine-only.")
    if has_bspline:
        print(f"bspline: {transforms['cpp'].name} (will use reg_resample per chunk)")

    # 4. Build composite transform: output_voxel -> input_voxel
    #    output_voxel(zyx) → output_mm(xyz) → input_mm(xyz) [affine_inv]
    #                                       → input_voxel(zyx)
    out_v2mm = _voxel_to_mm(out_um)
    in_mm2v = _mm_to_voxel(in_um)
    affine_inv = np.linalg.inv(affine_mm_to_mm)
    M_total = in_mm2v @ affine_inv @ out_v2mm   # 4x4, takes output_voxel -> input_voxel

    # 5. Build the dask graph
    arr_in_da = da.from_zarr(arr_in)
    if arr_in_da.ndim == 4:
        arr_in_da = arr_in_da[0]   # drop channel
    in_shape = arr_in_da.shape
    in_dtype = arr_in_da.dtype

    chunk = (args.chunk,) * 3
    out_template = da.zeros(out_shape, dtype=in_dtype, chunks=chunk)

    A_inv = M_total[:3, :3]
    b_inv = M_total[:3, 3]

    def _resample_block(block, block_info=None):
        loc = block_info[None]["array-location"]
        origin = np.array([loc[0][0], loc[1][0], loc[2][0]], dtype=float)
        sh = block.shape

        # bbox of input voxels we need
        corners_out = np.array([[0,0,0],[sh[0],0,0],[0,sh[1],0],[0,0,sh[2]],
                                [sh[0],sh[1],0],[sh[0],0,sh[2]],
                                [0,sh[1],sh[2]],[sh[0],sh[1],sh[2]]],
                               dtype=float) + origin
        corners_in = (A_inv @ corners_out.T).T + b_inv
        pad = args.order + 2
        in_min = np.maximum(np.floor(corners_in.min(axis=0)).astype(int) - pad, 0)
        in_max = np.minimum(np.ceil(corners_in.max(axis=0)).astype(int) + pad,
                            np.array(in_shape))
        if np.any(in_max <= in_min):
            return np.zeros(sh, dtype=in_dtype)
        sub = arr_in_da[in_min[0]:in_max[0],
                        in_min[1]:in_max[1],
                        in_min[2]:in_max[2]].compute()
        offset = A_inv @ origin + b_inv - in_min
        warped = affine_transform(
            sub, A_inv, offset=offset, output_shape=sh,
            order=args.order, mode="constant", cval=0,
        )
        if has_bspline:
            warped = _apply_bspline_to_chunk(warped, sh, origin, out_um,
                                              transforms["cpp"])
        return warped.astype(in_dtype)

    out_l0 = out_template.map_blocks(_resample_block, dtype=in_dtype, chunks=chunk)

    # 6. Write multiscale OME-Zarr
    print(f"writing level 0 → {args.zarr_out} (lazy, will materialise on store)")
    args.zarr_out.parent.mkdir(parents=True, exist_ok=True)
    if args.zarr_out.exists():
        shutil.rmtree(args.zarr_out)

    store = zarr.DirectoryStore(str(args.zarr_out))
    root_out = zarr.group(store=store)

    # Level 0
    da.to_zarr(out_l0, str(args.zarr_out / "0"), overwrite=True)
    levels_meta = [{"path": "0",
                    "coordinateTransformations": [
                        {"type": "scale", "scale": list(out_um)}]}]
    print("level 0 done.")

    # Coarser levels by 2× block-mean
    cur = out_l0
    cur_um = out_um
    for L in range(1, args.n_pyramid_levels):
        cur = da.coarsen(np.mean, cur, {0: 2, 1: 2, 2: 2}, trim_excess=True
                         ).astype(in_dtype)
        cur_um = tuple(v * 2 for v in cur_um)
        # rechunk to keep chunks reasonable
        cur = cur.rechunk(chunk)
        path = str(L)
        # Read level 0 from the freshly written zarr to avoid recomputing
        l0_z = zarr.open(str(args.zarr_out / "0"), mode="r")
        cur_lazy = da.from_zarr(l0_z)
        for _ in range(L):
            cur_lazy = da.coarsen(np.mean, cur_lazy, {0: 2, 1: 2, 2: 2},
                                  trim_excess=True).astype(in_dtype)
        cur_lazy = cur_lazy.rechunk(chunk)
        da.to_zarr(cur_lazy, str(args.zarr_out / path), overwrite=True)
        levels_meta.append({"path": path,
                            "coordinateTransformations": [
                                {"type": "scale", "scale": list(cur_um)}]})
        print(f"level {L} done @ {cur_um} µm")

    # OME-Zarr metadata
    root_out.attrs["multiscales"] = [{
        "version": "0.4",
        "axes": [
            {"name": "z", "type": "space", "unit": "micrometer"},
            {"name": "y", "type": "space", "unit": "micrometer"},
            {"name": "x", "type": "space", "unit": "micrometer"},
        ],
        "datasets": levels_meta,
    }]
    root_out.attrs["brainreg_source"] = str(args.brainreg_out)
    root_out.attrs["affine_matrix"] = affine_mm_to_mm.tolist()
    print(f"wrote multiscale OME-Zarr: {args.zarr_out}")


def _apply_bspline_to_chunk(affine_warped: np.ndarray,
                            sh: tuple[int, int, int],
                            origin: np.ndarray,
                            out_um: tuple[float, float, float],
                            cpp_path: Path) -> np.ndarray:
    """Apply NiftyReg B-spline CPP to a single chunk via reg_resample.

    Writes the chunk + a per-chunk reference NIfTI to a tempdir, calls
    reg_resample, reads back the result. Slow but correct.
    """
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # Need (x, y, z) ordering and mm spacing for NIfTI (RAS).
        flo = td / "flo.nii.gz"
        ref = td / "ref.nii.gz"
        res = td / "res.nii.gz"
        sz, sy, sx = (v * 1e-3 for v in out_um)
        affine = np.diag([sx, sy, sz, 1.0])
        affine[:3, 3] = (origin[2] * sx, origin[1] * sy, origin[0] * sz)
        nib.save(nib.Nifti1Image(np.transpose(affine_warped, (2, 1, 0)), affine), flo)
        nib.save(nib.Nifti1Image(np.zeros(sh[::-1], dtype=affine_warped.dtype), affine), ref)
        subprocess.run(
            ["reg_resample", "-ref", str(ref), "-flo", str(flo),
             "-trans", str(cpp_path), "-res", str(res), "-inter", "1"],
            check=True, capture_output=True,
        )
        warped = nib.load(str(res)).get_fdata().astype(affine_warped.dtype)
        return np.transpose(warped, (2, 1, 0))


if __name__ == "__main__":
    main()
