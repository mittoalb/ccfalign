# zrot

Tools for aligning large OME-Zarr volumes (X-ray microtomography, light-sheet, etc.) to
the Allen Mouse Brain Common Coordinate Framework (CCF), and producing a multiscale
OME-Zarr in atlas space suitable for Neuroglancer.

The package provides:

- An **interactive aligner** (`zrot-align align`) — a napari-based GUI for manual rigid
  alignment, working from a downsampled preview of one pyramid level.
- A **refinement step** (`zrot-align refine`) — elastix-based (residual rigid → affine →
  B-spline) automated fine alignment seeded by the manual rigid.
- A **QC viewer** (`zrot-align inspect`) — napari overlay of the pre-warp / refined
  output / atlas.
- Two **scripts** under `scripts/` that bridge to [BrainGlobe brainreg](https://brainglobe.info/documentation/brainreg/index.html):
  one to extract a downsampled NRRD from a zarr, one to apply brainreg's transforms back
  to the full-resolution zarr as a multiscale OME-Zarr.

## Install

```bash
conda create -n zrot -c conda-forge python=3.11 napari pyqt zarr ome-zarr dask scipy numpy
conda activate zrot
pip install brainglobe-atlasapi typer magicgui itk-elastix pynrrd scikit-image nibabel
pip install -e .
```

For the brainreg-based path (recommended, see below):

```bash
pip install brainreg
conda install -c conda-forge niftyreg     # only needed to apply B-spline
```

## Two paths

There are two ways to use `zrot`. Pick based on how much manual control you need.

### Path A — pure `zrot` (interactive)

```bash
zrot-align align /data/sample.zarr --level 2 --save transform.json \
    --ndisplay 2 --preview-voxels $((256**3)) --no-write-into-zarr
# ... hand-align in napari (translation only in 2D; rotation in 3D), save.

zrot-align refine /data/sample.zarr transform.json out/refine_run --skip-elastix
zrot-align inspect out/refine_run
# ... verify prewarp; loop back to `align` if needed.

zrot-align refine /data/sample.zarr transform.json out/refine_run
# ... runs residual-rigid + affine + bspline elastix; produces TransformParameters.*.txt
zrot-align inspect out/refine_run
```

Outputs: `transform.json` (rigid) plus `out/refine_run/elastix/TransformParameters.{0,1,2}.txt`.

**Limitations to know about:**

- **napari 2D rendering hides out-of-plane rotation.** Do not adjust rotation sliders in
  `--ndisplay 2` — what you see on screen does *not* reflect what gets saved. Use the
  "Refresh prewarp (truth)" button in the right dock to verify rotations correctly, or
  pass `--ndisplay 3` (slower).
- Elastix's deformable (B-spline) stage can over-fit when the joint brain mask is small
  (<10%). If `inspect` shows the result is worse than the prewarp, re-run with
  `--no-bspline`.

### Path B — `brainreg` (recommended for reliable atlas registration)

The interactive path requires careful manual alignment and elastix tuning. For most
applications, [BrainGlobe brainreg](https://brainglobe.info/documentation/brainreg/index.html)
is a more turnkey pipeline (NiftyReg under the hood, atlas-validated parameters).

```bash
# 1. Extract a downsampled NRRD from the zarr
python scripts/zarr_to_nrrd.py \
    /data/sample.zarr  sample_ds.nrrd \
    --level 2 --factor 9

# 2. Register with brainreg
brainreg sample_ds.nrrd ./brainreg_out \
    -v 24.84 24.84 24.84 \
    --orientation psr \
    --atlas allen_mouse_25um

# 3. Apply brainreg's transforms to the original full-res zarr,
#    producing a multiscale OME-Zarr in CCF space
python scripts/apply_brainreg_to_zarr.py \
    /data/sample.zarr  ./brainreg_out  out/sample_in_ccf.zarr \
    --atlas allen_mouse_25um --input-level 0 --output-um 25 \
    --chunk 128 --n-pyramid-levels 4
```

The output `out/sample_in_ccf.zarr` is a standard OME-Zarr 0.4 with a 4-level pyramid,
opens directly in Neuroglancer:

```bash
cd out && python -m http.server 8000 --bind 127.0.0.1
# In Neuroglancer:  zarr://http://127.0.0.1:8000/sample_in_ccf.zarr
```

## CLI reference

```
zrot-align info <zarr>                              # print pyramid table
zrot-align align <zarr> [options]                   # interactive aligner
zrot-align refine <zarr> <transform.json> <out_dir> # elastix fine alignment
zrot-align inspect <out_dir>                        # napari QC overlay
zrot-align apply-chain <moving> <out_dir> <tp>...   # apply elastix chain to a new image
```

Run `zrot-align <command> --help` for flags.

## Scripts

- [scripts/zarr_to_nrrd.py](scripts/zarr_to_nrrd.py) — extract a downsampled NRRD (mm
  spacing, LPS frame, gzip) suitable for `brainreg` / ITK-SNAP / 3D Slicer.
- [scripts/apply_brainreg_to_zarr.py](scripts/apply_brainreg_to_zarr.py) — apply
  brainreg's NiftyReg transforms (affine + B-spline) to the full-resolution zarr,
  chunk-by-chunk, writing a new multiscale OME-Zarr in atlas space.

## Coordinate conventions

- **numpy / OME-Zarr / zrot**: axis order is `(z, y, x)`; voxel sizes in **µm**.
- **NRRD / NIfTI / NiftyReg / brainreg**: physical coordinates in **mm**, `(x, y, z)`
  vector order. NIfTI is conventionally RAS, NRRD is conventionally LPS.

The transform stored in `transform.json` (`zrot.RigidTransform`) is in **physical
micrometres**, mapping sample µm → CCF µm. It's scale-invariant: the same 4×4 matrix is
valid at every pyramid level when composed with that level's voxel size.

NiftyReg's affine files (`.txt`) are 4×4 matrices in mm, mapping input voxel-index →
output mm (forward direction).

## Performance / scaling

- The aligner only loads a single downsampled preview (~30 MB by default) into RAM.
- `refine` operates at ~25 µm (atlas resolution) in RAM (~hundreds of MB).
- `apply_brainreg_to_zarr.py` is the only step that touches full-resolution data, and it
  does so chunk-by-chunk via dask — memory per worker ≈ `chunk_size³ × dtype + bbox_size`
  (typically <500 MB), regardless of total volume size.
- For full-resolution warping of TB-scale zarrs, pyramid output is generated by 2×
  block-mean coarsening of level 0 on disk after it lands. Disk requirement is roughly
  `output_volume_size × 1.15` (pyramid overhead).

## Caveats / known issues

- The interactive aligner's `magicgui` FloatSlider widget has a rendering bug at wide
  ranges; `zrot` ships a custom `FloatRangeSlider` widget that works around it.
- `napari` cannot fully render out-of-plane rotation in 2D slice view. The aligner
  defaults to 2D for speed; use the "Refresh prewarp (truth)" button in the right dock
  to verify rotations honestly, or switch to 3D via `--ndisplay 3`.
- Elastix's B-spline stage requires high mask overlap (>30%) to behave well. If the
  initial rigid is poor, elastix may make things worse — verify with `inspect` between
  `--skip-elastix` and the full run.
- The `apply-brainreg` script assumes brainreg's affine and your input are in the same
  physical frame (mm). The `--orientation` flag you give brainreg must match the actual
  anatomical orientation of your data; if the output is flipped, fix the orientation in
  brainreg and re-run.

## Layout

```
src/zrot/
  align.py          interactive aligner (napari + custom Qt widgets)
  apply.py          metadata + lazy chunked resampler
  atlas.py          Allen CCF loader (via brainglobe-atlasapi)
  cli.py            typer CLI
  io.py             OME-Zarr multiscale reader
  refine.py         elastix refinement + inspect viewer
  transform.py      RigidTransform (4x4 matrix, µm-space)
  params/           bundled elastix parameter files
scripts/
  zarr_to_nrrd.py            zarr  → NRRD
  apply_brainreg_to_zarr.py  brainreg output → multiscale OME-Zarr
```

## References

- BrainGlobe brainreg: https://brainglobe.info/documentation/brainreg/index.html
- BrainGlobe atlasapi: https://brainglobe.info/documentation/brainglobe-atlasapi/index.html
- NiftyReg: https://github.com/KCL-BMEIS/niftyreg
- Elastix: https://elastix.lumc.nl/
- OME-Zarr spec: https://ngff.openmicroscopy.org/0.4/
- Allen CCFv3: https://atlas.brain-map.org/
