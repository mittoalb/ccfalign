# zarr ↔ brainreg

Two scripts that bridge OME-Zarr volumes (X-ray µCT, light-sheet, etc.) and
[BrainGlobe brainreg](https://brainglobe.info/documentation/brainreg/index.html) for
registration to the Allen Mouse Brain CCF. The end result is a multiscale OME-Zarr
in atlas space — ready for Neuroglancer.

- [scripts/zarr_to_nrrd.py](scripts/zarr_to_nrrd.py) — extract a downsampled NRRD from a
  zarr (brainreg input).
- [scripts/apply_brainreg_to_zarr.py](scripts/apply_brainreg_to_zarr.py) — apply
  brainreg's transforms to the original full-resolution zarr, chunk-by-chunk, writing a
  new multiscale OME-Zarr in atlas space.

## Install

```bash
conda create -n bzar -c conda-forge python=3.11 zarr ome-zarr dask scipy numpy nibabel
conda activate bzar
pip install brainreg pynrrd
conda install -c conda-forge niftyreg     # only needed to apply B-spline
```

## Workflow

```bash
# 1. Extract a downsampled NRRD from the zarr (~25 µm to match the atlas)
python scripts/zarr_to_nrrd.py \
    /data/sample.zarr  sample_ds.nrrd \
    --level 2 --factor 9

# 2. Register with brainreg
brainreg sample_ds.nrrd ./brainreg_out \
    -v 24.84 24.84 24.84 \
    --orientation psr \
    --atlas allen_mouse_25um

# 3. Apply brainreg's transforms back to the full-res zarr
python scripts/apply_brainreg_to_zarr.py \
    /data/sample.zarr  ./brainreg_out  out/sample_in_ccf.zarr \
    --atlas allen_mouse_25um --input-level 0 --output-um 25 \
    --chunk 128 --n-pyramid-levels 4
```

The output `out/sample_in_ccf.zarr` is a standard OME-Zarr 0.4 with a 4-level pyramid.
Open it directly in Neuroglancer:

```bash
cd out && python -m http.server 8000 --bind 127.0.0.1
# In Neuroglancer:  zarr://http://127.0.0.1:8000/sample_in_ccf.zarr
```

## Notes

### Step 1: `zarr_to_nrrd.py`

- Reads the OME-Zarr `multiscales` metadata for voxel size, or override with
  `--voxel-um`.
- Uses **strided** decimation (not block-mean). Strided is faster and is fine for the
  atlas registration step — `brainreg` smooths internally during its own multi-resolution
  pyramid.
- Writes NRRD with **mm** spacing in the **LPS** frame — the conventions
  brainreg / NiftyReg / ITK-SNAP all expect.

### Step 2: `brainreg`

- `-v` is voxel size in **z y x µm**.
- `--orientation` is the anatomical orientation of *your* data, expressed as a 3-letter
  code in array-axis order (positive direction of each axis):
  - `a/p` = anterior/posterior, `s/i` = superior/inferior, `l/r` = left/right
  - e.g. `psr` = z↑ posterior, y↑ superior, x↑ right (common for top-down CT acquisition)
  - If the output is flipped, change this and re-run (it's fast).
- Outputs land in `./brainreg_out/niftyreg/`: a `.txt` 4×4 affine and an optional
  `.nii.gz` B-spline control-point grid.

### Step 3: `apply_brainreg_to_zarr.py`

- Composes brainreg's affine with your input/output voxel-size matrices into a single
  output→input voxel transform. For each output chunk, it pulls only the needed input
  bounding box, resamples with `scipy.ndimage.affine_transform`, and (if `reg_resample`
  is on PATH) pipes through NiftyReg for the B-spline residual.
- Writes level 0, then builds coarser pyramid levels by 2× block-mean coarsening of
  level 0 on disk (also chunked, never loads the whole thing).
- Stamps proper OME-Zarr 0.4 multiscale metadata so Neuroglancer / napari / Fiji open
  it directly.

## Output sizes

For a typical mouse-brain volume at 25 µm output spacing:
- Level 0: ~80–150 MB (matches Allen atlas dimensions)
- Levels 1-3: ~10 MB + ~1.5 MB + ~200 KB
- Total: well under 200 MB.

At finer `--output-um` (e.g. 5 µm), level 0 scales as `(25/5)³ = 125×` larger. Plan
disk accordingly.

## Coordinate conventions (gotchas)

- **numpy / OME-Zarr**: axis order is `(z, y, x)`; voxel sizes in **µm**.
- **NRRD / NIfTI / NiftyReg / brainreg**: physical coordinates in **mm**, vector order
  `(x, y, z)`. NIfTI is conventionally RAS; NRRD is conventionally LPS.

The application script keeps everything in physical mm internally and converts
numpy-zyx ↔ NIfTI-xyz at the boundaries via an explicit permutation matrix. brainreg's
affine matrix `.txt` is already in mm and respects the `--orientation` you gave it, so
no additional flips are needed in the script.

## Performance / scaling

- `apply_brainreg_to_zarr.py` is the only step that touches full-resolution data.
- Memory per worker ≈ `chunk_size³ × dtype + bbox_size`, typically < 500 MB,
  independent of total volume size.
- Input zarr chunks are read only where needed (dask + zarr).
- Output zarr is written incrementally — full TB never resides in RAM.
- B-spline application via `reg_resample` adds ~1 s overhead per chunk; without it
  (affine only) it's pure scipy and much faster.

## References

- BrainGlobe brainreg: https://brainglobe.info/documentation/brainreg/index.html
- BrainGlobe atlasapi: https://brainglobe.info/documentation/brainglobe-atlasapi/index.html
- NiftyReg: https://github.com/KCL-BMEIS/niftyreg
- OME-Zarr spec: https://ngff.openmicroscopy.org/0.4/
- Allen CCFv3: https://atlas.brain-map.org/
