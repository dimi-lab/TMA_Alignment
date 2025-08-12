# TMA_Alignment

Rigid registration between H&E and multiplex (MxIF/IF) tissue images using nuclei centroids:

- StarDist for nuclei segmentation (HE: `2D_versatile_he`; MxIF: DAPI + `2D_versatile_fluo`)
- Convert centroids to microns (μm) using per-axis pixel size
- CPD rigid registration (via `probreg`)
- Convert μm transform to OpenCV pixel affine matrix and warp HE into the MxIF grid
- Optional side-by-side visualization and centroid alignment plots
- Works with **OME-TIFF** and **standard multi-page/multi-channel TIFF**
- Handles both **uint8** and **uint16** microscopy images (DAPI kept in high bit-depth for segmentation)

------

## Table of Contents

- [Overview](#overview)
- [What’s included](#whats-included)
- [Requirements](#requirements)
- [Installation](#installation)
- [Data expectations](#data-expectations)
- [How it works (pipeline)](#how-it-works-pipeline)
- [Usage](#usage)
- [Outputs](#outputs)
- [Pixel size rules & overrides](#pixel-size-rules--overrides)
- [Troubleshooting](#troubleshooting)
- [Performance tips](#performance-tips)
- [Notes & limitations](#notes--limitations)
- [Cite & credits](#cite--credits)
- [License](#license)

------

## Overview

This tool registers an H&E tile to its corresponding MxIF tile by:

1. Segmenting nuclei on each image (HE model on H&E; **DAPI** model on MxIF).
2. Extracting nuclei centroids and converting them from pixel coordinates (y,x) to **μm** coordinates (x,y) using per-axis pixel size.
3. Running **rigid CPD** (rotation + translation; scale fixed) to align HE → MxIF in μm space.
4. Converting the CPD result to a 2×3 OpenCV **pixel** affine matrix, then warping HE into the MxIF grid.
5. (Optional) Saving side-by-side visualization and centroid scatter for QA.

The code defends against common pitfalls:

- DAPI is often **uint16**; it is kept in float32 and percentile-normalized for StarDist (no premature uint8 clipping).
- For OME-TIFF, channels are identified by **OME-XML** (e.g., “DAPI”).
   For standard TIFF, we fall back to a reasonable default (first channel/page).
- Pixel sizes are read from OME-XML when possible; otherwise we try TIFF tags; if both are missing, you can **manually set** pixel sizes.

------

## What’s included

- **Main script**: one Python file that:
  - Loads HE and MxIF (OME-TIFF or standard TIFF)
  - Parses pixel sizes (OME-XML → TIFF tags → manual fallback)
  - Segments nuclei with StarDist and extracts centroids
  - Runs CPD rigid registration (no scale update)
  - Warps HE into MxIF grid via OpenCV
  - Writes optional QA visuals and final aligned HE

------

## Requirements

See environment.yml.

------

## Data expectations

- **HE image**: OME-TIFF or standard TIFF. Can be RGB (uint8) or grayscale.
- **MxIF image**: OME-TIFF preferred; standard TIFF also supported.
  - DAPI channel should exist. For OME-TIFF, it is found by channel name (“DAPI”) in OME-XML.
  - For standard TIFF without channel names, the code defaults to **channel/page 0**.
- Pixel sizes:
  - Read from OME-XML if available; otherwise attempt from TIFF tags (`XResolution/YResolution` + `ResolutionUnit`).
  - If both are missing, **manually set** via code (see below).

------

## How it works (pipeline)

1. **Load HE** via `tifffile.TiffFile.series[0].asarray()` → float32.
2. **Load MxIF**:
   - If OME-XML is present: read channel names; find **DAPI** by name; extract that channel.
   - If not OME-TIFF: attempt standard TIFF (`(C,H,W)` or `(H,W,C)` or multi-page`) and take channel/page 0 by default.
   - DAPI is kept as float32; perform large-kernel Gaussian background correction and 1/99.8 percentile normalization to [0,1].
3. **StarDist**:
   - HE: if grayscale, expand to 3-channels; `2D_versatile_he`, axes=`YXC`.
   - MxIF: use DAPI normalization; `2D_versatile_fluo`, axes=`YX`.
   - Extract centroids (returns pixel coordinates `(y,x)`).
4. **Scale to μm**:
   - Convert `(y,x)[px]` → `(x,y)[μm]` using per-axis pixel size (`μm/px`).
5. **CPD rigid**:
   - `probreg.cpd.registration_cpd(..., update_scale=False)` on μm coordinates.
6. **Transform conversion**:
   - Convert CPD `(R,t)` in μm to OpenCV pixel affine `M` via `A = diag(ps/pt) @ R`, `b = t / pt`.
7. **Warp**:
   - `cv2.warpAffine(HE, M, (W,H))` using MxIF image size; background color sampled from HE image border.
8. **QA outputs** (optional if `-v`):
   - Side-by-side TIFF (left: DAPI, right: aligned HE)
   - Centroid scatter (before/after) in μm space and RMSE logs.

------

## Usage

```
bash
python your_script.py \
  -is /path/to/HE.tif \
  -it /path/to/MxIF_or_OME.tif \
  -o  /path/to/output_dir \
  -v
```

**Arguments**

- `-is, --he_img_fn` : Path to HE image (TIFF/OME-TIFF)
- `-it, --mxif_img_fn` : Path to MxIF image (TIFF/OME-TIFF)
- `-o,  --output_dir` : Output directory (default: current working dir)
- `-v,  --verbose` : Verbose QA outputs (side-by-side, centroid plots, RMSE)

------

## Outputs

In the output directory:

- `aligned_HE_<HE_basename>.tif` – the registered HE, warped into MxIF pixel grid (RGB uint8, resolution metadata set)
- `side_by_side_<HE_basename>.tif` (if `-v`) – left: DAPI (normalized); right: aligned HE
- `log_<HE_basename>_centroids_alignment.png` (if `-v`) – scatter plot in μm space (HE, MxIF, HE→MxIF)
- Console logs with **RMSE before/after (μm)**

------

## Pixel size rules & overrides

The script attempts to parse pixel size (μm/px) in this order:

1. **OME-XML** (`PhysicalSizeX`, `PhysicalSizeY`, units)
2. **TIFF tags** (`XResolution`, `YResolution`, `ResolutionUnit`=INCH/CM)
3. **Manual override (recommended fallback)**

If metadata is missing or wrong, **set pixel sizes manually** near the top of `main()`:

```python
# Example manual values (adjust to your imaging system)
ps_he = PixelSize(0.2201, 0.2201)
ps_mx = PixelSize(0.325,  0.325)
```

> If both OME-XML and TIFF tags fail, manual values are **required** for correct registration.
>  Incorrect pixel sizes typically cause **translation** errors even when rotation looks right.

