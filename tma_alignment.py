"""
HE ↔ MxIF Slice Registration (Refactored)
-------------------------------------------------
Feature overview:
1) Read HE and MxIF (OME-TIFF) images, parse pixel size and channel info from OME-XML.
2) Perform StarDist nuclei segmentation (HE uses 2D_versatile_he; MxIF uses DAPI + 2D_versatile_fluo).
3) Extract nuclei centroids, convert to μm coordinates, and run CPD rigid registration.
4) Convert the rigid transform in μm coordinates to an OpenCV pixel affine matrix and warp HE into the MxIF pixel grid.
5) Optionally save side-by-side visualization & the registered HE image (with pixel resolution metadata).
"""
from __future__ import annotations
import os
import sys
import copy
import argparse
import xml.etree.ElementTree as ET
from dataclasses import dataclass
import typing
from typing import Tuple, Optional
import typing_extensions
typing_extensions.Generic = typing.Generic
import numpy as np
import cv2
import tifffile as tf
from probreg import cpd
from stardist.models import StarDist2D
from csbdeep.utils import normalize
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# =============== 基础工具：类型与可视化 ===============

def to_uint8_gray(img: np.ndarray) -> np.ndarray:
    """Robustly normalize a grayscale image of any bit depth/type to uint8 [0,255].
    Uses the 1~99.9 percentile range for dynamic stretching to avoid outlier influence.
    """
    x = img.astype(np.float32)
    p1, p999 = np.percentile(x, (1, 99.9))
    if not np.isfinite(p1) or not np.isfinite(p999) or p999 <= p1:
        x = np.clip(x, 0, None)
        vmax = float(x.max()) if x.size else 1.0
        p1, p999 = 0.0, max(vmax, 1.0)
    x = np.clip((x - p1) / (p999 - p1 + 1e-8), 0, 1)
    return (x * 255.0).astype(np.uint8)


def ensure_3ch_uint8(img: np.ndarray) -> np.ndarray:
    """Ensure the output is an HxWx3 uint8 image (for visualization/saving)."""
    if img.ndim == 2:
        img = cv2.merge([img, img, img])
    elif img.ndim == 3 and img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    return img.astype(np.uint8)


def plot_centroids_trans(s_points_um, t_points_um, t_s_points_um,
                         legend, title, out_dir, fn):
    """
    Plot in μm coordinates:
    - s_points_um: source point cloud (HE)   (N,2) = (x_um, y_um)
    - t_points_um: target point cloud (MxIF) (M,2)
    - t_s_points_um: transformed source cloud R*X+t (N,2)
    """
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(dpi=300)
    
    plt.scatter(s_points_um[:, 0], s_points_um[:, 1], c='r', marker="^", s=1, alpha=0.25)
    plt.scatter(t_points_um[:, 0], t_points_um[:, 1], c='g', marker=".", s=1, alpha=0.7)
    plt.scatter(t_s_points_um[:, 0], t_s_points_um[:, 1], c='b', marker="*", s=1, alpha=0.7)

    r_patch = mpatches.Patch(color='red',   label=legend[0])
    g_patch = mpatches.Patch(color='green', label=legend[1])
    b_patch = mpatches.Patch(color='blue',  label=legend[2])
    plt.legend(handles=[r_patch, g_patch, b_patch])
    plt.axis('equal')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, fn))
    plt.close()

# =============== OME-XML parsing utilities ===============

@dataclass
class PixelSize:
    px_um_x: float  # X-direction pixel size (μm/px)
    px_um_y: float  # Y-direction pixel size (μm/px)


def read_ome_xml(path: str) -> Tuple[np.ndarray, str]:
    """Read highest-resolution series data and OME-XML text from an OME-TIFF.
    Returns: (arr, ome_xml)
    - arr shape can be (C,H,W) or (H,W,C) or (H,W)
    """
    with tf.TiffFile(path) as tif:
        series = tif.series[0]
        arr = series.asarray()
        ome_xml = tif.ome_metadata
    return arr, ome_xml


def _parse_pixel_size_from_ome_xml(ome_xml: str):
    try:
        root = ET.fromstring(ome_xml)
        ns = {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}
        pix = root.find('.//ome:Image/ome:Pixels', ns)
        if pix is None: 
            return None
        sx = pix.attrib.get('PhysicalSizeX')
        sy = pix.attrib.get('PhysicalSizeY')
        if sx is None or sy is None:
            return None
        ux = pix.attrib.get('PhysicalSizeXUnit', 'µm')
        uy = pix.attrib.get('PhysicalSizeYUnit', 'µm')
        sx = float(sx); sy = float(sy)
        
        if ux.lower().startswith('nm'): sx /= 1000.0
        if uy.lower().startswith('nm'): sy /= 1000.0
        return PixelSize(px_um_x=sx, px_um_y=sy)
    except Exception:
        return None

def _parse_pixel_size_from_tiff_tags(tif: tf.TiffFile):
    """
    Parse pixel size from TIFF tags:
    - XResolution/YResolution = pixels per unit
    - ResolutionUnit: 2=INCH, 3=CENTIMETER
    μm/px calculation:
      INCH:   μm/px = 25400 / XResolution
      CM:     μm/px = 10000 / XResolution
    """
    try:
        # Use the first page of the highest-resolution series
        page = tif.series[0].pages[0] if hasattr(tif.series[0], 'pages') else tif.pages[0]
        tags = page.tags
        unit_tag = tags.get('ResolutionUnit', None)
        xres_tag = tags.get('XResolution', None)
        yres_tag = tags.get('YResolution', None)
        if unit_tag is None or xres_tag is None or yres_tag is None:
            return None

        unit = int(unit_tag.value)  # 1=None, 2=INCH, 3=CENTIMETER
        xres = float(xres_tag.value)  # pixels per unit
        yres = float(yres_tag.value)

        if unit == 2:   # INCH
            px_um_x = 25400.0 / xres
            px_um_y = 25400.0 / yres
        elif unit == 3: # CENTIMETER
            px_um_x = 10000.0 / xres
            px_um_y = 10000.0 / yres
        else:
            # No declared unit (common for some exported TIFFs) — give up
            return None

        return PixelSize(px_um_x=px_um_x, px_um_y=px_um_y)
    except Exception:
        return None

def parse_pixel_size_any(path: str) -> PixelSize:
    """Prefer OME-XML; if not available, try TIFF tags; otherwise raise an error."""
    with tf.TiffFile(path) as tif:
        # 1) OME-XML
        ome_xml = getattr(tif, 'ome_metadata', None)
        if ome_xml:
            ps = _parse_pixel_size_from_ome_xml(ome_xml)
            if ps is not None:
                return ps
        # 2) TIFF tags
        ps = _parse_pixel_size_from_tiff_tags(tif)
        if ps is not None:
            return ps
    raise RuntimeError(f"Failed to parse pixel size from {path} (both OME-XML and TIFF tags are missing/incomplete).")


def list_channel_names(ome_xml: str) -> list[str]:
    """从 OME-XML 列出通道名（可能为空字符串）。"""
    root = ET.fromstring(ome_xml)
    ns = {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}
    return [c.attrib.get('Name', '') for c in root.findall('.//ome:Image/ome:Pixels/ome:Channel', ns)]


def find_channel_index_by_name(names: list[str], keyword: str, default: int = 0) -> int:
    """List channel names from OME-XML (may be empty strings)."""
    kw = keyword.upper()
    for i, n in enumerate(names):
        if kw in (n or '').upper():
            return i
    return default

# =============== Image loading & preprocessing ===============


def load_he_image(path: str) -> np.ndarray:
    """Read HE image (OME-TIFF), return float32, shape HxW or HxWxC.
    Note: Some HE images are RGB; if grayscale, adapt before segmentation.
    """
    with tf.TiffFile(path) as tif:
        he = tif.series[0].asarray()
    print(f"Original dtype: {he.dtype}, min={he.min()}, max={he.max()}")
    he = he.astype(np.float32)
    return he


def load_mxif_dapi(path: str, dapi_keyword: str = 'DAPI') -> Tuple[np.ndarray, np.ndarray, list[str]]:
    """Read MxIF and extract the DAPI channel.
    Returns (dapi_raw_float32, dapi_norm_float32_0to1, channel_names)
    - dapi_raw keeps the original bit depth (commonly uint16 -> float32)
    - dapi_norm applies large-kernel background correction + 1/99.8 percentile normalization
    """
    arr, ome_xml = read_ome_xml(path)

    if ome_xml:  # ===== OME-TIFF branch =====
        names = list_channel_names(ome_xml)
        c_dapi = find_channel_index_by_name(names, dapi_keyword, default=0)

        # Extract DAPI according to channel axis position
        if arr.ndim == 3 and len(names) > 0 and arr.shape[0] == len(names):  # (C,H,W)
            dapi_raw = arr[c_dapi].astype(np.float32)
        elif arr.ndim == 3 and len(names) > 0 and arr.shape[-1] == len(names):  # (H,W,C)
            dapi_raw = arr[..., c_dapi].astype(np.float32)
        else:  # Fallback for irregular structures
            if arr.ndim == 3:
                dapi_raw = arr[0].astype(np.float32) if arr.shape[0] < arr.shape[-1] else arr[..., 0].astype(np.float32)
            else:
                dapi_raw = arr.astype(np.float32)

    else:  # ===== Standard TIFF branch =====
        names = []
        with tf.TiffFile(path) as tif:
            series = tif.series[0]
            arr = series.asarray()
            # Assume (C,H,W) or (H,W,C) or multi-page
            if arr.ndim == 3 and arr.shape[0] <= 10:  # likely (C,H,W)
                c_dapi = 0  # without names, default to channel 0
                dapi_raw = arr[c_dapi].astype(np.float32)
            elif arr.ndim == 3 and arr.shape[-1] <= 10:  # likely (H,W,C)
                c_dapi = 0
                dapi_raw = arr[..., c_dapi].astype(np.float32)
            else:
                # Multi-page TIFF
                page0 = tif.pages[0].asarray()
                dapi_raw = page0.astype(np.float32)

    # Background correction + normalization
    bg = cv2.GaussianBlur(dapi_raw, (0, 0), 50)
    dapi_corr = np.clip(dapi_raw - bg, 0, None)
    p1, p998 = np.percentile(dapi_corr, (1, 99.8))
    dapi_norm = np.clip((dapi_corr - p1) / (p998 - p1 + 1e-8), 0, 1).astype(np.float32)
    return dapi_raw, dapi_norm, names

# =============== StarDist segmentation & centroid extraction ===============

def segment_he_nuclei(he_img: np.ndarray) -> np.ndarray:
    """Run StarDist on HE to segment nuclei; output centroids (N,2) as (y,x) [pixels].
    - If he_img is grayscale, stack to 3 channels to meet 'YXC' requirement for 2D_versatile_he.
    """
    he_for_net = he_img
    if he_img.ndim == 2:
        he_for_net = np.stack([he_img, he_img, he_img], axis=-1)  # H,W -> H,W,3
    he_norm = normalize(he_for_net)
    model = StarDist2D.from_pretrained('2D_versatile_he')
    lbl, _ = model.predict_instances_big(
        he_norm, axes='YXC',
        prob_thresh=0.5, block_size=2048, min_overlap=128, context=128
    )
    return get_centroids_from_label(lbl)


def segment_mxif_nuclei(dapi_norm: np.ndarray) -> np.ndarray:
    """Run StarDist on DAPI to segment nuclei; output centroids (N,2) as (y,x) [pixels]."""
    model = StarDist2D.from_pretrained('2D_versatile_fluo')
    lbl, _ = model.predict_instances_big(
        dapi_norm, axes='YX',
        block_size=1024, min_overlap=128, context=128,
        prob_thresh=0.30, nms_thresh=0.30
    )
    return get_centroids_from_label(lbl)


def get_centroids_from_label(lbl: np.ndarray) -> np.ndarray:
    """Extract centroids from a label image; return (N,2) = (y,x) pixel coordinates."""
    from skimage.measure import regionprops_table
    props = regionprops_table(lbl, properties=('centroid',))
    cent = np.c_[props['centroid-0'], props['centroid-1']]
    return cent.astype(np.float32)

# =============== CPD registration & matrix conversion ===============


def to_um_xy(centroids_yx: np.ndarray, px_um: PixelSize) -> np.ndarray:
    """(y,x)[px] → (x,y)[μm], accounting for potentially different X/Y pixel sizes."""
    x_px = centroids_yx[:, 1]
    y_px = centroids_yx[:, 0]
    x_um = x_px * float(px_um.px_um_x)
    y_um = y_px * float(px_um.px_um_y)
    return np.c_[x_um, y_um].astype(np.float32)


def run_cpd_rigid(src_um_xy: np.ndarray, dst_um_xy: np.ndarray,
                  maxiter: int = 100, tol: float = 1e-7):
    """Run rigid CPD in μm coordinates; return tf_param."""
    tf_param, sigma2, q = cpd.registration_cpd(
        src_um_xy, dst_um_xy, update_scale=False,
        maxiter=maxiter, tol=tol
    )
    return tf_param


def rigid_cpd_to_cv2_affine(tf_param, ps: PixelSize, pt: PixelSize) -> np.ndarray:
    """Convert rigid CPD (in μm) to an OpenCV 2x3 pixel affine matrix.
    Formula: A = diag(ps/pt) @ R,  b = t / pt
    - If X/Y pixel sizes differ, use a diagonal scaling matrix rather than a scalar.
    """
    R = np.asarray(tf_param.rot, dtype=np.float64)[:2, :2]
    t = np.asarray(tf_param.t,   dtype=np.float64).reshape(2)

    sx = float(ps.px_um_x) / float(pt.px_um_x)
    sy = float(ps.px_um_y) / float(pt.px_um_y)
    S = np.diag([sx, sy])  

    A = S @ R               # 2x2
    b = np.array([t[0]/pt.px_um_x, t[1]/pt.px_um_y], dtype=np.float64)  

    M = np.array([[A[0,0], A[0,1], b[0]],
                  [A[1,0], A[1,1], b[1]]], dtype=np.float32)
    return M

# =============== Saving & visualization ===============

def estimate_bg_from_border(img_rgb: np.ndarray, border: int = 50) -> tuple:
    """Sample background color (RGB median) from the four borders."""
    if img_rgb.ndim == 2:
        img_rgb = cv2.merge([img_rgb, img_rgb, img_rgb])
    h, w = img_rgb.shape[:2]
    rim = np.zeros((h, w), dtype=bool)
    b = min(border, h//4, w//4)  
    rim[:b, :] = rim[-b:, :] = True
    rim[:, :b] = rim[:, -b:] = True
    col = np.median(img_rgb[rim], axis=0)
    return tuple(np.clip(col, 0, 255).astype(np.uint8).tolist())


def save_transformed_HE(he_img: np.ndarray, M: np.ndarray,
                        target_shape_wh: Tuple[int,int], target_px_um: PixelSize,
                        out_path: str) -> np.ndarray:
    """Warp HE to the target grid using pixel affine matrix M and save as OME-TIFF.
    - he_img: original HE image (grayscale or color, any dtype)
    - M: 2x3 OpenCV affine matrix (pixel coordinates)
    - target_shape_wh: (W,H) target image size (note the order!)
    - target_px_um: target pixel size (μm/px), used to write resolution metadata
    - out_path: output file path
    Returns: saved uint8 RGB image (with background replaced by sampled paper color)
    """
    W, H = target_shape_wh

    bg_color = estimate_bg_from_border(he_img)


    # Ensure 3-channel uint8
    if he_img.ndim == 2:
        print(f"HE image is single-channel")
        he_img = cv2.merge([he_img, he_img, he_img])
    if he_img.dtype != np.uint8:
        print(f"HE image is not uint8; casting to uint8 for saving/visualization")
        he_img = he_img.astype(np.uint8)

    affined = cv2.warpAffine(src=he_img, M=M, dsize=(W, H),flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=bg_color)

    # Save with resolution metadata
    res_xy = (10000.0 / target_px_um.px_um_x, 10000.0 / target_px_um.px_um_y)
    tf.imwrite(out_path, affined, photometric='rgb',
               resolution=res_xy, resolutionunit='CENTIMETER')


def save_side_by_side(mxif_img: np.ndarray, he_img: np.ndarray,
                      out_path: str, target_px_um: PixelSize) -> None:
    """Save a side-by-side visualization (left: MxIF/DAPI; right: registered HE).
    Inputs can be float or uint16.
    """
    mx_u8 = ensure_3ch_uint8(to_uint8_gray(mxif_img))
    he_u8 = ensure_3ch_uint8(to_uint8_gray(he_img))
    H, W = mx_u8.shape[:2]
    canvas = np.zeros((H, W*2, 3), dtype=np.uint8)
    canvas[:, 0:W, :] = mx_u8
    canvas[:, W:2*W, :] = he_u8
    res_xy = (10000.0 / target_px_um.px_um_x, 10000.0 / target_px_um.px_um_y)
    tf.imwrite(out_path, canvas, photometric='rgb', resolution=res_xy, resolutionunit='CENTIMETER')

# =============== Main pipeline ===============

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-is','--he_img_fn', required=True, help='HE OME-TIFF 路径')
    parser.add_argument('-it','--mxif_img_fn', required=True, help='MxIF OME-TIFF 路径')
    parser.add_argument('-o','--output_dir', default=os.getcwd(), help='输出目录')
    parser.add_argument('-v','--verbose', action='store_true', help='输出调试图')
    args = parser.parse_args()

    he_path, mxif_path = args.he_img_fn, args.mxif_img_fn
    outdir = args.output_dir
    os.makedirs(outdir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(he_path))[0]

    # Read images and (if any) OME metadata
    he_img = load_he_image(he_path)
    dapi_raw, dapi_norm, mxif_chan_names = load_mxif_dapi(mxif_path, dapi_keyword='DAPI')

    # Read pixel sizes from metadata (fallback to manual if needed)
    #ps_he = parse_pixel_size_any(he_path) # In the HE images from Brenna, OME-XML and TIFF lack pixel_size; must assign manually
    ps_he = PixelSize(0.2628140277841161,0.2628140277841161)
    
    ps_mx = parse_pixel_size_any(mxif_path)

    if ps_he is None:
        raise RuntimeError('Failed to parse pixel size for HE from OME-XML/TIFF metadata.')
    if ps_mx is None:
        raise RuntimeError('Failed to parse pixel size for MxIF from OME-XML/TIFF metadata.')
    
    print(f"HE_Pixel_Size:{ps_he}")
    print(f"Multi_Plex_Pixel_Size:{ps_mx}")

    # --- Segmentation & centroids (pixel YX) ---
    he_centroids_yx  = segment_he_nuclei(he_img)
    mx_centroids_yx  = segment_mxif_nuclei(dapi_norm)

    if args.verbose:
        print(f'HE cells: {len(he_centroids_yx)}  MxIF cells: {len(mx_centroids_yx)}')

    # --- Pixels → μm (x,y) ---
    he_um_xy  = to_um_xy(he_centroids_yx, ps_he)
    mx_um_xy  = to_um_xy(mx_centroids_yx, ps_mx)

    # --- CPD rigid registration (in μm space）---
    tf_param = run_cpd_rigid(he_um_xy, mx_um_xy, maxiter=100, tol=1e-7)
    he_um_xy_aligned = tf_param.transform(he_um_xy)

    # Rough RMSE evaluation (nearest-neighbor)
    if args.verbose:
        from sklearn.neighbors import NearestNeighbors
        def rmse(a,b):
            nn = NearestNeighbors(n_neighbors=1).fit(b)
            d,_ = nn.kneighbors(a)
            return float(np.sqrt((d**2).mean()))
        print('RMSE before (μm):', rmse(he_um_xy, mx_um_xy), ' after (μm):', rmse(he_um_xy_aligned, mx_um_xy))

    # --- μm rigid → pixel affine (OpenCV) ---
    M_cv2 = rigid_cpd_to_cv2_affine(tf_param, ps_he, ps_mx)

    # --- Warp HE to MxIF pixel grid ---
    H, W = dapi_norm.shape[:2]
    he_warp = cv2.warpAffine(he_img, M_cv2, (W, H), flags=cv2.INTER_LINEAR)

    # --- save side-by-side visualization ---
    if args.verbose:
        sbs_path = os.path.join(outdir, f'side_by_side_{stem}.tif')
        save_side_by_side(dapi_norm, he_warp, sbs_path, ps_mx)
    # --- Save “centroid scatter before/after registration” ---
    if args.verbose:
        title  = "Cell Centroids Before and after alignment"
        legend = ["HE cells", "MxIF cells", "trans_HE"]
        fn     = f"log_{stem}_centroids_alignment.png"   
        plot_centroids_trans(
            he_um_xy,           # original HE (x,y) in μm
            mx_um_xy,           # original MxIF (x,y) in μm
            he_um_xy_aligned,   # transformed HE (x,y) in μm
            legend, title, outdir, fn
        )

    # --- Save registered HE ---
    out_he_path = os.path.join(outdir, f'aligned_HE_{stem}.tif')
    save_transformed_HE(he_img=he_img, M=M_cv2,
                        target_shape_wh=(W, H), target_px_um=ps_mx,
                        out_path=out_he_path)


if __name__ == '__main__':
    main()
