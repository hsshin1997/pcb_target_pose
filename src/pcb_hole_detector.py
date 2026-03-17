#!/usr/bin/env python3
from __future__ import annotations
import argparse, os, json, math
from dataclasses import dataclass, asdict
import numpy as np
import cv2
from scipy import ndimage as ndi

@dataclass
class Hole:
    id: int
    center_uv_mm: list[float]
    center_xyz_mm: list[float]
    diameter_mm: float
    circularity: float
    area_px: int
    bbox_xywh: list[int]

def fit_circle_kasa(xy: np.ndarray):
    x = xy[:, 0].astype(np.float64)
    y = xy[:, 1].astype(np.float64)
    A = np.c_[2*x, 2*y, np.ones_like(x)]
    b = x*x + y*y
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy, c = sol
    r = math.sqrt(max(c + cx*cx + cy*cy, 0.0))
    return float(cx), float(cy), float(r)

def robust_plane_fit(points: np.ndarray):
    p = points[:, :3].astype(np.float64)
    c0 = np.median(p, axis=0)
    x0 = p - c0
    samp = x0[np.random.choice(len(x0), min(50000, len(x0)), replace=False)]
    _, _, vh = np.linalg.svd(samp, full_matrices=False)
    n = vh[-1]
    n = n / np.linalg.norm(n)

    d = (p - c0) @ n
    hist, bins = np.histogram(d, bins=256)
    k = int(np.argmax(hist))
    mode_d = 0.5 * (bins[k] + bins[k+1])
    mad = np.median(np.abs(d - mode_d)) + 1e-9
    band = max(0.06, 3.0 * 1.4826 * mad)
    inlier = np.abs(d - mode_d) <= band

    pin = p[inlier]
    cin = np.mean(pin, axis=0)
    xin = pin - cin
    samp = xin[np.random.choice(len(xin), min(50000, len(xin)), replace=False)]
    _, _, vh = np.linalg.svd(samp, full_matrices=False)
    n = vh[-1]
    n = n / np.linalg.norm(n)

    # make normal consistent
    if abs(n[2]) < 0.5 and abs(n[0]) > 0.5:
        if n[0] < 0:
            n = -n
    else:
        if n[2] < 0:
            n = -n

    a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, a); u /= np.linalg.norm(u)
    v = np.cross(n, u); v /= np.linalg.norm(v)

    d = (p - cin) @ n
    hist, bins = np.histogram(d, bins=256)
    k = int(np.argmax(hist))
    mode_d = 0.5 * (bins[k] + bins[k+1])
    mad = np.median(np.abs(d - mode_d)) + 1e-9
    band = max(0.06, 3.0 * 1.4826 * mad)
    return cin, u, v, n, mode_d, band, d

def rasterize_surface(uv: np.ndarray, res_mm: float, pad_mm: float):
    uv_min = uv.min(axis=0) - pad_mm
    ij = np.floor((uv - uv_min) / res_mm).astype(np.int32)
    W = int(ij[:, 0].max() + 2)
    H = int(ij[:, 1].max() + 2)
    occ = np.zeros((H, W), np.uint8)
    occ[ij[:, 1], ij[:, 0]] = 255
    occ = cv2.dilate(occ, np.ones((3, 3), np.uint8), iterations=1)
    occ = cv2.morphologyEx(occ, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    return occ, uv_min

def build_board_mask(occ: np.ndarray, max_hole_diam_mm: float, res_mm: float):
    k = int(round((max_hole_diam_mm * 2.2) / res_mm))
    k = max(9, k | 1)
    board = cv2.morphologyEx(occ, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8), iterations=1)
    board = cv2.morphologyEx(board, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)
    num, labels, stats, _ = cv2.connectedComponentsWithStats((board > 0).astype(np.uint8), 8)
    if num <= 1:
        return board
    areas = stats[1:, cv2.CC_STAT_AREA]
    best = 1 + int(np.argmax(areas))
    board = ((labels == best) * 255).astype(np.uint8)
    board = ndi.binary_fill_holes(board > 0).astype(np.uint8) * 255
    return board

def detect_holes(points_xyz: np.ndarray, min_diam_mm: float, max_diam_mm: float, res_mm: float):
    c, u, v, n, mode_d, band, dist = robust_plane_fit(points_xyz)
    use = np.abs(dist - mode_d) <= band
    surface_pts = points_xyz[use]
    X = surface_pts - c
    uv = np.c_[X @ u, X @ v]

    occ, uv_min = rasterize_surface(uv, res_mm=res_mm, pad_mm=1.0)
    board = build_board_mask(occ, max_hole_diam_mm=max_diam_mm, res_mm=res_mm)
    holes = ((board > 0) & (occ == 0)).astype(np.uint8) * 255
    holes = cv2.morphologyEx(holes, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

    num, labels, stats, _ = cv2.connectedComponentsWithStats((holes > 0).astype(np.uint8), 8)
    found = []
    debug_lines = []
    min_area = math.pi * ((min_diam_mm * 0.28) / res_mm) ** 2
    max_area = math.pi * ((max_diam_mm * 0.75) / res_mm) ** 2

    for i in range(1, num):
        x, y, w, h, area = stats[i]
        mask = (labels == i).astype(np.uint8)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)
        cnt_area = max(float(cv2.contourArea(cnt)), 1.0)
        peri = max(float(cv2.arcLength(cnt, True)), 1.0)
        circularity = 4.0 * math.pi * cnt_area / (peri * peri + 1e-12)

        contour_xy = cnt[:, 0, :].astype(np.float64)
        cx_px, cy_px, r_px = fit_circle_kasa(contour_xy)
        diam_mm = 2.0 * r_px * res_mm

        keep = (
            area >= min_area and
            area <= max_area and
            min_diam_mm * 0.45 <= diam_mm <= max_diam_mm * 1.10 and
            circularity >= 0.45 and
            min(w, h) >= max(4, int(round((min_diam_mm * 0.30) / res_mm)))
        )

        debug_lines.append(
            f"[DEBUG] comp={i:02d} area_px={area:4d} bbox=({x},{y},{w},{h}) "
            f"circ={circularity:.3f} diam_mm={diam_mm:.3f} keep={keep}"
        )

        if not keep:
            continue

        uv_center = uv_min + np.array([cx_px * res_mm, cy_px * res_mm])
        xyz = c + u * uv_center[0] + v * uv_center[1] + n * mode_d

        found.append(Hole(
            id=len(found),
            center_uv_mm=[float(uv_center[0]), float(uv_center[1])],
            center_xyz_mm=[float(xyz[0]), float(xyz[1]), float(xyz[2])],
            diameter_mm=float(diam_mm),
            circularity=float(circularity),
            area_px=int(area),
            bbox_xywh=[int(x), int(y), int(w), int(h)],
        ))

    found.sort(key=lambda h: (round(h.center_uv_mm[1], 3), round(h.center_uv_mm[0], 3)))
    for k, h in enumerate(found):
        h.id = k

    meta = {
        "plane_center_xyz": [float(x) for x in c],
        "plane_normal": [float(x) for x in n],
        "axis_u": [float(x) for x in u],
        "axis_v": [float(x) for x in v],
        "plane_mode_offset_mm": float(mode_d),
        "surface_band_mm": float(band),
        "surface_points_used": int(len(surface_pts)),
        "grid_resolution_mm": float(res_mm),
    }
    return found, occ, board, holes, meta, debug_lines

def save_images(base_name: str, out_dir: str, occ: np.ndarray, board: np.ndarray, holes: np.ndarray, found: list[Hole], res_mm: float):
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, f"{base_name}_01_occupancy.png"), occ)
    cv2.imwrite(os.path.join(out_dir, f"{base_name}_02_board_mask.png"), board)
    cv2.imwrite(os.path.join(out_dir, f"{base_name}_03_holes_mask.png"), holes)

    occ_rgb = cv2.cvtColor(occ, cv2.COLOR_GRAY2BGR)
    for h in found:
        x = int(round((h.center_uv_mm[0] - 0.0) / res_mm))
        y = int(round((h.center_uv_mm[1] - 0.0) / res_mm))
    # draw using bbox-derived coordinates instead, because uv origin is not 0 in global image space

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="Input .xyz files")
    ap.add_argument("--out-dir", default="hole_detection_output")
    ap.add_argument("--min-diam-mm", type=float, default=1.0)
    ap.add_argument("--max-diam-mm", type=float, default=2.0)
    ap.add_argument("--grid-mm", type=float, default=0.04)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    summary = {}
    for path in args.inputs:
        pts = np.loadtxt(path)[:, :3]
        base = os.path.splitext(os.path.basename(path))[0]
        out_dir = os.path.join(args.out_dir, base)
        os.makedirs(out_dir, exist_ok=True)

        found, occ, board, holes, meta, debug_lines = detect_holes(
            pts, min_diam_mm=args.min_diam_mm, max_diam_mm=args.max_diam_mm, res_mm=args.grid_mm
        )

        # overlay
        overlay = cv2.cvtColor(occ, cv2.COLOR_GRAY2BGR)
        for h in found:
            x, y, w, hh = h.bbox_xywh
            cx = x + w // 2
            cy = y + hh // 2
            r = max(3, int(round((h.diameter_mm / 2.0) / args.grid_mm)))
            cv2.circle(overlay, (cx, cy), r, (0, 0, 255), 2)
            cv2.drawMarker(overlay, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 12, 1)
            cv2.putText(overlay, str(h.id), (cx + 5, cy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imwrite(os.path.join(out_dir, f"{base}_01_occupancy.png"), occ)
        cv2.imwrite(os.path.join(out_dir, f"{base}_02_board_mask.png"), board)
        cv2.imwrite(os.path.join(out_dir, f"{base}_03_holes_mask.png"), holes)
        cv2.imwrite(os.path.join(out_dir, f"{base}_04_detected_overlay.png"), overlay)

        with open(os.path.join(out_dir, f"{base}_debug.txt"), "w") as f:
            f.write(f"[INFO] file={path}\n")
            f.write(f"[INFO] detected_holes={len(found)}\n")
            f.write(f"[INFO] plane_normal={meta['plane_normal']}\n")
            f.write(f"[INFO] surface_band_mm={meta['surface_band_mm']:.4f}\n")
            f.write(f"[INFO] surface_points_used={meta['surface_points_used']}\n")
            for line in debug_lines:
                f.write(line + "\n")

        with open(os.path.join(out_dir, f"{base}_holes.json"), "w") as f:
            json.dump({
                "file": path,
                "detected_holes": len(found),
                "meta": meta,
                "holes": [asdict(h) for h in found],
            }, f, indent=2)

        with open(os.path.join(out_dir, f"{base}_holes.csv"), "w") as f:
            f.write("id,cx_uv_mm,cy_uv_mm,cx_xyz_mm,cy_xyz_mm,cz_xyz_mm,diameter_mm,circularity,area_px\n")
            for h in found:
                f.write(
                    f"{h.id},{h.center_uv_mm[0]:.6f},{h.center_uv_mm[1]:.6f},"
                    f"{h.center_xyz_mm[0]:.6f},{h.center_xyz_mm[1]:.6f},{h.center_xyz_mm[2]:.6f},"
                    f"{h.diameter_mm:.6f},{h.circularity:.6f},{h.area_px}\n"
                )

        summary[base] = {
            "detected_holes": len(found),
            "output_dir": out_dir,
            "holes_json": os.path.join(out_dir, f"{base}_holes.json"),
            "debug_txt": os.path.join(out_dir, f"{base}_debug.txt"),
        }

    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()