#!/usr/bin/env python3
import argparse
import json
import math
import time
from dataclasses import dataclass, asdict

import numpy as np

# matplotlib is optional unless --viz-out is used
try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False


@dataclass
class HoleResult:
    center_xyz_mm: list
    normal_xyz: list
    diameter_mm: float
    radius_mm: float
    fit_rmse_mm: float
    fit_mae_mm: float
    n_boundary_points: int


def load_xyz(path: str) -> np.ndarray:
    """
    Load xyz or xyz+extra columns text file.
    Uses first 3 columns as x y z in mm.
    """
    pts = np.loadtxt(path, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts[None, :]
    if pts.shape[1] < 3:
        raise ValueError("Input file must contain at least 3 columns: x y z")
    return pts[:, :3]


def fit_plane_pca(points: np.ndarray):
    """
    Fit plane by PCA.
    Returns:
      centroid: (3,)
      normal: (3,) unit
      e1, e2: orthonormal in-plane basis
    """
    c = np.mean(points, axis=0)
    X = points - c
    cov = (X.T @ X) / max(len(points), 1)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)
    normal = evecs[:, order[0]]
    e1 = evecs[:, order[2]]
    e2 = np.cross(normal, e1)
    e2 /= np.linalg.norm(e2)

    # Make right-handed and stable
    e1 /= np.linalg.norm(e1)
    normal /= np.linalg.norm(normal)
    if np.dot(np.cross(e1, e2), normal) < 0:
        e2 = -e2
    return c, normal, e1, e2


def project_to_plane(points: np.ndarray, origin: np.ndarray, e1: np.ndarray, e2: np.ndarray):
    """
    Project 3D points to 2D plane coordinates.
    """
    X = points - origin
    u = X @ e1
    v = X @ e2
    return np.column_stack([u, v])


def backproject_to_3d(uv: np.ndarray, origin: np.ndarray, e1: np.ndarray, e2: np.ndarray):
    return origin + uv[0] * e1 + uv[1] * e2


def build_occupancy(uv: np.ndarray, grid_mm: float):
    """
    Create occupancy grid from projected points.
    """
    uv_min = uv.min(axis=0) - grid_mm * 2.0
    uv_max = uv.max(axis=0) + grid_mm * 2.0
    dims = np.ceil((uv_max - uv_min) / grid_mm).astype(int) + 1

    occ = np.zeros((dims[1], dims[0]), dtype=np.uint8)  # row=v, col=u
    ij = np.floor((uv - uv_min) / grid_mm).astype(int)
    ij[:, 0] = np.clip(ij[:, 0], 0, dims[0] - 1)
    ij[:, 1] = np.clip(ij[:, 1], 0, dims[1] - 1)
    occ[ij[:, 1], ij[:, 0]] = 1
    return occ, uv_min, dims


def binary_dilate(img: np.ndarray, iterations: int = 1):
    out = img.astype(bool).copy()
    for _ in range(iterations):
        padded = np.pad(out, ((1, 1), (1, 1)), mode="constant", constant_values=False)
        new = np.zeros_like(out, dtype=bool)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                new |= padded[1 + dy:1 + dy + out.shape[0], 1 + dx:1 + dx + out.shape[1]]
        out = new
    return out.astype(np.uint8)


def binary_erode(img: np.ndarray, iterations: int = 1):
    out = img.astype(bool).copy()
    for _ in range(iterations):
        padded = np.pad(out, ((1, 1), (1, 1)), mode="constant", constant_values=True)
        new = np.ones_like(out, dtype=bool)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                new &= padded[1 + dy:1 + dy + out.shape[0], 1 + dx:1 + dx + out.shape[1]]
        out = new
    return out.astype(np.uint8)


def binary_close(img: np.ndarray, iterations: int = 1):
    return binary_erode(binary_dilate(img, iterations), iterations)


def connected_components(bin_img: np.ndarray):
    """
    Simple 8-connected component labeling.
    Returns list of dicts with pixel coords.
    """
    h, w = bin_img.shape
    visited = np.zeros_like(bin_img, dtype=bool)
    comps = []

    for y in range(h):
        for x in range(w):
            if not bin_img[y, x] or visited[y, x]:
                continue

            stack = [(x, y)]
            visited[y, x] = True
            coords = []

            while stack:
                cx, cy = stack.pop()
                coords.append((cx, cy))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = cx + dx, cy + dy
                        if 0 <= nx < w and 0 <= ny < h and bin_img[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((nx, ny))

            comps.append({"pixels": np.array(coords, dtype=int)})
    return comps


def estimate_hole_candidates_from_empty_regions(
    occ: np.ndarray,
    grid_mm: float,
    expected_diameter_mm: float,
    uv_min: np.ndarray,
):
    """
    Detect holes as empty connected regions inside occupied board area.
    """
    occ2 = binary_close(occ, iterations=1)

    # board mask = occupied area slightly expanded
    board_mask = binary_dilate(occ2, iterations=max(1, int(round((expected_diameter_mm * 0.6) / grid_mm))))
    empty = (board_mask > 0) & (occ2 == 0)

    comps = connected_components(empty.astype(np.uint8))
    candidates = []

    expected_area = math.pi * (expected_diameter_mm * 0.5) ** 2
    pix_area = grid_mm * grid_mm

    for comp in comps:
        px = comp["pixels"]
        if len(px) < 8:
            continue

        area_mm2 = len(px) * pix_area
        if not (0.25 * expected_area <= area_mm2 <= 3.0 * expected_area):
            continue

        xs = px[:, 0]
        ys = px[:, 1]
        w_mm = (xs.max() - xs.min() + 1) * grid_mm
        h_mm = (ys.max() - ys.min() + 1) * grid_mm

        # roughly circular size gate
        if not (0.4 * expected_diameter_mm <= w_mm <= 2.0 * expected_diameter_mm):
            continue
        if not (0.4 * expected_diameter_mm <= h_mm <= 2.0 * expected_diameter_mm):
            continue

        center_px = np.array([xs.mean(), ys.mean()], dtype=np.float64)
        center_uv = uv_min + (center_px + 0.5) * grid_mm
        candidates.append({
            "center_uv": center_uv,
            "area_mm2": area_mm2,
            "bbox_mm": [w_mm, h_mm],
            "pixels": px,
        })

    return candidates


def circle_fit_kasa(points_2d: np.ndarray):
    """
    Algebraic circle fit (Kasa).
    x^2 + y^2 + Ax + By + C = 0
    """
    x = points_2d[:, 0]
    y = points_2d[:, 1]
    A = np.column_stack([x, y, np.ones_like(x)])
    b = -(x * x + y * y)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    A_, B_, C_ = sol
    cx = -A_ / 2.0
    cy = -B_ / 2.0
    r_sq = cx * cx + cy * cy - C_
    r_sq = max(r_sq, 0.0)
    r = math.sqrt(r_sq)
    return np.array([cx, cy]), r


def refine_circle_geometric(points_2d: np.ndarray, center_init: np.ndarray, r_init: float, iters: int = 10):
    """
    Lightweight Gauss-Newton refinement on geometric distance residuals.
    residual_i = ||p_i - c|| - r
    """
    c = center_init.astype(np.float64).copy()
    r = float(r_init)

    for _ in range(iters):
        d = points_2d - c[None, :]
        dist = np.linalg.norm(d, axis=1) + 1e-12
        resid = dist - r

        J = np.zeros((len(points_2d), 3), dtype=np.float64)
        J[:, 0] = -d[:, 0] / dist
        J[:, 1] = -d[:, 1] / dist
        J[:, 2] = -1.0

        H = J.T @ J
        g = J.T @ resid
        try:
            delta = -np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break

        c[0] += delta[0]
        c[1] += delta[1]
        r += delta[2]

        if np.linalg.norm(delta) < 1e-10:
            break

    d = np.linalg.norm(points_2d - c[None, :], axis=1)
    resid = d - r
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    mae = float(np.mean(np.abs(resid)))
    return c, r, rmse, mae


def extract_boundary_points_for_candidate(
    uv: np.ndarray,
    center_uv: np.ndarray,
    expected_radius_mm: float,
    band_half_width_mm: float = 0.20,
):
    """
    Use annulus around candidate center to gather boundary points.
    """
    d = np.linalg.norm(uv - center_uv[None, :], axis=1)
    mask = (d >= (expected_radius_mm - band_half_width_mm)) & (d <= (expected_radius_mm + band_half_width_mm))
    pts = uv[mask]

    # If too sparse, widen band
    if len(pts) < 30:
        mask = (d >= (expected_radius_mm - 0.35)) & (d <= (expected_radius_mm + 0.35))
        pts = uv[mask]
    return pts


def deduplicate_centers(candidates, min_dist_mm=0.4):
    out = []
    for c in candidates:
        keep = True
        for o in out:
            if np.linalg.norm(c["center_uv"] - o["center_uv"]) < min_dist_mm:
                # keep the one with area closer to expected / currently first is fine
                keep = False
                break
        if keep:
            out.append(c)
    return out


def detect_holes(
    points_xyz: np.ndarray,
    expected_diameter_mm: float = 1.0,
    grid_mm: float = 0.03,
):
    """
    Main detection pipeline.
    """
    t0 = time.perf_counter()

    # 1. Fit board plane
    origin, normal, e1, e2 = fit_plane_pca(points_xyz)

    # 2. Project to 2D
    uv = project_to_plane(points_xyz, origin, e1, e2)

    # 3. Build occupancy and detect empty circular regions
    occ, uv_min, dims = build_occupancy(uv, grid_mm)
    candidates = estimate_hole_candidates_from_empty_regions(
        occ=occ,
        grid_mm=grid_mm,
        expected_diameter_mm=expected_diameter_mm,
        uv_min=uv_min,
    )
    candidates = deduplicate_centers(candidates, min_dist_mm=0.4)

    # 4. Refine each candidate with circle fit on nearby boundary points
    results = []
    expected_r = expected_diameter_mm * 0.5

    for cand in candidates:
        center0 = cand["center_uv"]
        boundary_pts = extract_boundary_points_for_candidate(
            uv, center0, expected_radius_mm=expected_r, band_half_width_mm=0.20
        )
        if len(boundary_pts) < 20:
            continue

        c0, r0 = circle_fit_kasa(boundary_pts)

        # robust trimming by residual
        dist0 = np.linalg.norm(boundary_pts - c0[None, :], axis=1)
        resid0 = np.abs(dist0 - r0)
        thr = max(0.04, np.percentile(resid0, 75) * 1.5)
        inliers = boundary_pts[resid0 <= thr]

        if len(inliers) < 16:
            continue

        c1, r1, rmse, mae = refine_circle_geometric(inliers, c0, r0, iters=15)

        # filter by radius consistency
        if not (0.3 <= r1 <= 1.5):
            continue

        center_xyz = backproject_to_3d(c1, origin, e1, e2)

        results.append(HoleResult(
            center_xyz_mm=center_xyz.tolist(),
            normal_xyz=normal.tolist(),
            diameter_mm=float(2.0 * r1),
            radius_mm=float(r1),
            fit_rmse_mm=float(rmse),
            fit_mae_mm=float(mae),
            n_boundary_points=int(len(inliers)),
        ))

    # 5. Final dedup in 3D
    final = []
    for r in sorted(results, key=lambda x: x.fit_rmse_mm):
        p = np.array(r.center_xyz_mm)
        keep = True
        for rr in final:
            q = np.array(rr.center_xyz_mm)
            if np.linalg.norm(p - q) < 0.3:
                keep = False
                break
        if keep:
            final.append(r)

    elapsed = time.perf_counter() - t0
    return {
        "elapsed_sec": elapsed,
        "n_input_points": int(len(points_xyz)),
        "plane_origin_xyz_mm": origin.tolist(),
        "plane_normal_xyz": normal.tolist(),
        "holes": [asdict(x) for x in final],
        "debug": {
            "n_candidates_initial": len(candidates),
            "grid_mm": grid_mm,
            "expected_diameter_mm": expected_diameter_mm,
            "occupancy_shape": [int(dims[1]), int(dims[0])],
        }
    }, uv, origin, normal, e1, e2


def save_json(path: str, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def make_viz(
    path: str,
    points_xyz: np.ndarray,
    result: dict,
    uv: np.ndarray,
    origin: np.ndarray,
    normal: np.ndarray,
    e1: np.ndarray,
    e2: np.ndarray,
):
    if not HAS_MPL:
        print("matplotlib not available; skipping visualization")
        return

    holes = result["holes"]

    fig = plt.figure(figsize=(12, 5))

    # 2D plane view
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.scatter(uv[:, 0], uv[:, 1], s=0.2)
    for i, h in enumerate(holes):
        c3 = np.array(h["center_xyz_mm"])
        rel = c3 - origin
        cu = np.dot(rel, e1)
        cv = np.dot(rel, e2)
        r = h["radius_mm"]
        t = np.linspace(0, 2 * np.pi, 200)
        ax1.plot(cu + r * np.cos(t), cv + r * np.sin(t))
        ax1.scatter([cu], [cv], s=40)
        ax1.text(cu, cv, f"H{i}", fontsize=9)
    ax1.set_title("Projected plane view")
    ax1.set_aspect("equal", adjustable="box")
    ax1.set_xlabel("u [mm]")
    ax1.set_ylabel("v [mm]")

    # 3D view
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    stride = max(1, len(points_xyz) // 30000)
    pts_show = points_xyz[::stride]
    ax2.scatter(pts_show[:, 0], pts_show[:, 1], pts_show[:, 2], s=0.2)

    for i, h in enumerate(holes):
        c = np.array(h["center_xyz_mm"])
        n = np.array(h["normal_xyz"])
        ax2.scatter([c[0]], [c[1]], [c[2]], s=50)
        ax2.text(c[0], c[1], c[2], f"H{i}")
        scale = 1.0
        ax2.plot(
            [c[0], c[0] + n[0] * scale],
            [c[1], c[1] + n[1] * scale],
            [c[2], c[2] + n[2] * scale],
        )

    ax2.set_title("3D view")
    ax2.set_xlabel("X [mm]")
    ax2.set_ylabel("Y [mm]")
    ax2.set_zlabel("Z [mm]")

    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Fast and accurate PCB hole detection from point cloud")
    parser.add_argument("xyz_path", help="Input .xyz text file (first 3 cols = x y z in mm)")
    parser.add_argument("--expected-diameter", type=float, default=1.0, help="Expected hole diameter in mm")
    parser.add_argument("--grid-mm", type=float, default=0.03, help="2D occupancy grid size in mm")
    parser.add_argument("--json-out", type=str, default="", help="Optional JSON output path")
    parser.add_argument("--viz-out", type=str, default="", help="Optional visualization PNG output path")
    args = parser.parse_args()

    points_xyz = load_xyz(args.xyz_path)

    result, uv, origin, normal, e1, e2 = detect_holes(
        points_xyz,
        expected_diameter_mm=args.expected_diameter,
        grid_mm=args.grid_mm,
    )

    print(json.dumps(result, indent=2))

    if args.json_out:
        save_json(args.json_out, result)
        print(f"\nSaved JSON: {args.json_out}")

    if args.viz_out:
        make_viz(args.viz_out, points_xyz, result, uv, origin, normal, e1, e2)
        print(f"Saved visualization: {args.viz_out}")


if __name__ == "__main__":
    main()