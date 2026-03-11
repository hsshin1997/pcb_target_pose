#!/usr/bin/env python3
from __future__ import annotations

import math
import itertools
import argparse
from dataclasses import dataclass
from scipy.optimize import least_squares

import cv2
import numpy as np
import open3d as o3d
from scipy.ndimage import gaussian_filter
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt



@dataclass
class PlaneFrameResult:
    plane_normal_3d: np.ndarray
    axis1_3d: np.ndarray
    axis2_3d: np.ndarray
    origin_3d: np.ndarray
    hole_centers_2d_all: np.ndarray
    hole_centers_2d_used: np.ndarray
    hole_centers_3d_used: np.ndarray
    plane_basis_u: np.ndarray
    plane_basis_v: np.ndarray
    plane_point: np.ndarray
    projected_2d: np.ndarray
    fitted_rectangle_2d: np.ndarray
    ordered_rectangle_points_2d: np.ndarray
    rectangle_half_lengths_2d: np.ndarray
    rectangle_edge_lengths_2d: np.ndarray


def normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        return v.copy()
    return v / n


def ensure_right_handed(x: np.ndarray, y: np.ndarray, z: np.ndarray):
    x = normalize(x)
    y = normalize(y)
    z = normalize(z)
    if np.dot(np.cross(x, y), z) < 0:
        y = -y
    return x, y, z


def load_xyz_points(path: str) -> np.ndarray:
    pts = np.loadtxt(path, usecols=(0, 1, 2), dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"Failed to load xyz from {path}")
    return pts

def compute_edge_lengths(points_2d: np.ndarray) -> np.ndarray:
    pts = order_points_cyclic(points_2d)
    lengths = []
    for i in range(4):
        p0 = pts[i]
        p1 = pts[(i + 1) % 4]
        lengths.append(np.linalg.norm(p1 - p0))
    return np.array(lengths, dtype=np.float64)

def crop_points_center_size(
    points: np.ndarray,
    center_x: float | None = None,
    center_y: float | None = None,
    center_z: float | None = None,
    size_x: float | None = None,
    size_y: float | None = None,
    size_z: float | None = None,
) -> np.ndarray:
    """
    Axis-aligned crop using box center and box size.

    For each axis:
        [center - size/2, center + size/2]

    If either center or size is None for an axis, that axis is not cropped.
    """
    if len(points) == 0:
        return points

    mask = np.ones(len(points), dtype=bool)

    if center_x is not None and size_x is not None:
        x_min = center_x - 0.5 * size_x
        x_max = center_x + 0.5 * size_x
        mask &= (points[:, 0] >= x_min) & (points[:, 0] <= x_max)

    if center_y is not None and size_y is not None:
        y_min = center_y - 0.5 * size_y
        y_max = center_y + 0.5 * size_y
        mask &= (points[:, 1] >= y_min) & (points[:, 1] <= y_max)

    if center_z is not None and size_z is not None:
        z_min = center_z - 0.5 * size_z
        z_max = center_z + 0.5 * size_z
        mask &= (points[:, 2] >= z_min) & (points[:, 2] <= z_max)

    return points[mask]


def fit_circle_kasa(points_2d: np.ndarray):
    """
    Algebraic circle fit:
        x^2 + y^2 + A x + B y + C = 0
    center = (-A/2, -B/2), radius = sqrt((A^2 + B^2)/4 - C)
    """
    if len(points_2d) < 3:
        raise RuntimeError("Need at least 3 points for circle fit.")

    x = points_2d[:, 0]
    y = points_2d[:, 1]

    M = np.column_stack([x, y, np.ones_like(x)])
    b = -(x * x + y * y)

    sol, *_ = np.linalg.lstsq(M, b, rcond=None)
    A, B, C = sol

    cx = -0.5 * A
    cy = -0.5 * B
    r_sq = 0.25 * (A * A + B * B) - C
    r_sq = max(r_sq, 0.0)
    r = math.sqrt(r_sq)

    return np.array([cx, cy], dtype=np.float64), float(r)


def sample_component_boundary_points(comp_mask: np.ndarray, meta: dict) -> np.ndarray:
    """
    Extract contour points from a binary connected-component mask
    and convert contour pixels to plane coordinates.
    """
    contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.zeros((0, 2), dtype=np.float64)

    cnt = max(contours, key=cv2.contourArea)
    pts = []
    for p in cnt[:, 0, :]:
        px, py = float(p[0]), float(p[1])
        pts.append(img_to_plane_coords(px, py, meta))
    return np.array(pts, dtype=np.float64)


def fit_plane_ransac_open3d(points: np.ndarray,
                            distance_threshold: float = 0.2,
                            ransac_n: int = 3,
                            num_iterations: int = 3000):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations
    )

    a, b, c, d = plane_model
    normal = normalize(np.array([a, b, c], dtype=np.float64))

    inlier_points = points[np.asarray(inliers)]
    centroid = np.mean(inlier_points, axis=0)

    X = inlier_points - centroid
    _, _, vh = np.linalg.svd(X, full_matrices=False)
    refined_normal = normalize(vh[-1])

    if np.dot(refined_normal, normal) < 0:
        refined_normal = -refined_normal

    return refined_normal, centroid, inlier_points, np.asarray(inliers, dtype=int)


def build_plane_basis(normal: np.ndarray,
                      preferred_ref: np.ndarray | None = None):
    n = normalize(normal)

    if preferred_ref is None:
        preferred_ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    if abs(np.dot(preferred_ref, n)) > 0.9:
        preferred_ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    u = preferred_ref - np.dot(preferred_ref, n) * n
    u = normalize(u)
    v = normalize(np.cross(n, u))
    u, v, n = ensure_right_handed(u, v, n)
    return u, v


def project_points_to_plane(points: np.ndarray,
                            origin_3d: np.ndarray,
                            u: np.ndarray,
                            v: np.ndarray) -> np.ndarray:
    rel = points - origin_3d[None, :]
    x = rel @ u
    y = rel @ v
    return np.column_stack([x, y])


def unproject_2d_to_3d(points_2d: np.ndarray,
                       origin_3d: np.ndarray,
                       u: np.ndarray,
                       v: np.ndarray) -> np.ndarray:
    return origin_3d[None, :] + points_2d[:, [0]] * u[None, :] + points_2d[:, [1]] * v[None, :]


def rasterize_points_2d(points_2d: np.ndarray, pixel_size: float):
    min_xy = np.min(points_2d, axis=0)
    max_xy = np.max(points_2d, axis=0)

    pad = 5 * pixel_size
    min_xy = min_xy - pad
    max_xy = max_xy + pad

    width = int(np.ceil((max_xy[0] - min_xy[0]) / pixel_size)) + 1
    height = int(np.ceil((max_xy[1] - min_xy[1]) / pixel_size)) + 1

    ij = np.floor((points_2d - min_xy[None, :]) / pixel_size).astype(np.int32)
    ij[:, 0] = np.clip(ij[:, 0], 0, width - 1)
    ij[:, 1] = np.clip(ij[:, 1], 0, height - 1)

    count_img = np.zeros((height, width), dtype=np.float32)
    for x_idx, y_idx in ij:
        count_img[height - 1 - y_idx, x_idx] += 1.0

    occupancy = (count_img > 0).astype(np.uint8)

    meta = {
        "min_xy": min_xy,
        "pixel_size": pixel_size,
        "height": height,
        "width": width,
    }
    return count_img, occupancy, meta


def img_to_plane_coords(px: float, py: float, meta: dict) -> np.ndarray:
    min_xy = meta["min_xy"]
    pixel_size = meta["pixel_size"]
    height = meta["height"]

    x = min_xy[0] + (px + 0.5) * pixel_size
    y_idx = (height - 1 - py)
    y = min_xy[1] + (y_idx + 0.5) * pixel_size
    return np.array([x, y], dtype=np.float64)


def detect_holes_small_diameter_refined(
    count_img: np.ndarray,
    occupancy: np.ndarray,
    meta: dict,
    expected_hole_diameter_mm: float = 1.0,
    support_blur_sigma_px: float = 2.0,
    support_thresh_ratio: float = 0.05,
    circularity_min: float = 0.30,
    debug: bool = False,
):
    """
    Detect hole candidates from projected occupancy and refine center by circle fitting
    on component boundary points.
    """
    px = meta["pixel_size"]
    exp_diam_px = max(2.0, expected_hole_diameter_mm / px)
    exp_rad_px = 0.5 * exp_diam_px

    dense = gaussian_filter(count_img, sigma=support_blur_sigma_px)
    support_mask = (dense > support_thresh_ratio * max(dense.max(), 1e-12)).astype(np.uint8)

    occ_u8 = (occupancy * 255).astype(np.uint8)
    k_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    occ_closed = cv2.morphologyEx(occ_u8, cv2.MORPH_CLOSE, k_small, iterations=1)
    occ_closed = (occ_closed > 0).astype(np.uint8)

    holes_mask = ((support_mask == 1) & (occ_closed == 0)).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(holes_mask, connectivity=8)

    centers_2d = []
    radii_2d = []
    kept_contours = []
    component_masks = []

    expected_area = math.pi * (exp_rad_px ** 2)
    min_area = max(4.0, 0.20 * expected_area)
    max_area = 3.50 * expected_area

    H, W = holes_mask.shape

    if debug:
        print(f"[DEBUG] expected hole diameter px: {exp_diam_px:.3f}")
        print(f"[DEBUG] expected area px: {expected_area:.3f}")
        print(f"[DEBUG] accepted area range px: [{min_area:.3f}, {max_area:.3f}]")

    for lab in range(1, num_labels):
        x = stats[lab, cv2.CC_STAT_LEFT]
        y = stats[lab, cv2.CC_STAT_TOP]
        w = stats[lab, cv2.CC_STAT_WIDTH]
        h = stats[lab, cv2.CC_STAT_HEIGHT]
        area = float(stats[lab, cv2.CC_STAT_AREA])

        if x <= 0 or y <= 0 or x + w >= W - 1 or y + h >= H - 1:
            continue

        if not (min_area <= area <= max_area):
            continue

        comp = (labels == lab).astype(np.uint8) * 255
        contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        cnt = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter < 1e-9:
            continue

        contour_area = cv2.contourArea(cnt)
        circularity = 4.0 * math.pi * contour_area / (perimeter * perimeter + 1e-12)
        if circularity < circularity_min:
            continue

        _, _, bw, bh = cv2.boundingRect(cnt)
        aspect = bw / max(bh, 1e-9)
        if aspect < 0.4 or aspect > 2.5:
            continue

        boundary_pts_2d = sample_component_boundary_points(comp, meta)
        if len(boundary_pts_2d) < 6:
            continue

        try:
            center_2d, radius_2d = fit_circle_kasa(boundary_pts_2d)
        except Exception:
            continue

        if radius_2d <= 0:
            continue

        centers_2d.append(center_2d)
        radii_2d.append(radius_2d)
        kept_contours.append(cnt)
        component_masks.append(comp)

    centers_2d = np.array(centers_2d, dtype=np.float64) if centers_2d else np.zeros((0, 2), dtype=np.float64)
    radii_2d = np.array(radii_2d, dtype=np.float64) if radii_2d else np.zeros((0,), dtype=np.float64)

    debug_data = {
        "dense": dense,
        "support_mask": support_mask,
        "occ_closed": occ_closed,
        "holes_mask": holes_mask,
        "kept_contours": kept_contours,
        "component_masks": component_masks,
        "hole_radii_2d": radii_2d,
    }

    if debug:
        print(f"[DEBUG] refined hole candidates: {len(centers_2d)}")
        for i, (c, r) in enumerate(zip(centers_2d, radii_2d)):
            print(f"  hole {i}: center={c}, radius={r:.6f}")

    return centers_2d, debug_data


def estimate_axes_from_hole_centers(centers_2d: np.ndarray,
                                    k_neighbors: int = 3,
                                    angle_bin_deg: float = 2.0,
                                    orth_tol_deg: float = 20.0):
    if centers_2d.shape[0] < 3:
        raise RuntimeError("Need at least 3 hole centers to estimate axes.")

    nbrs = NearestNeighbors(n_neighbors=min(k_neighbors + 1, len(centers_2d)), algorithm="auto")
    nbrs.fit(centers_2d)
    distances, indices = nbrs.kneighbors(centers_2d)

    dirs = []
    weights = []

    for i in range(len(centers_2d)):
        p = centers_2d[i]
        for j_idx, dist in zip(indices[i, 1:], distances[i, 1:]):
            if dist < 1e-9:
                continue
            q = centers_2d[j_idx]
            d = q - p
            dn = d / np.linalg.norm(d)
            angle = math.atan2(dn[1], dn[0])
            if angle < 0:
                angle += math.pi
            dirs.append(angle)
            weights.append(1.0 / (dist + 1e-6))

    if len(dirs) == 0:
        raise RuntimeError("Could not build neighbor directions.")

    dirs = np.array(dirs, dtype=np.float64)
    weights = np.array(weights, dtype=np.float64)

    bin_width = np.deg2rad(angle_bin_deg)
    nbins = max(18, int(np.ceil(math.pi / bin_width)))
    hist, edges = np.histogram(dirs, bins=nbins, range=(0, math.pi), weights=weights)

    peak_idx = int(np.argmax(hist))
    theta1 = 0.5 * (edges[peak_idx] + edges[peak_idx + 1])

    tol = np.deg2rad(orth_tol_deg)
    delta = ((dirs - theta1 + math.pi / 2) % math.pi) - math.pi / 2
    mask = np.abs(delta) < tol
    if np.sum(mask) >= 2:
        theta1 = 0.5 * math.atan2(np.sum(weights[mask] * np.sin(2 * dirs[mask])),
                                  np.sum(weights[mask] * np.cos(2 * dirs[mask])))
        if theta1 < 0:
            theta1 += math.pi

    e1 = np.array([math.cos(theta1), math.sin(theta1)], dtype=np.float64)
    e2 = np.array([-e1[1], e1[0]], dtype=np.float64)
    return normalize(e1), normalize(e2)


def rectangle_subset_score(pts4: np.ndarray):
    """
    Lower is better.
    Score based on:
    - 2-level structure along PCA axes
    - orthogonality of adjacent edges
    - opposite-edge similarity
    """
    c = np.mean(pts4, axis=0)
    X = pts4 - c

    # PCA basis
    cov = X.T @ X
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    e1 = normalize(eigvecs[:, order[0]])
    e2 = np.array([-e1[1], e1[0]])

    proj = np.column_stack([X @ e1, X @ e2])

    xs = np.sort(proj[:, 0])
    ys = np.sort(proj[:, 1])

    # rectangle-like if there are 2 left + 2 right and 2 bottom + 2 top
    x_gap_mid = xs[2] - xs[1]
    y_gap_mid = ys[2] - ys[1]
    x_spread_in = (xs[1] - xs[0]) + (xs[3] - xs[2])
    y_spread_in = (ys[1] - ys[0]) + (ys[3] - ys[2])

    # bigger mid-gap, smaller within-pair spread is better
    level_score = (x_spread_in / (abs(x_gap_mid) + 1e-6)) + (y_spread_in / (abs(y_gap_mid) + 1e-6))

    # sort by angle around centroid to make polygon
    ang = np.arctan2(X[:, 1], X[:, 0])
    order = np.argsort(ang)
    P = pts4[order]

    edges = []
    lens = []
    for i in range(4):
        d = P[(i + 1) % 4] - P[i]
        edges.append(normalize(d))
        lens.append(np.linalg.norm(d))
    edges = np.array(edges)
    lens = np.array(lens)

    # adjacent edges should be orthogonal
    ortho_score = 0.0
    for i in range(4):
        ortho_score += abs(np.dot(edges[i], edges[(i + 1) % 4]))

    # opposite edges should have similar length
    opp_score = abs(lens[0] - lens[2]) / (max(lens[0], lens[2], 1e-6)) + \
                abs(lens[1] - lens[3]) / (max(lens[1], lens[3], 1e-6))

    score = level_score + 2.0 * ortho_score + opp_score
    return score

def rectangle_residuals(params: np.ndarray, points_2d: np.ndarray, sign_pairs: np.ndarray) -> np.ndarray:
    """
    params = [ox, oy, theta, a, b]
    model corner i:
        o + sx_i * a * ex(theta) + sy_i * b * ey(theta)
    """
    ox, oy, theta, a, b = params
    o = np.array([ox, oy], dtype=np.float64)

    ex = np.array([math.cos(theta), math.sin(theta)], dtype=np.float64)
    ey = np.array([-math.sin(theta), math.cos(theta)], dtype=np.float64)

    residuals = []
    for p, (sx, sy) in zip(points_2d, sign_pairs):
        pred = o + sx * a * ex + sy * b * ey
        residuals.extend((p - pred).tolist())
    return np.array(residuals, dtype=np.float64)


def fit_constrained_rectangle(points_2d: np.ndarray, debug: bool = False):
    """
    Fit:
        p_i ≈ o + sx_i * a * ex(theta) + sy_i * b * ey(theta),
    with orthonormal axes defined by theta.

    Returns:
        origin_2d, axis1_2d, axis2_2d, half_lengths, ordered_pts
    """
    if points_2d.shape != (4, 2):
        raise RuntimeError("fit_constrained_rectangle expects exactly 4 points.")

    ordered_pts = order_points_cyclic(points_2d)
    o0 = np.mean(ordered_pts, axis=0)

    # PCA init for theta
    X = ordered_pts - o0
    cov = X.T @ X
    eigvals, eigvecs = np.linalg.eigh(cov)
    e_init = eigvecs[:, np.argmax(eigvals)]
    theta0 = math.atan2(e_init[1], e_init[0])

    # initial side lengths from projections
    ex0 = np.array([math.cos(theta0), math.sin(theta0)], dtype=np.float64)
    ey0 = np.array([-math.sin(theta0), math.cos(theta0)], dtype=np.float64)
    proj_x = X @ ex0
    proj_y = X @ ey0
    a0 = 0.5 * (np.max(proj_x) - np.min(proj_x))
    b0 = 0.5 * (np.max(proj_y) - np.min(proj_y))
    a0 = max(a0, 1e-6)
    b0 = max(b0, 1e-6)

    # Use all 4 sign assignments tied to cyclicly ordered corners
    # canonical rectangle corners around center:
    # (-a,-b), (+a,-b), (+a,+b), (-a,+b)
    sign_pairs = np.array([
        [-1.0, -1.0],
        [+1.0, -1.0],
        [+1.0, +1.0],
        [-1.0, +1.0],
    ], dtype=np.float64)

    # Because cyclic order may start at any corner and orientation may reverse,
    # test all 8 equivalent assignments.
    best_cost = float("inf")
    best = None

    for shift in range(4):
        sp = np.roll(sign_pairs, shift=shift, axis=0)
        for reverse in [False, True]:
            sp_use = sp[::-1].copy() if reverse else sp.copy()

            x0 = np.array([o0[0], o0[1], theta0, a0, b0], dtype=np.float64)
            res = least_squares(
                rectangle_residuals,
                x0=x0,
                args=(ordered_pts, sp_use),
                method="lm",
                max_nfev=200
            )

            cost = float(np.sum(res.fun ** 2))
            if cost < best_cost:
                best_cost = cost
                best = (res.x, sp_use)

    if best is None:
        raise RuntimeError("Rectangle fit failed.")

    params_opt, sign_pairs_opt = best
    ox, oy, theta, a, b = params_opt
    origin_2d = np.array([ox, oy], dtype=np.float64)

    axis1_2d = np.array([math.cos(theta), math.sin(theta)], dtype=np.float64)
    axis1_2d = normalize(axis1_2d)
    axis2_2d = np.array([-axis1_2d[1], axis1_2d[0]], dtype=np.float64)

    # Make axis1 correspond to longer rectangle side
    if b > a:
        axis1_2d, axis2_2d = axis2_2d, -axis1_2d
        a, b = b, a

    # Rebuild fitted rectangle corners in cyclic order for plotting/inspection
    fitted_corners = []
    for sx, sy in sign_pairs_opt:
        pred = origin_2d + sx * a * axis1_2d + sy * b * axis2_2d
        fitted_corners.append(pred)
    fitted_corners = np.array(fitted_corners, dtype=np.float64)
    fitted_corners = order_points_cyclic(fitted_corners)

    if debug:
        print(f"[DEBUG] constrained rectangle fit cost: {best_cost:.12f}")
        print(f"[DEBUG] origin_2d: {origin_2d}")
        print(f"[DEBUG] axis1_2d: {axis1_2d}")
        print(f"[DEBUG] axis2_2d: {axis2_2d}")
        print(f"[DEBUG] half lengths: a={a:.9f}, b={b:.9f}")

    return origin_2d, axis1_2d, axis2_2d, np.array([a, b], dtype=np.float64), fitted_corners, ordered_pts

def rectangle_subset_score_from_fit(pts4: np.ndarray, debug: bool = False) -> float:
    """
    Lower is better.
    Score each 4-hole subset by actually fitting the constrained rectangle model.
    Return inf for obviously bad quadrilaterals.
    """
    if pts4.shape != (4, 2):
        return float("inf")

    try:
        origin_2d, axis1_2d, axis2_2d, half_lengths_2d, fitted_rect, ordered_pts = \
            fit_constrained_rectangle(pts4, debug=False)
    except Exception:
        return float("inf")

    ordered_pts = order_points_cyclic(ordered_pts)
    fitted_rect = order_points_cyclic(fitted_rect)

    # 1) fit residual: measured corners vs fitted rectangle corners
    fit_err = float(np.sum(np.linalg.norm(ordered_pts - fitted_rect, axis=1) ** 2))

    # 2) edge lengths
    L = compute_edge_lengths(ordered_pts)
    L0, L1, L2, L3 = L

    opp_err = (
        abs(L0 - L2) / max(L0, L2, 1e-9)
        + abs(L1 - L3) / max(L1, L3, 1e-9)
    )

    # 3) diagonals should match for a rectangle
    D0 = np.linalg.norm(ordered_pts[2] - ordered_pts[0])
    D1 = np.linalg.norm(ordered_pts[3] - ordered_pts[1])
    diag_err = abs(D0 - D1) / max(D0, D1, 1e-9)

    # 4) adjacent edges should be orthogonal
    unit_edges = []
    for i in range(4):
        d = ordered_pts[(i + 1) % 4] - ordered_pts[i]
        unit_edges.append(d / max(np.linalg.norm(d), 1e-9))
    unit_edges = np.array(unit_edges)

    ortho_err = 0.0
    for i in range(4):
        ortho_err += abs(np.dot(unit_edges[i], unit_edges[(i + 1) % 4]))

    # 5) side balance: for your good case, all four edges are nearly equal.
    # This strongly rejects the bad case like 6, 6, 12, 17.
    side_balance_err = np.std(L) / max(np.mean(L), 1e-9)

    # ---- hard reject gates ----
    if opp_err > 0.20:
        return float("inf")
    if diag_err > 0.15:
        return float("inf")
    if ortho_err > 0.60:
        return float("inf")
    if side_balance_err > 0.20:
        return float("inf")

    # ---- final score ----
    score = (
        20.0 * fit_err
        + 5.0 * opp_err
        + 5.0 * diag_err
        + 2.0 * ortho_err
        + 8.0 * side_balance_err
    )

    if debug:
        print(
            f"[DEBUG] fit={fit_err:.6e}, "
            f"opp={opp_err:.6f}, diag={diag_err:.6f}, "
            f"ortho={ortho_err:.6f}, side_balance={side_balance_err:.6f}, "
            f"total={score:.6f}"
        )

    return float(score)



def select_best_rectangular_4hole_subset(centers_2d: np.ndarray, debug: bool = False):
    if len(centers_2d) < 4:
        raise RuntimeError("Need at least 4 detected holes to select rectangle subset.")
    if len(centers_2d) == 4:
        return centers_2d

    best_score = float("inf")
    best_subset = None

    for idxs in itertools.combinations(range(len(centers_2d)), 4):
        subset = centers_2d[list(idxs)]
        score = rectangle_subset_score_from_fit(subset, debug=False)

        if debug:
            ordered = order_points_cyclic(subset)
            L = compute_edge_lengths(ordered)
            D0 = np.linalg.norm(ordered[2] - ordered[0])
            D1 = np.linalg.norm(ordered[3] - ordered[1])
            print(
                f"[DEBUG] idxs={idxs}, "
                f"edges={[float(x) for x in L]}, "
                f"diag=({D0:.6f}, {D1:.6f}), "
                f"score={score}"
            )

        if score < best_score:
            best_score = score
            best_subset = subset

    if best_subset is None or not np.isfinite(best_score):
        raise RuntimeError("No valid rectangle-like 4-hole subset found.")

    if debug:
        print(f"[DEBUG] best 4-hole fit-based score: {best_score:.12f}")

    return best_subset


def compute_plane_axes_from_holes(
    xyz_path: str,
    plane_dist_thresh: float = 0.20,
    pixel_size: float = 0.10,
    expected_hole_diameter_mm: float = 1.0,
    support_blur_sigma_px: float = 2.0,
    support_thresh_ratio: float = 0.05,
    circularity_min: float = 0.30,
    crop: list[float] | None = None,
    debug: bool = False,
):
    points = load_xyz_points(xyz_path)
    if debug:
        print(f"[INFO] loaded {len(points)} points")

    if crop is not None:
        if len(crop) != 6:
            raise RuntimeError("--crop requires exactly 6 numbers: x y z sizex sizey sizez")

        crop_center_x, crop_center_y, crop_center_z, crop_size_x, crop_size_y, crop_size_z = crop

        points = crop_points_center_size(
            points,
            center_x=crop_center_x,
            center_y=crop_center_y,
            center_z=crop_center_z,
            size_x=crop_size_x,
            size_y=crop_size_y,
            size_z=crop_size_z,
        )

        if debug:
            print(
                f"[INFO] crop center=({crop_center_x}, {crop_center_y}, {crop_center_z}), "
                f"size=({crop_size_x}, {crop_size_y}, {crop_size_z})"
            )

    if debug:
        print(f"[INFO] cropped points: {len(points)}")

    if len(points) < 10:
        raise RuntimeError("Too few points remain after cropping.")

    plane_normal, plane_point, plane_points, _ = fit_plane_ransac_open3d(
        points,
        distance_threshold=plane_dist_thresh,
        ransac_n=3,
        num_iterations=3000,
    )

    if debug:
        print(f"[INFO] plane inliers: {len(plane_points)}")
        print(f"[INFO] plane normal: {plane_normal}")

    u, v = build_plane_basis(plane_normal)
    projected_2d = project_points_to_plane(plane_points, plane_point, u, v)

    count_img, occupancy, meta = rasterize_points_2d(projected_2d, pixel_size=pixel_size)

    hole_centers_2d_all, dbg = detect_holes_small_diameter_refined(
        count_img=count_img,
        occupancy=occupancy,
        meta=meta,
        expected_hole_diameter_mm=expected_hole_diameter_mm,
        support_blur_sigma_px=support_blur_sigma_px,
        support_thresh_ratio=support_thresh_ratio,
        circularity_min=circularity_min,
        debug=debug,
    )

    if len(hole_centers_2d_all) < 4:
        raise RuntimeError(
            f"Detected only {len(hole_centers_2d_all)} refined hole(s). "
            f"Try reducing pixel_size, support_blur_sigma_px, or circularity_min."
        )

    hole_centers_2d_used = select_best_rectangular_4hole_subset(hole_centers_2d_all, debug=debug)

    origin_2d, axis1_2d, axis2_2d, half_lengths_2d, fitted_rectangle_2d, ordered_pts_2d = \
        fit_constrained_rectangle(hole_centers_2d_used, debug=debug)

    axis1_3d = normalize(axis1_2d[0] * u + axis1_2d[1] * v)
    axis2_3d = normalize(axis2_2d[0] * u + axis2_2d[1] * v)
    plane_normal = normalize(plane_normal)

    axis1_3d, axis2_3d, plane_normal = ensure_right_handed(axis1_3d, axis2_3d, plane_normal)

    origin_3d = plane_point + origin_2d[0] * u + origin_2d[1] * v
    hole_centers_3d_used = unproject_2d_to_3d(hole_centers_2d_used, plane_point, u, v)

    edge_lengths_2d = compute_edge_lengths(fitted_rectangle_2d)

    return PlaneFrameResult(
        plane_normal_3d=plane_normal,
        axis1_3d=axis1_3d,
        axis2_3d=axis2_3d,
        origin_3d=origin_3d,
        hole_centers_2d_all=hole_centers_2d_all,
        hole_centers_2d_used=hole_centers_2d_used,
        hole_centers_3d_used=hole_centers_3d_used,
        plane_basis_u=u,
        plane_basis_v=v,
        plane_point=plane_point,
        projected_2d=projected_2d,
        fitted_rectangle_2d=fitted_rectangle_2d,
        ordered_rectangle_points_2d=ordered_pts_2d,
        rectangle_half_lengths_2d=half_lengths_2d,
        rectangle_edge_lengths_2d=edge_lengths_2d,
    ), dbg

def order_points_cyclic(points_2d: np.ndarray) -> np.ndarray:
    c = np.mean(points_2d, axis=0)
    ang = np.arctan2(points_2d[:, 1] - c[1], points_2d[:, 0] - c[0])
    order = np.argsort(ang)
    return points_2d[order]

def order_points_cyclic(points_2d: np.ndarray) -> np.ndarray:
    c = np.mean(points_2d, axis=0)
    ang = np.arctan2(points_2d[:, 1] - c[1], points_2d[:, 0] - c[0])
    order = np.argsort(ang)
    return points_2d[order]




def compute_edge_lengths(points_2d: np.ndarray) -> np.ndarray:
    """
    points_2d should already be cyclically ordered.
    Returns 4 edge lengths.
    """
    lengths = []
    for i in range(4):
        p0 = points_2d[i]
        p1 = points_2d[(i + 1) % 4]
        lengths.append(np.linalg.norm(p1 - p0))
    return np.array(lengths, dtype=np.float64)

def order_points_cyclic(points_2d: np.ndarray) -> np.ndarray:
    """
    Order 2D points cyclically around their centroid.
    Useful for drawing polygon/rectangle boundary.
    """
    c = np.mean(points_2d, axis=0)
    ang = np.arctan2(points_2d[:, 1] - c[1], points_2d[:, 0] - c[0])
    order = np.argsort(ang)
    return points_2d[order]

def visualize_result(result: PlaneFrameResult, dbg: dict):
    fig, axs = plt.subplots(2, 3, figsize=(15, 9))

    axs[0, 0].set_title("Projected plane points")
    axs[0, 0].scatter(result.projected_2d[:, 0], result.projected_2d[:, 1], s=0.2)
    axs[0, 0].axis("equal")
    axs[0, 0].grid(True)

    axs[0, 1].set_title("Dense image")
    axs[0, 1].imshow(dbg["dense"], cmap="gray")

    axs[0, 2].set_title("Support mask")
    axs[0, 2].imshow(dbg["support_mask"], cmap="gray")

    axs[1, 0].set_title("Occupancy closed")
    axs[1, 0].imshow(dbg["occ_closed"], cmap="gray")

    axs[1, 1].set_title("Holes mask")
    axs[1, 1].imshow(dbg["holes_mask"], cmap="gray")

    axs[1, 2].set_title("Refined holes + fitted rectangle")
    axs[1, 2].scatter(result.projected_2d[:, 0], result.projected_2d[:, 1], s=0.2, alpha=0.2)

    if len(result.hole_centers_2d_all) > 0:
        axs[1, 2].scatter(
            result.hole_centers_2d_all[:, 0],
            result.hole_centers_2d_all[:, 1],
            s=80,
            marker="x",
            label="all refined holes"
        )

    if len(result.hole_centers_2d_used) > 0:
        axs[1, 2].scatter(
            result.hole_centers_2d_used[:, 0],
            result.hole_centers_2d_used[:, 1],
            s=120,
            marker="o",
            facecolors="none",
            label="used 4 holes"
        )

        rect_pts = result.fitted_rectangle_2d
        rect_pts_closed = np.vstack([rect_pts, rect_pts[0]])
        axs[1, 2].plot(
            rect_pts_closed[:, 0],
            rect_pts_closed[:, 1],
            linewidth=2.0,
            label="fitted rectangle"
        )

        o = np.mean(rect_pts, axis=0)
        axs[1, 2].scatter(o[0], o[1], s=100, marker="+", label="origin")

        a1 = np.array([
            np.dot(result.axis1_3d, result.plane_basis_u),
            np.dot(result.axis1_3d, result.plane_basis_v)
        ])
        a2 = np.array([
            np.dot(result.axis2_3d, result.plane_basis_u),
            np.dot(result.axis2_3d, result.plane_basis_v)
        ])

        L = max(np.ptp(result.projected_2d[:, 0]), np.ptp(result.projected_2d[:, 1])) * 0.25
        axs[1, 2].arrow(o[0], o[1], L * a1[0], L * a1[1], width=0.0, head_width=0.5)
        axs[1, 2].arrow(o[0], o[1], L * a2[0], L * a2[1], width=0.0, head_width=0.5)

        edge_lengths = result.rectangle_edge_lengths_2d
        for i in range(4):
            p0 = rect_pts[i]
            p1 = rect_pts[(i + 1) % 4]
            mid = 0.5 * (p0 + p1)
            axs[1, 2].text(mid[0], mid[1], f"{edge_lengths[i]:.4f}", fontsize=9)

    axs[1, 2].legend()
    axs[1, 2].axis("equal")
    axs[1, 2].grid(True)

    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xyz", type=str, required=True)
    parser.add_argument("--plane_dist_thresh", type=float, default=0.20)
    parser.add_argument("--pixel_size", type=float, default=0.10)
    parser.add_argument("--expected_hole_diameter_mm", type=float, default=1.0)
    parser.add_argument("--support_blur_sigma_px", type=float, default=2.0)
    parser.add_argument("--support_thresh_ratio", type=float, default=0.05)
    parser.add_argument("--circularity_min", type=float, default=0.30)
    parser.add_argument("--no_vis", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--crop",
        type=float,
        nargs=6,
        default=None,
        metavar=("X", "Y", "Z", "SIZEX", "SIZEY", "SIZEZ"),
        help="Crop box as: center_x center_y center_z size_x size_y size_z",
    )
    args = parser.parse_args()

    result, dbg = compute_plane_axes_from_holes(
        xyz_path=args.xyz,
        plane_dist_thresh=args.plane_dist_thresh,
        pixel_size=args.pixel_size,
        expected_hole_diameter_mm=args.expected_hole_diameter_mm,
        support_blur_sigma_px=args.support_blur_sigma_px,
        support_thresh_ratio=args.support_thresh_ratio,
        circularity_min=args.circularity_min,
        crop=args.crop,
        debug=args.debug,
    )

    print("\n========== FINAL RESULT ==========")
    print("origin_3d       :", result.origin_3d)
    print("axis1_3d        :", result.axis1_3d)
    print("axis2_3d        :", result.axis2_3d)
    print("plane_normal_3d :", result.plane_normal_3d)
    print("num_holes_all   :", len(result.hole_centers_2d_all))
    print("num_holes_used  :", len(result.hole_centers_2d_used))

    print("hole_centers_2d_used:")
    for i, p in enumerate(result.hole_centers_2d_used):
        print(f"  H{i}: {p}")

    print("fitted_rectangle_2d:")
    for i, p in enumerate(result.fitted_rectangle_2d):
        print(f"  R{i}: {p}")

    print("rectangle_half_lengths_2d:", result.rectangle_half_lengths_2d)
    print("rectangle_edge_lengths_2d:")
    for i, L in enumerate(result.rectangle_edge_lengths_2d):
        print(f"  edge {i} (R{i}->R{(i+1)%4}): {L:.9f}")

    print("hole_centers_3d_used:")
    for i, p in enumerate(result.hole_centers_3d_used):
        print(f"  {i}: {p}")

    rect_pts = order_points_cyclic(result.hole_centers_2d_used)
    edge_lengths = compute_edge_lengths(rect_pts)

    print("rectangle_2d_used (ordered):")
    for i, p in enumerate(rect_pts):
        print(f"  R{i}: {p}")

    print("edge_lengths_2d:")
    for i, L in enumerate(edge_lengths):
        print(f"  edge {i} (R{i}->R{(i+1)%4}): {L}")

    if not args.no_vis:
        visualize_result(result, dbg)


if __name__ == "__main__":
    main()