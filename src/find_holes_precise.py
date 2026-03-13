import os
import json
import math
import argparse
import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt


# -----------------------------
# Basic IO
# -----------------------------
def load_xyz(path):
    pts = np.loadtxt(path, usecols=(0, 1, 2))
    if pts.ndim == 1:
        pts = pts.reshape(1, 3)
    return pts


# -----------------------------
# Plane fitting
# -----------------------------
def fit_plane_pca(pts):
    mu = pts.mean(axis=0)
    X = pts - mu
    cov = np.cov(X, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)

    n = eigvecs[:, 0]
    u = eigvecs[:, 2]
    v = np.cross(n, u)
    v /= np.linalg.norm(v)
    u = np.cross(v, n)
    u /= np.linalg.norm(u)

    return mu, u, v, n


def plane_distance(pts, mu, n):
    return (pts - mu) @ n


def fit_plane_ransac(pts, n_iters=300, dist_thresh=0.06, random_seed=0):
    rng = np.random.default_rng(random_seed)
    n_pts = len(pts)

    if n_pts < 3:
        return fit_plane_pca(pts)

    best_inliers = None
    best_count = -1

    for _ in range(n_iters):
        idx = rng.choice(n_pts, size=3, replace=False)
        p0, p1, p2 = pts[idx]

        v1 = p1 - p0
        v2 = p2 - p0
        n = np.cross(v1, v2)
        norm = np.linalg.norm(n)
        if norm < 1e-12:
            continue
        n = n / norm

        d = np.abs((pts - p0) @ n)
        inliers = d < dist_thresh
        count = int(inliers.sum())

        if count > best_count:
            best_count = count
            best_inliers = inliers

    if best_inliers is None or best_count < 10:
        return fit_plane_pca(pts)

    return fit_plane_pca(pts[best_inliers])


def project_to_plane(pts, mu, u, v, n):
    X = pts - mu
    pu = X @ u
    pv = X @ v
    pw = X @ n
    return np.column_stack([pu, pv, pw])


def uv_to_xyz(uv, mu, u, v):
    return mu + uv[0] * u + uv[1] * v


# -----------------------------
# Coarse 2D occupancy for candidate finding
# -----------------------------
def build_occupancy(xy, res):
    xy_min = xy.min(axis=0)
    xy_max = xy.max(axis=0)

    shape_xy = np.ceil((xy_max - xy_min) / res).astype(int) + 1
    w, h = int(shape_xy[0]), int(shape_xy[1])

    grid = np.zeros((h, w), dtype=np.uint8)

    ix = np.floor((xy[:, 0] - xy_min[0]) / res).astype(int)
    iy = np.floor((xy[:, 1] - xy_min[1]) / res).astype(int)

    valid = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
    grid[iy[valid], ix[valid]] = 1

    return grid, xy_min


def connected_components(binary_mask):
    structure = np.ones((3, 3), dtype=np.uint8)
    labeled, num = ndi.label(binary_mask, structure=structure)
    return labeled, num


def component_stats(labeled, num):
    objs = ndi.find_objects(labeled)
    stats = []

    for i in range(1, num + 1):
        sl = objs[i - 1]
        if sl is None:
            continue

        comp = (labeled[sl] == i)
        area = int(comp.sum())

        ys, xs = np.nonzero(comp)
        ys = ys + sl[0].start
        xs = xs + sl[1].start

        minr = int(ys.min())
        maxr = int(ys.max()) + 1
        minc = int(xs.min())
        maxc = int(xs.max()) + 1

        cy = float(ys.mean())
        cx = float(xs.mean())

        eroded = ndi.binary_erosion(comp, structure=np.ones((3, 3), dtype=bool))
        boundary = comp & (~eroded)
        perimeter = float(boundary.sum())

        stats.append({
            "label": i,
            "area": area,
            "bbox": (minr, minc, maxr, maxc),
            "centroid_rc": (cy, cx),
            "perimeter_px": perimeter,
        })
    return stats


def detect_coarse_holes_from_grid(grid, res, expected_diam_mm=1.1):
    occ = grid > 0

    occ = ndi.binary_closing(occ, structure=np.ones((3, 3), dtype=bool), iterations=1)
    occ = ndi.binary_opening(occ, structure=np.ones((2, 2), dtype=bool), iterations=1)

    labeled, num = connected_components(occ)
    stats = component_stats(labeled, num)
    if len(stats) == 0:
        return [], occ, None, None

    largest = max(stats, key=lambda s: s["area"])
    board_mask = (labeled == largest["label"])

    filled = ndi.binary_fill_holes(board_mask)
    holes_mask = filled & (~board_mask)

    hlbl, hnum = connected_components(holes_mask)
    hstats = component_stats(hlbl, hnum)

    results = []
    for s in hstats:
        area_px = s["area"]
        area_mm2 = area_px * (res ** 2)
        eq_diam = 2.0 * math.sqrt(area_mm2 / math.pi)

        minr, minc, maxr, maxc = s["bbox"]
        w = (maxc - minc) * res
        h = (maxr - minr) * res
        aspect = min(w, h) / max(w, h) if max(w, h) > 0 else 0.0

        perim_mm = s["perimeter_px"] * res
        circularity = 0.0
        if perim_mm > 1e-12:
            circularity = 4.0 * math.pi * area_mm2 / (perim_mm ** 2)

        if (
            0.5 * expected_diam_mm <= eq_diam <= 1.8 * expected_diam_mm
            and aspect > 0.45
            and circularity > 0.20
        ):
            cy, cx = s["centroid_rc"]
            results.append({
                "center_uv_local": np.array([cx * res, cy * res]),
                "eq_diam_mm": float(eq_diam),
                "bbox_mm": (float(w), float(h)),
                "area_mm2": float(area_mm2),
                "circularity": float(circularity),
                "bbox_px": s["bbox"],
            })

    return results, occ, board_mask, holes_mask


# -----------------------------
# Local raster around a coarse candidate
# -----------------------------
def make_local_density_image(xy_local_pts, center_uv, half_size_mm, res_mm, blur_sigma_px=1.2):
    x0 = center_uv[0] - half_size_mm
    x1 = center_uv[0] + half_size_mm
    y0 = center_uv[1] - half_size_mm
    y1 = center_uv[1] + half_size_mm

    w = int(np.ceil((x1 - x0) / res_mm)) + 1
    h = int(np.ceil((y1 - y0) / res_mm)) + 1

    img = np.zeros((h, w), dtype=np.float32)

    ix = np.floor((xy_local_pts[:, 0] - x0) / res_mm).astype(int)
    iy = np.floor((xy_local_pts[:, 1] - y0) / res_mm).astype(int)

    valid = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
    ix = ix[valid]
    iy = iy[valid]

    for xpix, ypix in zip(ix, iy):
        img[ypix, xpix] += 1.0

    if blur_sigma_px > 0:
        img = ndi.gaussian_filter(img, sigma=blur_sigma_px)

    meta = {
        "x0": x0,
        "y0": y0,
        "res_mm": res_mm,
        "width": w,
        "height": h,
    }
    return img, meta


def uv_to_img(uv, meta):
    xpix = (uv[0] - meta["x0"]) / meta["res_mm"]
    ypix = (uv[1] - meta["y0"]) / meta["res_mm"]
    return np.array([xpix, ypix])


def img_to_uv(xpix, ypix, meta):
    u = meta["x0"] + xpix * meta["res_mm"]
    v = meta["y0"] + ypix * meta["res_mm"]
    return np.array([u, v])


# -----------------------------
# Edge extraction
# -----------------------------
def compute_edge_strength_for_hole(img, sigma_smooth=1.0):
    sm = ndi.gaussian_filter(img, sigma=sigma_smooth)

    gx = ndi.sobel(sm, axis=1)
    gy = ndi.sobel(sm, axis=0)
    grad_mag = np.sqrt(gx * gx + gy * gy)

    # hole is a low-density region. The edge is where density rises sharply.
    return sm, grad_mag


def bilinear_sample(img, x, y):
    h, w = img.shape

    if x < 0 or y < 0 or x >= (w - 1) or y >= (h - 1):
        return 0.0

    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = x0 + 1
    y1 = y0 + 1

    dx = x - x0
    dy = y - y0

    v00 = img[y0, x0]
    v10 = img[y0, x1]
    v01 = img[y1, x0]
    v11 = img[y1, x1]

    return (
        v00 * (1 - dx) * (1 - dy) +
        v10 * dx * (1 - dy) +
        v01 * (1 - dx) * dy +
        v11 * dx * dy
    )


def radial_edge_search(
    edge_img,
    center_px,
    expected_radius_px,
    r_min_px,
    r_max_px,
    radial_step_px=0.5,
    n_angles=720,
    min_edge_value=0.01
):
    cx, cy = center_px
    edge_points = []
    strengths = []

    thetas = np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False)

    for th in thetas:
        ct = math.cos(th)
        st = math.sin(th)

        best_r = None
        best_val = -1.0

        r = r_min_px
        while r <= r_max_px:
            x = cx + r * ct
            y = cy + r * st
            val = bilinear_sample(edge_img, x, y)

            if val > best_val:
                best_val = val
                best_r = r
            r += radial_step_px

        if best_r is not None and best_val >= min_edge_value:
            x = cx + best_r * ct
            y = cy + best_r * st
            edge_points.append([x, y])
            strengths.append(best_val)

    if len(edge_points) == 0:
        return None

    return {
        "edge_points_px": np.array(edge_points, dtype=np.float64),
        "strengths": np.array(strengths, dtype=np.float64),
        "thetas": thetas[:len(edge_points)]
    }


# -----------------------------
# Circle fitting
# -----------------------------
def fit_circle_least_squares(x, y):
    A = np.column_stack([x, y, np.ones_like(x)])
    b = -(x * x + y * y)
    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    aa, bb, cc = sol

    cx = -aa / 2.0
    cy = -bb / 2.0
    r_sq = cx * cx + cy * cy - cc
    if r_sq <= 0:
        return None
    r = math.sqrt(r_sq)
    return cx, cy, r


def robust_circle_fit(points_xy, inlier_thresh=2.0, max_iters=5):
    if len(points_xy) < 8:
        return None

    x = points_xy[:, 0]
    y = points_xy[:, 1]

    fit = fit_circle_least_squares(x, y)
    if fit is None:
        return None

    cx, cy, r = fit

    for _ in range(max_iters):
        rr = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        resid = np.abs(rr - r)
        inlier = resid < inlier_thresh

        if inlier.sum() < 8:
            break

        fit2 = fit_circle_least_squares(x[inlier], y[inlier])
        if fit2 is None:
            break

        cx, cy, r = fit2

    rr = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    resid = np.abs(rr - r)
    rms = float(np.sqrt(np.mean((rr - r) ** 2)))

    return {
        "cx": float(cx),
        "cy": float(cy),
        "r": float(r),
        "residuals": resid,
        "rms": rms,
        "inliers": resid < inlier_thresh
    }


# -----------------------------
# Candidate refinement
# -----------------------------
def refine_single_hole(
    xy_plane,
    coarse_center_uv,
    coarse_diam_mm,
    local_res_mm=0.01,
    patch_scale=2.4,
    blur_sigma_px=1.0,
    radial_step_mm=0.005,
    n_angles=720,
    circle_inlier_thresh_mm=0.03,
):
    coarse_radius_mm = coarse_diam_mm / 2.0
    half_size_mm = max(1.5 * coarse_diam_mm, patch_scale * coarse_radius_mm)

    duv = xy_plane - coarse_center_uv[None, :]
    d = np.linalg.norm(duv, axis=1)

    # local crop around candidate
    crop_mask = d < (half_size_mm * 1.2)
    local_pts = xy_plane[crop_mask]
    if len(local_pts) < 30:
        return None

    img, meta = make_local_density_image(
        local_pts,
        center_uv=coarse_center_uv,
        half_size_mm=half_size_mm,
        res_mm=local_res_mm,
        blur_sigma_px=blur_sigma_px
    )

    sm, edge_img = compute_edge_strength_for_hole(img, sigma_smooth=1.0)

    center_px = uv_to_img(coarse_center_uv, meta)
    expected_radius_px = coarse_radius_mm / local_res_mm

    r_min_px = max(1.0, 0.5 * expected_radius_px)
    r_max_px = 1.6 * expected_radius_px
    radial_step_px = max(0.25, radial_step_mm / local_res_mm)

    edge_search = radial_edge_search(
        edge_img=edge_img,
        center_px=center_px,
        expected_radius_px=expected_radius_px,
        r_min_px=r_min_px,
        r_max_px=r_max_px,
        radial_step_px=radial_step_px,
        n_angles=n_angles,
        min_edge_value=0.01
    )

    if edge_search is None:
        return None

    edge_points_px = edge_search["edge_points_px"]

    fit_px = robust_circle_fit(
        edge_points_px,
        inlier_thresh=max(1.0, circle_inlier_thresh_mm / local_res_mm),
        max_iters=5
    )
    if fit_px is None:
        return None

    center_uv_refined = img_to_uv(fit_px["cx"], fit_px["cy"], meta)
    radius_mm_refined = fit_px["r"] * local_res_mm

    edge_points_uv = np.array([img_to_uv(p[0], p[1], meta) for p in edge_points_px])

    return {
        "center_uv": center_uv_refined,
        "radius_mm": float(radius_mm_refined),
        "diameter_mm": float(2.0 * radius_mm_refined),
        "rms_residual_mm": float(fit_px["rms"] * local_res_mm),
        "num_edge_points": int(len(edge_points_px)),
        "edge_points_uv": edge_points_uv,
        "local_img": img,
        "local_edge_img": edge_img,
        "local_meta": meta,
        "coarse_center_uv": coarse_center_uv,
    }


# -----------------------------
# Visualization
# -----------------------------
def save_2d_plot(path, xy_plane, coarse_results, refined_results):
    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(111)

    ax.scatter(xy_plane[:, 0], xy_plane[:, 1], s=1, alpha=0.25, label="plane points")

    for i, h in enumerate(coarse_results):
        c = h["center_uv"]
        r = h["eq_diam_mm"] / 2.0
        circ = plt.Circle((c[0], c[1]), r, fill=False, linewidth=1.2, linestyle="--")
        ax.add_patch(circ)
        ax.text(c[0], c[1], f"C{i}", fontsize=9)

    for i, h in enumerate(refined_results):
        c = h["center_uv"]
        r = h["radius_mm"]
        circ = plt.Circle((c[0], c[1]), r, fill=False, linewidth=2.0)
        ax.add_patch(circ)
        ax.scatter([c[0]], [c[1]], s=25)
        if "edge_points_uv" in h and len(h["edge_points_uv"]) > 0:
            ep = h["edge_points_uv"]
            ax.scatter(ep[:, 0], ep[:, 1], s=4, alpha=0.5)
        ax.text(c[0], c[1], f"R{i}", fontsize=10)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("u [mm]")
    ax.set_ylabel("v [mm]")
    ax.set_title("PCB hole detection and refined circle fitting")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close(fig)


def save_mask_plot(path, occ, board_mask, holes_mask):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(occ, origin="lower")
    axes[0].set_title("Occupancy")

    axes[1].imshow(board_mask, origin="lower")
    axes[1].set_title("Board mask")

    axes[2].imshow(holes_mask, origin="lower")
    axes[2].set_title("Coarse holes mask")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close(fig)


def save_3d_plot(path, pts, hole_centers_xyz):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    stride = max(1, len(pts) // 20000)
    p = pts[::stride]

    ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=1, alpha=0.25)

    if len(hole_centers_xyz) > 0:
        hc = np.array(hole_centers_xyz)
        ax.scatter(hc[:, 0], hc[:, 1], hc[:, 2], s=60)
        for i, c in enumerate(hc):
            ax.text(c[0], c[1], c[2], f"H{i}", fontsize=9)

    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Y [mm]")
    ax.set_zlabel("Z [mm]")
    ax.set_title("3D point cloud with refined hole centers")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close(fig)


def save_local_refinement_panels(path, refined_results, max_cols=3):
    if len(refined_results) == 0:
        return

    n = len(refined_results)
    cols = min(max_cols, n)
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows))
    axes = np.atleast_2d(axes)

    for idx in range(rows * cols):
        r = idx // cols
        c = idx % cols
        ax = axes[r, c]

        if idx >= n:
            ax.axis("off")
            continue

        rr = refined_results[idx]
        img = rr["local_edge_img"]
        meta = rr["local_meta"]
        center_px = uv_to_img(rr["center_uv"], meta)
        radius_px = rr["radius_mm"] / meta["res_mm"]

        ax.imshow(img, origin="lower")
        circ = plt.Circle((center_px[0], center_px[1]), radius_px, fill=False, linewidth=2.0)
        ax.add_patch(circ)

        ep = rr["edge_points_uv"]
        ep_px = np.array([uv_to_img(p, meta) for p in ep])
        ax.scatter(ep_px[:, 0], ep_px[:, 1], s=5)

        ax.set_title(
            f"H{idx}: d={rr['diameter_mm']:.4f} mm\n"
            f"rms={rr['rms_residual_mm']:.4f} mm"
        )
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close(fig)


# -----------------------------
# Main pipeline
# -----------------------------
def find_pcb_holes_precise(
    path,
    expected_diam_mm=1.1,
    plane_thresh_mm=0.08,
    coarse_grid_res_mm=0.03,
    local_res_mm=0.01,
    radial_step_mm=0.005,
    n_angles=720,
    circle_inlier_thresh_mm=0.03,
    out_dir="hole_detection_output",
):
    os.makedirs(out_dir, exist_ok=True)

    pts = load_xyz(path)

    # robust plane fit
    mu, u, v, n = fit_plane_ransac(
        pts,
        n_iters=300,
        dist_thresh=max(0.03, plane_thresh_mm),
        random_seed=0
    )

    proj = project_to_plane(pts, mu, u, v, n)
    near = np.abs(proj[:, 2]) < plane_thresh_mm
    xy_plane = proj[near, :2]

    # coarse detection
    grid, xy_min = build_occupancy(xy_plane, coarse_grid_res_mm)
    coarse_local, occ, board_mask, holes_mask = detect_coarse_holes_from_grid(
        grid, coarse_grid_res_mm, expected_diam_mm
    )

    coarse_results = []
    refined_results = []
    final_results = []

    for i, h in enumerate(coarse_local):
        coarse_center_uv = h["center_uv_local"] + xy_min

        coarse_results.append({
            "center_uv": coarse_center_uv,
            "eq_diam_mm": h["eq_diam_mm"],
            "bbox_mm": h["bbox_mm"],
            "area_mm2": h["area_mm2"],
            "circularity": h["circularity"],
        })

        refined = refine_single_hole(
            xy_plane=xy_plane,
            coarse_center_uv=coarse_center_uv,
            coarse_diam_mm=h["eq_diam_mm"],
            local_res_mm=local_res_mm,
            patch_scale=2.4,
            blur_sigma_px=1.0,
            radial_step_mm=radial_step_mm,
            n_angles=n_angles,
            circle_inlier_thresh_mm=circle_inlier_thresh_mm,
        )

        if refined is None:
            continue

        center_xyz = uv_to_xyz(refined["center_uv"], mu, u, v)

        refined_results.append(refined)
        final_results.append({
            "index": i,
            "center_uv_mm": refined["center_uv"].tolist(),
            "center_xyz_mm": center_xyz.tolist(),
            "diameter_mm": refined["diameter_mm"],
            "radius_mm": refined["radius_mm"],
            "rms_residual_mm": refined["rms_residual_mm"],
            "num_edge_points": refined["num_edge_points"],
        })

    final_results.sort(key=lambda d: (d["center_xyz_mm"][1], d["center_xyz_mm"][0]))

    save_2d_plot(
        os.path.join(out_dir, "holes_2d_refined.png"),
        xy_plane,
        coarse_results,
        refined_results
    )

    if occ is not None and board_mask is not None and holes_mask is not None:
        save_mask_plot(
            os.path.join(out_dir, "holes_masks_refined.png"),
            occ, board_mask, holes_mask
        )

    hole_centers_xyz = [r["center_xyz_mm"] for r in final_results]
    save_3d_plot(
        os.path.join(out_dir, "holes_3d_refined.png"),
        pts,
        hole_centers_xyz
    )

    save_local_refinement_panels(
        os.path.join(out_dir, "holes_local_refinement.png"),
        refined_results
    )

    output = {
        "input_file": path,
        "num_holes": len(final_results),
        "plane_origin_mm": mu.tolist(),
        "plane_u_axis": u.tolist(),
        "plane_v_axis": v.tolist(),
        "plane_normal": n.tolist(),
        "holes": final_results,
    }

    with open(os.path.join(out_dir, "holes_results_refined.json"), "w") as f:
        json.dump(output, f, indent=2)

    return output


# -----------------------------
# CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Precise PCB hole detection from XYZ point cloud")
    parser.add_argument("--input", required=True, help="Path to input .xyz file")
    parser.add_argument("--expected-diam-mm", type=float, default=1.1, help="Expected hole diameter in mm")
    parser.add_argument("--plane-thresh-mm", type=float, default=0.08, help="Distance-to-plane threshold in mm")
    parser.add_argument("--coarse-grid-res-mm", type=float, default=0.03, help="Coarse occupancy grid resolution in mm")
    parser.add_argument("--local-res-mm", type=float, default=0.01, help="Local refinement image resolution in mm")
    parser.add_argument("--radial-step-mm", type=float, default=0.005, help="Radial search step in mm")
    parser.add_argument("--n-angles", type=int, default=720, help="Number of angular samples for radial edge search")
    parser.add_argument("--circle-inlier-thresh-mm", type=float, default=0.03, help="Circle fit inlier threshold in mm")
    parser.add_argument("--out-dir", default="hole_detection_output", help="Output directory")
    args = parser.parse_args()

    result = find_pcb_holes_precise(
        path=args.input,
        expected_diam_mm=args.expected_diam_mm,
        plane_thresh_mm=args.plane_thresh_mm,
        coarse_grid_res_mm=args.coarse_grid_res_mm,
        local_res_mm=args.local_res_mm,
        radial_step_mm=args.radial_step_mm,
        n_angles=args.n_angles,
        circle_inlier_thresh_mm=args.circle_inlier_thresh_mm,
        out_dir=args.out_dir,
    )

    print(f"Found {result['num_holes']} refined holes")
    for i, h in enumerate(result["holes"]):
        c = h["center_xyz_mm"]
        print(
            f"[H{i}] center_xyz = ({c[0]:.6f}, {c[1]:.6f}, {c[2]:.6f}) mm, "
            f"diameter = {h['diameter_mm']:.6f} mm, "
            f"rms_residual = {h['rms_residual_mm']:.6f} mm, "
            f"num_edge_points = {h['num_edge_points']}"
        )

    print(f"Saved outputs to: {args.out_dir}")


if __name__ == "__main__":
    main()