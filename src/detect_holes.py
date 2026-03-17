#!/usr/bin/env python3
"""
PCB Point Cloud Hole Detection  v3
====================================
Robust hole detection for PCB point clouds — works on CAD or sensor data,
handles arbitrarily oriented boards (not just XY-flat).

Algorithm
---------
1. Load XYZ point cloud (with optional normals)
2. RANSAC plane fitting (tries X/Y/Z as dependent axis, picks best)
3. Project surface-inlier points onto plane's local 2D frame
4. Rasterize to binary occupancy grid at adaptive resolution
5. Adaptive morphological closing to bridge point-sampling gaps
6. Flood-fill from border → isolate interior voids
7. Connected-component labelling + filtering by:
     – Compensated equivalent diameter (accounts for closing erosion)
     – Circularity (real holes are round; sampling artefacts are not)
8. Back-project hole centres to original 3D coordinates
9. Save comprehensive debug images at every stage + CSV of results
"""

import sys, os, time, textwrap
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import cv2
from sklearn.linear_model import RANSACRegressor
from skimage import measure, color
from pathlib import Path


# ═══════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════
class Config:
    """All tuneable knobs in one place."""

    # --- Hole size range (mm) ---
    MIN_HOLE_DIAMETER_MM = 0.8    # slightly below 1 mm to catch edge cases
    MAX_HOLE_DIAMETER_MM = 2.5    # slightly above 2 mm for tolerance

    # --- Grid ---
    GRID_RESOLUTION_MM = None     # None → auto (~20 px per min-hole diameter)

    # --- RANSAC plane fitting ---
    RANSAC_RESIDUAL_MM = 0.3      # max dist from plane for inlier
    RANSAC_MIN_SAMPLES = 3
    SURFACE_THICKNESS_MM = 1.0    # keep points within this of the plane

    # --- Morphological closing ---
    # The kernel radius (px) is computed adaptively from point density.
    # This multiplier controls how aggressively gaps are bridged:
    #   close_r_px = ceil(avg_spacing_px * CLOSE_MULT)
    # 1.5–2.0 is a good range; higher merges more gaps but erodes holes.
    CLOSE_MULT = 1.8

    # --- Hole filtering ---
    # Solidity = area / convex_hull_area. Robust to jagged sensor edges.
    # Real holes: > 0.85.  Noise artefacts: < 0.7.
    MIN_SOLIDITY = 0.80
    # Axis ratio = minor_axis / major_axis.  Circle ≈ 1.0.
    MIN_AXIS_RATIO = 0.60
    BORDER_MARGIN_PX = 3


def log(msg, level="INFO"):
    print(f"[{time.strftime('%H:%M:%S')}] [{level}] {msg}")


# ═══════════════════════════════════════════════════════════
#  STEP 1  ─  LOAD
# ═══════════════════════════════════════════════════════════
def load_point_cloud(filepath):
    log(f"Loading: {filepath}")
    data = np.loadtxt(filepath)
    pts = data[:, :3].copy()
    normals = data[:, 3:6] if data.shape[1] >= 6 else None
    log(f"  {pts.shape[0]:,} pts | X[{pts[:,0].min():.2f},{pts[:,0].max():.2f}] "
        f"Y[{pts[:,1].min():.2f},{pts[:,1].max():.2f}] "
        f"Z[{pts[:,2].min():.2f},{pts[:,2].max():.2f}]")
    return pts, normals


# ═══════════════════════════════════════════════════════════
#  STEP 2  ─  RANSAC PLANE FIT
# ═══════════════════════════════════════════════════════════
def fit_plane_ransac(points, cfg):
    log("RANSAC plane fitting …")
    n_pts = len(points)
    rng = np.random.RandomState(42)
    si = rng.choice(n_pts, min(100_000, n_pts), replace=False) if n_pts > 100_000 else np.arange(n_pts)
    ps = points[si]

    best_ni, best = 0, None
    for dep in range(3):
        indep = [i for i in range(3) if i != dep]
        try:
            ransac = RANSACRegressor(residual_threshold=cfg.RANSAC_RESIDUAL_MM,
                                     min_samples=cfg.RANSAC_MIN_SAMPLES,
                                     max_trials=3000, random_state=42)
            ransac.fit(ps[:, indep], ps[:, dep])
            ni = ransac.inlier_mask_.sum()
            if ni > best_ni:
                a, b = ransac.estimator_.coef_
                nrm = np.zeros(3)
                nrm[indep[0]], nrm[indep[1]], nrm[dep] = -a, -b, 1.0
                nrm /= np.linalg.norm(nrm)
                ppt = ps[ransac.inlier_mask_].mean(0)
                best_ni, best = ni, (nrm, ppt)
        except Exception:
            pass

    if best is None:
        raise RuntimeError("Plane fitting failed")
    nrm, ppt = best
    dists = np.abs((points - ppt) @ nrm)
    inlier = dists < cfg.SURFACE_THICKNESS_MM
    tilt = np.degrees(np.arccos(min(1.0, abs(nrm[2]))))
    log(f"  Normal=[{nrm[0]:.4f},{nrm[1]:.4f},{nrm[2]:.4f}]  tilt={tilt:.1f}°")
    log(f"  Inliers: {inlier.sum():,}/{n_pts:,} ({100*inlier.mean():.1f}%)")
    return nrm, ppt, inlier, dists


# ═══════════════════════════════════════════════════════════
#  STEP 3  ─  PROJECT TO 2D
# ═══════════════════════════════════════════════════════════
def project_to_plane(points, nrm, ppt):
    log("Projecting to 2D …")
    arb = np.array([0,0,1.]) if abs(nrm[2]) < 0.9 else np.array([1,0,0.])
    u_ax = np.cross(nrm, arb); u_ax /= np.linalg.norm(u_ax)
    v_ax = np.cross(nrm, u_ax); v_ax /= np.linalg.norm(v_ax)
    c = points - ppt
    return c @ u_ax, c @ v_ax, u_ax, v_ax


# ═══════════════════════════════════════════════════════════
#  STEP 4  ─  RASTERIZE
# ═══════════════════════════════════════════════════════════
def rasterize(u, v, cfg):
    log("Rasterizing …")
    u_mn, u_mx = u.min(), u.max()
    v_mn, v_mx = v.min(), v.max()
    su, sv = u_mx - u_mn, v_mx - v_mn
    area = su * sv
    dens = len(u) / area
    avg_sp = 1.0 / np.sqrt(dens)
    log(f"  density={dens:.1f} pts/mm²  avg_spacing={avg_sp:.4f} mm")

    if cfg.GRID_RESOLUTION_MM is None:
        res = min(avg_sp / 2.0, cfg.MIN_HOLE_DIAMETER_MM / 20.0)
    else:
        res = cfg.GRID_RESOLUTION_MM

    margin = 1.0
    cols = int(np.ceil((su + 2*margin) / res))
    rows = int(np.ceil((sv + 2*margin) / res))
    mx = 6000
    if max(rows, cols) > mx:
        res *= max(rows, cols) / mx
        cols = int(np.ceil((su + 2*margin) / res))
        rows = int(np.ceil((sv + 2*margin) / res))

    log(f"  res={res:.4f} mm/px  grid={cols}×{rows}")

    up = np.clip(((u - u_mn + margin) / res).astype(int), 0, cols-1)
    vp = np.clip(((v - v_mn + margin) / res).astype(int), 0, rows-1)
    grid = np.zeros((rows, cols), dtype=np.uint8)
    grid[vp, up] = 255

    occ = (grid > 0).sum()
    log(f"  occupied: {occ:,}/{rows*cols:,} ({100*occ/(rows*cols):.1f}%)")

    avg_sp_px = avg_sp / res
    return grid, res, (u_mn - margin, v_mn - margin), (rows, cols), avg_sp_px


# ═══════════════════════════════════════════════════════════
#  STEP 5  ─  MORPHOLOGICAL CLOSING + FLOOD FILL
# ═══════════════════════════════════════════════════════════
def morph_and_flood(grid, avg_sp_px, cfg):
    log("Morphological closing + flood fill …")
    close_r = max(2, int(np.ceil(avg_sp_px * cfg.CLOSE_MULT)))
    log(f"  close_r = {close_r} px  (avg_sp={avg_sp_px:.1f} px × {cfg.CLOSE_MULT})")

    ks = 2 * close_r + 1
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
    closed = cv2.morphologyEx(grid, cv2.MORPH_CLOSE, kern)

    h, w = closed.shape
    inv = cv2.bitwise_not(closed)
    flood = inv.copy()
    ff_mask = np.zeros((h+2, w+2), dtype=np.uint8)

    # Fill from all border pixels
    for x in range(w):
        if flood[0, x] == 255:
            cv2.floodFill(flood, ff_mask, (x, 0), 128)
        if flood[h-1, x] == 255:
            cv2.floodFill(flood, ff_mask, (x, h-1), 128)
    for y in range(h):
        if flood[y, 0] == 255:
            cv2.floodFill(flood, ff_mask, (0, y), 128)
        if flood[y, w-1] == 255:
            cv2.floodFill(flood, ff_mask, (w-1, y), 128)

    interior = (flood == 255).astype(np.uint8) * 255
    log(f"  interior void pixels: {(interior>0).sum():,}")
    return closed, inv, interior, close_r


# ═══════════════════════════════════════════════════════════
#  STEP 6  ─  DETECT & FILTER HOLES
# ═══════════════════════════════════════════════════════════
def detect_holes(interior, res, cfg, close_r):
    log("Connected-component detection …")
    labels = measure.label(interior > 0, connectivity=2)
    regions = measure.regionprops(labels)
    log(f"  raw candidates: {len(regions)}")

    # Diameter compensation: closing shrinks holes by ~close_r px on each side
    comp_mm = 2 * close_r * res

    # Size bounds (in pixels) BEFORE compensation
    min_raw_d_px = max(1, (cfg.MIN_HOLE_DIAMETER_MM - comp_mm) / res)
    max_raw_d_px = (cfg.MAX_HOLE_DIAMETER_MM + comp_mm) / res  # generous upper
    min_area = np.pi * (min_raw_d_px / 2)**2 * 0.30
    max_area = np.pi * (max_raw_d_px / 2)**2 * 2.5

    h_img, w_img = interior.shape
    holes = []
    rej = {"size": 0, "shape": 0, "border": 0}

    for reg in regions:
        a = reg.area
        if a < min_area or a > max_area:
            rej["size"] += 1; continue
        bb = reg.bbox
        bm = cfg.BORDER_MARGIN_PX
        if bb[0] < bm or bb[1] < bm or bb[2] > h_img-bm or bb[3] > w_img-bm:
            rej["border"] += 1; continue

        # Shape filters — robust to jagged sensor-data edges
        solidity = reg.solidity   # area / convex_hull_area
        maj = reg.axis_major_length
        mnr = reg.axis_minor_length
        axis_ratio = (mnr / maj) if maj > 0 else 0
        p = reg.perimeter
        circ = (4*np.pi*a / p**2) if p > 0 else 0   # kept for reporting

        if solidity < cfg.MIN_SOLIDITY or axis_ratio < cfg.MIN_AXIS_RATIO:
            rej["shape"] += 1; continue

        raw_d = reg.equivalent_diameter_area * res
        d = raw_d + comp_mm

        # Final size check on compensated diameter
        if d < cfg.MIN_HOLE_DIAMETER_MM or d > cfg.MAX_HOLE_DIAMETER_MM:
            rej["size"] += 1; continue

        cy, cx = reg.centroid
        holes.append(dict(
            label=reg.label, center_px=(cx, cy),
            area_px=a, raw_diam_mm=raw_d, diameter_mm=d,
            circularity=circ, solidity=solidity,
            axis_ratio=axis_ratio, bbox=bb,
        ))

    log(f"  rejected: size={rej['size']}  shape={rej['shape']}  border={rej['border']}")
    log(f"  ✓ DETECTED: {len(holes)} holes")
    for i, h in enumerate(holes):
        log(f"    #{i+1}: ⌀{h['diameter_mm']:.3f}mm  sol={h['solidity']:.3f}  "
            f"ar={h['axis_ratio']:.3f}  circ={h['circularity']:.3f}")
    return holes, labels


# ═══════════════════════════════════════════════════════════
#  STEP 7  ─  BACK-PROJECT TO 3D
# ═══════════════════════════════════════════════════════════
def backproject(holes, origin, res, ppt, u_ax, v_ax):
    log("Back-projecting to 3D …")
    for h in holes:
        cx, cy = h['center_px']
        u_mm = cx * res + origin[0]
        v_mm = cy * res + origin[1]
        h['center_3d'] = ppt + u_mm * u_ax + v_mm * v_ax
        h['center_2d_mm'] = (u_mm, v_mm)
    return holes


# ═══════════════════════════════════════════════════════════
#  STEP 8  ─  SAVE DEBUG IMAGES + CSV
# ═══════════════════════════════════════════════════════════
def save_images(pts, inlier, dists, nrm, u, v,
                grid_raw, closed, inv, interior, labels, holes,
                res, origin, gshape, outdir):
    os.makedirs(outdir, exist_ok=True)
    log(f"Saving images → {outdir}/")
    D = 150

    # --- 1  3D cloud coloured by plane distance ---
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ss = np.random.RandomState(0).choice(len(pts), min(50000, len(pts)), replace=False)
    sc = ax.scatter(pts[ss,0], pts[ss,1], pts[ss,2],
                    c=dists[ss], cmap='coolwarm', s=0.3,
                    vmin=0, vmax=np.percentile(dists, 95))
    plt.colorbar(sc, ax=ax, label='mm', shrink=0.6)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title(f'Step 1: Point Cloud — Distance to Fitted Plane\n'
                 f'Normal=[{nrm[0]:.3f},{nrm[1]:.3f},{nrm[2]:.3f}]')
    fig.savefig(f'{outdir}/01_plane_distance.png', dpi=D, bbox_inches='tight'); plt.close()
    log("  01_plane_distance.png")

    # --- 2  2D scatter ---
    fig, ax = plt.subplots(figsize=(12, 10))
    ss2 = np.random.RandomState(0).choice(len(u), min(80000, len(u)), replace=False)
    ax.scatter(u[ss2], v[ss2], s=0.1, c='steelblue', alpha=0.4)
    ax.set_aspect('equal'); ax.invert_yaxis()
    ax.set_xlabel('U (mm)'); ax.set_ylabel('V (mm)')
    ax.set_title(f'Step 2: 2D Projection ({len(u):,} pts)')
    fig.savefig(f'{outdir}/02_projected_2d.png', dpi=D, bbox_inches='tight'); plt.close()
    log("  02_projected_2d.png")

    # --- 3  Raw occupancy ---
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(grid_raw, cmap='gray'); ax.set_title('Step 3: Raw Binary Occupancy')
    fig.savefig(f'{outdir}/03_occupancy_raw.png', dpi=D, bbox_inches='tight'); plt.close()
    log("  03_occupancy_raw.png")

    # --- 4  After closing ---
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(closed, cmap='gray'); ax.set_title('Step 4: After Morphological Closing')
    fig.savefig(f'{outdir}/04_closed.png', dpi=D, bbox_inches='tight'); plt.close()
    log("  04_closed.png")

    # --- 5  Inverted ---
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(inv, cmap='gray'); ax.set_title('Step 5: Inverted (white = void)')
    fig.savefig(f'{outdir}/05_inverted.png', dpi=D, bbox_inches='tight'); plt.close()
    log("  05_inverted.png")

    # --- 6  Interior holes ---
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(interior, cmap='gray')
    ax.set_title(f'Step 6: Interior Holes (exterior flood-filled)')
    fig.savefig(f'{outdir}/06_interior_holes.png', dpi=D, bbox_inches='tight'); plt.close()
    log("  06_interior_holes.png")

    # --- 7  Candidates (coloured) ---
    fig, ax = plt.subplots(figsize=(12, 10))
    if labels.max() > 0:
        lr = color.label2rgb(labels, bg_label=0, bg_color=(0,0,0))
        bg = np.where(closed[:,:,None] > 0, 0.25, 0)
        ax.imshow(np.clip(bg + lr*0.75, 0, 1))
    else:
        ax.imshow(closed, cmap='gray')
    ax.set_title(f'Step 7: All Candidate Regions ({labels.max()})')
    fig.savefig(f'{outdir}/07_candidates.png', dpi=D, bbox_inches='tight'); plt.close()
    log("  07_candidates.png")

    # --- 8  FINAL RESULT ---
    fig, ax = plt.subplots(figsize=(14, 12))
    rgb = np.stack([closed]*3, axis=-1).copy()
    for h in holes:
        rgb[labels == h['label']] = [255, 50, 50]
    ax.imshow(rgb, interpolation='nearest')
    for i, h in enumerate(holes):
        cx, cy = h['center_px']
        r_px = (h['diameter_mm']/2) / res
        ax.add_patch(Circle((cx, cy), r_px*1.25, fill=False, ec='lime', lw=1.5))
        ax.plot(cx, cy, '+', color='cyan', ms=8, mew=1.5)
        ax.text(cx + r_px*1.4, cy,
                f"#{i+1}\n⌀{h['diameter_mm']:.2f}mm",
                color='yellow', fontsize=7, fontweight='bold', va='center',
                bbox=dict(boxstyle='round,pad=0.15', fc='black', alpha=0.75))
    if holes:
        ds = [h['diameter_mm'] for h in holes]
        title = (f"DETECTED HOLES: {len(holes)}   |   "
                 f"⌀ {min(ds):.2f} – {max(ds):.2f} mm")
    else:
        title = "DETECTED HOLES: 0"
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(f'pixels  ({res:.4f} mm/px)')
    fig.savefig(f'{outdir}/08_final_result.png', dpi=200, bbox_inches='tight'); plt.close()
    log("  08_final_result.png")

    # --- 9  Close-ups ---
    if holes:
        nh = len(holes)
        nc = min(7, nh)
        nr = max(1, (nh + nc - 1) // nc)
        fig, axes = plt.subplots(nr, nc, figsize=(3*nc, 3*nr), squeeze=False)
        for idx, hh in enumerate(holes):
            r, c = divmod(idx, nc)
            a = axes[r][c]
            cx, cy = hh['center_px']
            m = int((hh['diameter_mm']*2.0) / res)
            y0, y1 = max(0,int(cy)-m), min(gshape[0],int(cy)+m)
            x0, x1 = max(0,int(cx)-m), min(gshape[1],int(cx)+m)
            a.imshow(rgb[y0:y1, x0:x1], interpolation='nearest')
            a.set_title(f'#{idx+1}: ⌀{hh["diameter_mm"]:.2f}mm\n'
                        f'sol={hh["solidity"]:.2f} ar={hh["axis_ratio"]:.2f}', fontsize=8)
            a.set_xticks([]); a.set_yticks([])
        for idx in range(nh, nr*nc):
            axes[idx//nc][idx%nc].set_visible(False)
        fig.suptitle('Individual Hole Close-ups', fontweight='bold')
        fig.tight_layout()
        fig.savefig(f'{outdir}/09_closeups.png', dpi=D, bbox_inches='tight'); plt.close()
        log("  09_closeups.png")

    # --- 10  Montage ---
    fig, axs = plt.subplots(2, 4, figsize=(22, 11))
    panels = [
        (grid_raw, 'gray', 'Raw Occupancy'),
        (closed, 'gray', 'After Closing'),
        (inv, 'gray', 'Inverted'),
        (interior, 'gray', 'Interior Holes'),
        ('cand', None, 'Candidates'),
        (rgb, None, 'Final'),
        ('hist', None, 'Diameter Histogram'),
        ('info', None, 'Summary'),
    ]
    for ax, (img, cm, t) in zip(axs.flat, panels):
        if isinstance(img, str):
            if img == 'cand' and labels.max() > 0:
                ax.imshow(color.label2rgb(labels, bg_label=0, bg_color=(0,0,0)))
            elif img == 'hist' and holes:
                ds = [hx['diameter_mm'] for hx in holes]
                ax.hist(ds, bins=max(5, len(ds)//2+1), color='steelblue', ec='black')
                ax.axvline(1.0, c='red', ls='--', lw=1); ax.axvline(2.0, c='red', ls='--', lw=1)
                ax.set_xlabel('⌀ (mm)')
            elif img == 'info':
                info = f"Holes: {len(holes)}\n"
                if holes:
                    ds = [hx['diameter_mm'] for hx in holes]
                    info += f"Min ⌀: {min(ds):.3f} mm\nMax ⌀: {max(ds):.3f} mm\n"
                    info += f"Mean ⌀: {np.mean(ds):.3f} mm\nRes: {res:.4f} mm/px"
                ax.text(0.1, 0.5, info, transform=ax.transAxes, fontsize=11,
                        va='center', family='monospace',
                        bbox=dict(boxstyle='round', fc='lightyellow'))
            ax.set_title(t, fontsize=10)
        elif img is not None:
            ax.imshow(img, cmap=cm, interpolation='nearest')
            ax.set_title(t, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle('PCB Hole Detection — Debug Montage', fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(f'{outdir}/10_montage.png', dpi=D, bbox_inches='tight'); plt.close()
    log("  10_montage.png")


def save_csv(holes, outdir):
    path = f'{outdir}/hole_centers.csv'
    with open(path, 'w') as f:
        f.write("hole_id,x_3d,y_3d,z_3d,u_2d_mm,v_2d_mm,"
                "diameter_mm,solidity,axis_ratio,circularity\n")
        for i, h in enumerate(holes):
            c3 = h['center_3d']; c2 = h['center_2d_mm']
            f.write(f"{i+1},{c3[0]:.6f},{c3[1]:.6f},{c3[2]:.6f},"
                    f"{c2[0]:.6f},{c2[1]:.6f},{h['diameter_mm']:.4f},"
                    f"{h['solidity']:.4f},{h['axis_ratio']:.4f},"
                    f"{h['circularity']:.4f}\n")
    return path


# ═══════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ═══════════════════════════════════════════════════════════
def detect_holes_pipeline(filepath, outdir=None, cfg=None):
    if cfg is None:
        cfg = Config()
    if outdir is None:
        outdir = f'/home/hyun/output_{Path(filepath).stem}'

    log("=" * 65)
    log("  PCB HOLE DETECTION  v3")
    log(f"  Input : {filepath}")
    log(f"  Output: {outdir}/")
    log(f"  Target hole ⌀: {cfg.MIN_HOLE_DIAMETER_MM}–{cfg.MAX_HOLE_DIAMETER_MM} mm")
    log("=" * 65)

    pts, _ = load_point_cloud(filepath)
    nrm, ppt, inlier, dists = fit_plane_ransac(pts, cfg)
    in_pts = pts[inlier]
    log(f"Surface inliers: {in_pts.shape[0]:,}")

    u, v, u_ax, v_ax = project_to_plane(in_pts, nrm, ppt)
    grid_raw, res, origin, gshape, avg_sp_px = rasterize(u, v, cfg)
    closed, inv, interior, close_r = morph_and_flood(grid_raw, avg_sp_px, cfg)
    holes, labels = detect_holes(interior, res, cfg, close_r)
    holes = backproject(holes, origin, res, ppt, u_ax, v_ax)

    save_images(pts, inlier, dists, nrm, u, v,
                grid_raw, closed, inv, interior, labels, holes,
                res, origin, gshape, outdir)
    save_csv(holes, outdir)

    # ── SUMMARY ──
    log("=" * 65)
    log("  RESULTS")
    log("=" * 65)
    log(f"  Holes: {len(holes)}")
    if holes:
        ds = [h['diameter_mm'] for h in holes]
        log(f"  ⌀ range: {min(ds):.3f} – {max(ds):.3f} mm  (mean {np.mean(ds):.3f})")
    log("")
    hdr = f"{'#':>3}  {'X':>10} {'Y':>10} {'Z':>10}  {'⌀(mm)':>8} {'Sol':>6} {'AR':>6}"
    log(hdr)
    log("─" * len(hdr))
    for i, h in enumerate(holes):
        c = h['center_3d']
        log(f"{i+1:>3}  {c[0]:>10.4f} {c[1]:>10.4f} {c[2]:>10.4f}  "
            f"{h['diameter_mm']:>8.3f} {h['solidity']:>6.3f} {h['axis_ratio']:>6.3f}")
    log("=" * 65)
    return holes, outdir


# ───────────────────────────────────────────────────────────
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python pcb_hole_detect.py <file.xyz> [outdir]")
        sys.exit(1)
    detect_holes_pipeline(sys.argv[1],
                          sys.argv[2] if len(sys.argv) > 2 else None)