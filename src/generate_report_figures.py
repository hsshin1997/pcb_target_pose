#!/usr/bin/env python3
"""
Generate all figures for the PCB Hole Detection Comparative Analysis report.

Figures 01-08 are synthetic/analytical visualisations created by this script.
Figures 09-12 are real detection results copied from Method A/B output directories.

Usage:
    python generate_report_figures.py [--outdir ./report_figs]

Dependencies:
    pip install matplotlib numpy opencv-python-headless scikit-image
"""

import os
import argparse
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle

# ═══════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════
DPI = 200


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', default='./report_figs')
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    OUT = args.outdir

    # ═══════════════════════════════════════════════════════════
    #  FIGURE 1: Pipeline Architecture Comparison
    # ═══════════════════════════════════════════════════════════
    print("Generating Figure 01: Pipeline architecture comparison...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 9))

    steps_a = [
        ("1. Load XYZ", "#4A90D9"),
        ("2. RANSAC Plane Fit\n(multi-axis, sklearn)", "#7B68EE"),
        ("3. Project \u2192 2D\n(orthonormal basis)", "#2E8B57"),
        ("4. Adaptive Rasterisation\n(auto resolution)", "#D4A017"),
        ("5. Morph Close\n+ Flood Fill", "#CD5C5C"),
        ("6. CC Labelling\n+ Solidity/AR Filter", "#20B2AA"),
        ("7. Back-project \u2192 3D", "#9370DB"),
    ]
    steps_b = [
        ("1. Load XYZ", "#4A90D9"),
        ("2. SVD Plane Fit\n(MAD refinement)", "#7B68EE"),
        ("3. Project \u2192 2D\n(orthonormal basis)", "#2E8B57"),
        ("4. Fixed Rasterisation\n+ Dilate + Close", "#D4A017"),
        ("5. Board Mask Build\n(giant close + subtract)", "#CD5C5C"),
        ("6. CC Labelling\n+ Circ/Kasa Filter", "#20B2AA"),
        ("", "white"),  # empty slot to match height
    ]

    for ax, steps, title in [(ax1, steps_a, "Method A (detect_holes.py)"),
                              (ax2, steps_b, "Method B (pcb_hole_detector.py)")]:
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        ax.set_title(title, fontsize=13, fontweight='bold', pad=10)

        y_positions = np.linspace(8.8, 0.8, len(steps))
        for i, ((text, color), y) in enumerate(zip(steps, y_positions)):
            if not text:
                continue
            box = FancyBboxPatch(
                (1.5, y - 0.45), 7, 0.9,
                boxstyle="round,pad=0.15",
                facecolor=color, alpha=0.25,
                edgecolor=color, linewidth=1.5
            )
            ax.add_patch(box)
            ax.text(5, y, text, ha='center', va='center',
                    fontsize=9, fontweight='bold')
            # Draw arrow to next step
            if i < len(steps) - 1 and steps[i + 1][0]:
                ax.annotate(
                    '', xy=(5, y_positions[i + 1] + 0.5),
                    xytext=(5, y - 0.5),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1.5)
                )

    fig.tight_layout()
    fig.savefig(f'{OUT}/01_pipeline_comparison.png', dpi=DPI, bbox_inches='tight')
    plt.close()

    # ═══════════════════════════════════════════════════════════
    #  FIGURE 2: Plane Fitting Comparison
    # ═══════════════════════════════════════════════════════════
    print("Generating Figure 02: Plane fitting comparison...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Left: RANSAC illustration ---
    ax = axes[0]
    np.random.seed(42)
    x = np.random.uniform(0, 10, 200)
    y = 0.5 * x + 1 + np.random.normal(0, 0.3, 200)
    outliers_x = np.random.uniform(0, 10, 30)
    outliers_y = np.random.uniform(0, 8, 30)
    ax.scatter(x, y, c='steelblue', s=15, alpha=0.6, label='Inliers')
    ax.scatter(outliers_x, outliers_y, c='red', s=15, alpha=0.6, label='Outliers')
    xx = np.linspace(0, 10, 100)
    ax.plot(xx, 0.5 * xx + 1, 'g-', lw=2, label='RANSAC fit')
    ax.fill_between(xx, 0.5 * xx + 1 - 0.6, 0.5 * xx + 1 + 0.6,
                    alpha=0.1, color='green', label='\u03b5 = 0.3 mm band')
    ax.set_title('Method A: RANSAC Plane Fitting', fontweight='bold')
    ax.set_xlabel('Coordinate 1')
    ax.set_ylabel('Dependent axis')
    ax.legend(fontsize=8)
    ax.set_xlim(0, 10)
    ax.set_ylim(-1, 8)

    # --- Right: SVD + MAD illustration ---
    ax = axes[1]
    # Simulate distance histogram: main peak (inliers) + outlier tail
    d_inliers = np.random.normal(0, 0.08, 500)
    d_outliers = np.random.normal(2, 0.5, 30)
    d = np.concatenate([d_inliers, d_outliers])
    ax.hist(d, bins=80, color='steelblue', alpha=0.7, density=True,
            label='Distance histogram')
    mode = 0.0
    mad = np.median(np.abs(d_inliers - mode)) * 1.4826
    band = 3 * mad
    ax.axvline(mode, color='green', lw=2, ls='-', label=f'Mode = {mode:.2f}')
    ax.axvline(mode - band, color='red', lw=1.5, ls='--',
               label=f'\u00b13\u03c3_MAD = \u00b1{band:.3f}')
    ax.axvline(mode + band, color='red', lw=1.5, ls='--')
    ax.axvspan(mode - band, mode + band, alpha=0.1, color='green')
    ax.set_title('Method B: SVD + MAD Band Selection', fontweight='bold')
    ax.set_xlabel('Signed distance to plane (mm)')
    ax.set_ylabel('Density')
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(f'{OUT}/02_plane_fitting.png', dpi=DPI, bbox_inches='tight')
    plt.close()

    # ═══════════════════════════════════════════════════════════
    #  FIGURE 3: Morphology Strategy Comparison
    # ═══════════════════════════════════════════════════════════
    print("Generating Figure 03: Morphology comparison...")
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # Create a synthetic sparse surface image with one circular hole
    img = np.zeros((200, 200), dtype=np.uint8)
    rng = np.random.RandomState(42)
    for _ in range(8000):
        r, c = rng.randint(10, 190), rng.randint(10, 190)
        dist = np.sqrt((r - 100)**2 + (c - 100)**2)
        if dist > 25:  # hole of radius 25 px at centre
            img[r, c] = 255

    # --- Top row: Method A pipeline ---
    axes[0, 0].imshow(img, cmap='gray')
    axes[0, 0].set_title('Step 1: Raw Occupancy', fontsize=9)

    k_a = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    closed_a = cv2.morphologyEx(img, cv2.MORPH_CLOSE, k_a)
    axes[0, 1].imshow(closed_a, cmap='gray')
    axes[0, 1].set_title('Step 2: Morph Close (r=4)', fontsize=9)

    inv_a = cv2.bitwise_not(closed_a)
    axes[0, 2].imshow(inv_a, cmap='gray')
    axes[0, 2].set_title('Step 3: Inverted', fontsize=9)

    # Flood fill from border
    flood_a = inv_a.copy()
    mask_ff = np.zeros((202, 202), dtype=np.uint8)
    cv2.floodFill(flood_a, mask_ff, (0, 0), 128)
    for x_border in range(200):
        if flood_a[0, x_border] == 255:
            cv2.floodFill(flood_a, mask_ff, (x_border, 0), 128)
        if flood_a[199, x_border] == 255:
            cv2.floodFill(flood_a, mask_ff, (x_border, 199), 128)
    for y_border in range(200):
        if flood_a[y_border, 0] == 255:
            cv2.floodFill(flood_a, mask_ff, (0, y_border), 128)
        if flood_a[y_border, 199] == 255:
            cv2.floodFill(flood_a, mask_ff, (199, y_border), 128)
    interior_a = (flood_a == 255).astype(np.uint8) * 255
    axes[0, 3].imshow(interior_a, cmap='gray')
    axes[0, 3].set_title('Step 4: Interior (flood fill)', fontsize=9)

    axes[0, 0].set_ylabel('Method A\n(bottom-up)', fontsize=11, fontweight='bold')

    # --- Bottom row: Method B pipeline ---
    dilated = cv2.dilate(img, np.ones((3, 3), np.uint8), iterations=1)
    occ_b = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    axes[1, 0].imshow(occ_b, cmap='gray')
    axes[1, 0].set_title('Step 1: Dilated + Closed', fontsize=9)

    board = cv2.morphologyEx(occ_b, cv2.MORPH_CLOSE, np.ones((71, 71), np.uint8))
    axes[1, 1].imshow(board, cmap='gray')
    axes[1, 1].set_title('Step 2: Board Mask (k=71)', fontsize=9)

    holes_b = ((board > 0) & (occ_b == 0)).astype(np.uint8) * 255
    axes[1, 2].imshow(holes_b, cmap='gray')
    axes[1, 2].set_title('Step 3: Board \u2212 Surface', fontsize=9)

    holes_b2 = cv2.morphologyEx(holes_b, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    axes[1, 3].imshow(holes_b2, cmap='gray')
    axes[1, 3].set_title('Step 4: Open (clean)', fontsize=9)

    axes[1, 0].set_ylabel('Method B\n(top-down)', fontsize=11, fontweight='bold')

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle('Hole Isolation Strategy Comparison', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(f'{OUT}/03_morphology_comparison.png', dpi=DPI, bbox_inches='tight')
    plt.close()

    # ═══════════════════════════════════════════════════════════
    #  FIGURE 4: Shape Metrics Comparison
    # ═══════════════════════════════════════════════════════════
    print("Generating Figure 04: Shape metrics comparison...")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    files = ['pcb_crop1\n(flat CAD)', 'test_crop\n(sensor 36\u00b0)',
             'pcb_crop_ptcld\n(sensor 36\u00b0)']
    # Mean values from actual detection runs
    sol_a = [0.965, 0.896, 0.909]   # Method A solidity
    ar_a = [0.990, 0.877, 0.876]    # Method A axis ratio
    circ_a = [0.948, 0.423, 0.478]  # Method A circularity (raw grid)
    circ_b = [0.800, 0.714, 0.763]  # Method B circularity (pre-dilated)

    x = np.arange(3)
    w = 0.35

    # Solidity (Method A only)
    ax = axes[0]
    ax.bar(x - w / 2, sol_a, w, color='steelblue', label='Method A', alpha=0.8)
    ax.axhline(0.80, color='red', ls='--', lw=1, label='Threshold (0.80)')
    ax.set_ylabel('Solidity')
    ax.set_title('Solidity (Method A only)', fontweight='bold', fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(files, fontsize=8)
    ax.legend(fontsize=8)
    ax.set_ylim(0.5, 1.05)

    # Axis Ratio (Method A only)
    ax = axes[1]
    ax.bar(x - w / 2, ar_a, w, color='steelblue', label='Method A', alpha=0.8)
    ax.axhline(0.60, color='red', ls='--', lw=1, label='Threshold (0.60)')
    ax.set_ylabel('Axis Ratio')
    ax.set_title('Axis Ratio (Method A only)', fontweight='bold', fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(files, fontsize=8)
    ax.legend(fontsize=8)
    ax.set_ylim(0.4, 1.05)

    # Circularity (Both Methods)
    ax = axes[2]
    ax.bar(x - w / 2, circ_a, w, color='steelblue', label='Method A', alpha=0.8)
    ax.bar(x + w / 2, circ_b, w, color='darkorange', label='Method B', alpha=0.8)
    ax.axhline(0.45, color='red', ls='--', lw=1, label='B threshold (0.45)')
    ax.axhline(0.70, color='darkred', ls=':', lw=1, label='Standard threshold')
    ax.set_ylabel('Circularity')
    ax.set_title('Circularity (Both Methods)', fontweight='bold', fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(files, fontsize=8)
    ax.legend(fontsize=7)
    ax.set_ylim(0.0, 1.05)

    fig.tight_layout()
    fig.savefig(f'{OUT}/04_shape_metrics.png', dpi=DPI, bbox_inches='tight')
    plt.close()

    # ═══════════════════════════════════════════════════════════
    #  FIGURE 5: Diameter Comparison
    # ═══════════════════════════════════════════════════════════
    print("Generating Figure 05: Diameter comparison...")
    fig, ax = plt.subplots(figsize=(10, 5))

    files_short = ['pcb_crop1', 'test_crop', 'pcb_crop_ptcld']
    ma_means = [1.297, 1.128, 1.141]
    mb_means = [0.882, 0.865, 0.887]
    ma_mins = [1.238, 1.043, 1.060]
    ma_maxs = [1.347, 1.186, 1.180]
    mb_mins = [0.813, 0.802, 0.811]
    mb_maxs = [0.950, 0.904, 0.915]

    x = np.arange(3)
    w = 0.3
    ax.bar(x - w / 2, ma_means, w, color='steelblue', alpha=0.85,
           label='Method A (compensated)')
    ax.bar(x + w / 2, mb_means, w, color='darkorange', alpha=0.85,
           label='Method B (Kasa fit)')

    # Error bars showing min-max range
    ax.errorbar(x - w / 2, ma_means,
                yerr=[np.array(ma_means) - np.array(ma_mins),
                      np.array(ma_maxs) - np.array(ma_means)],
                fmt='none', ecolor='navy', capsize=4)
    ax.errorbar(x + w / 2, mb_means,
                yerr=[np.array(mb_means) - np.array(mb_mins),
                      np.array(mb_maxs) - np.array(mb_means)],
                fmt='none', ecolor='darkred', capsize=4)

    # Reference line: standard 50-mil via
    ax.axhline(1.27, color='green', ls='--', lw=1.5, alpha=0.7,
               label='50 mil via (1.27 mm)')

    ax.set_xticks(x)
    ax.set_xticklabels(files_short)
    ax.set_ylabel('Diameter (mm)')
    ax.set_title('Measured Hole Diameters: Method A vs Method B',
                 fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_ylim(0.5, 1.6)

    # Annotate differences
    for i in range(3):
        diff = ma_means[i] - mb_means[i]
        ax.annotate(f'\u0394={diff:.2f}mm',
                    xy=(i, max(ma_maxs[i], mb_maxs[i]) + 0.03),
                    ha='center', fontsize=8, color='gray')

    fig.tight_layout()
    fig.savefig(f'{OUT}/05_diameter_comparison.png', dpi=DPI, bbox_inches='tight')
    plt.close()

    # ═══════════════════════════════════════════════════════════
    #  FIGURE 6: Kasa Circle Fit Bias Illustration
    # ═══════════════════════════════════════════════════════════
    print("Generating Figure 06: Kasa circle fit bias...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    theta = np.linspace(0, 2 * np.pi, 50)
    r_true = 1.0
    cx_true, cy_true = 5.0, 5.0

    # --- Left: Clean contour ---
    x_clean = cx_true + r_true * np.cos(theta)
    y_clean = cy_true + r_true * np.sin(theta)

    ax = axes[0]
    ax.plot(x_clean, y_clean, 'bo', ms=4, label='Contour points')

    # Kasa fit on clean data
    A_clean = np.c_[2 * x_clean, 2 * y_clean, np.ones_like(x_clean)]
    b_clean = x_clean**2 + y_clean**2
    sol_clean, *_ = np.linalg.lstsq(A_clean, b_clean, rcond=None)
    cx_k_clean, cy_k_clean = sol_clean[0], sol_clean[1]
    r_k_clean = np.sqrt(max(sol_clean[2] + cx_k_clean**2 + cy_k_clean**2, 0))

    circle_clean = plt.Circle(
        (cx_k_clean, cy_k_clean), r_k_clean,
        fill=False, color='green', lw=2, ls='--',
        label=f'Kasa fit r={r_k_clean:.3f}'
    )
    ax.add_patch(circle_clean)
    ax.plot(cx_k_clean, cy_k_clean, 'g+', ms=15, mew=2)
    ax.set_aspect('equal')
    ax.set_title('Clean contour \u2192 accurate Kasa fit',
                 fontweight='bold', fontsize=10)
    ax.legend(fontsize=8)
    ax.set_xlim(3.5, 6.5)
    ax.set_ylim(3.5, 6.5)

    # --- Right: Noisy contour (sensor-like) ---
    rng = np.random.RandomState(7)
    r_noisy = r_true + rng.uniform(-0.15, 0.15, len(theta))
    x_noisy = cx_true + r_noisy * np.cos(theta)
    y_noisy = cy_true + r_noisy * np.sin(theta)

    # Kasa fit on noisy data
    A_noisy = np.c_[2 * x_noisy, 2 * y_noisy, np.ones_like(x_noisy)]
    b_noisy = x_noisy**2 + y_noisy**2
    sol_noisy, *_ = np.linalg.lstsq(A_noisy, b_noisy, rcond=None)
    cx_k, cy_k = sol_noisy[0], sol_noisy[1]
    r_k = np.sqrt(max(sol_noisy[2] + cx_k**2 + cy_k**2, 0))

    ax = axes[1]
    ax.plot(x_noisy, y_noisy, 'bo', ms=4, label='Noisy contour')
    circle_kasa = plt.Circle(
        (cx_k, cy_k), r_k, fill=False, color='red', lw=2, ls='--',
        label=f'Kasa fit r={r_k:.3f} (biased)'
    )
    circle_true = plt.Circle(
        (cx_true, cy_true), r_true, fill=False, color='green', lw=1.5, ls=':',
        label=f'True r={r_true:.3f}'
    )
    ax.add_patch(circle_true)
    ax.add_patch(circle_kasa)
    ax.plot(cx_k, cy_k, 'r+', ms=15, mew=2)
    ax.set_aspect('equal')
    ax.set_title('Noisy contour \u2192 Kasa underestimates radius',
                 fontweight='bold', fontsize=10)
    ax.legend(fontsize=8)
    ax.set_xlim(3.5, 6.5)
    ax.set_ylim(3.5, 6.5)

    fig.tight_layout()
    fig.savefig(f'{OUT}/06_kasa_fit.png', dpi=DPI, bbox_inches='tight')
    plt.close()

    # ═══════════════════════════════════════════════════════════
    #  FIGURE 7: Production Enhancement Roadmap
    # ═══════════════════════════════════════════════════════════
    print("Generating Figure 07: Production roadmap...")
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    categories = [
        ("SPEED", 8.5, "#4A90D9", [
            "GPU rasterisation\n(CUDA)",
            "Open3D KDTree\nfor density",
            "Parallel batch\nprocessing",
            "C++ morph\noperations",
        ]),
        ("ACCURACY", 5.5, "#2E8B57", [
            "Taubin circle fit\n(unbiased)",
            "Sub-pixel edge\nrefinement",
            "Multi-scale\ndetection",
            "Point-density\nnormalisation",
        ]),
        ("ROBUSTNESS", 2.5, "#CD5C5C", [
            "Multi-plane\nsegmentation",
            "Adaptive MAD\n+ RANSAC hybrid",
            "Deep learning\nhole classifier",
            "Warped/curved\nboard support",
        ]),
    ]

    for title, y, color, items in categories:
        # Category label box on the left
        box = FancyBboxPatch(
            (0.5, y - 0.9), 3.5, 1.8,
            boxstyle="round,pad=0.2",
            facecolor=color, alpha=0.2,
            edgecolor=color, linewidth=2
        )
        ax.add_patch(box)
        ax.text(2.25, y + 0.4, title, ha='center', va='center',
                fontsize=12, fontweight='bold', color=color)

        # Item boxes on the right
        for i, item in enumerate(items):
            xi = 5.5 + i * 2.2
            box2 = FancyBboxPatch(
                (xi - 0.9, y - 0.55), 1.8, 1.1,
                boxstyle="round,pad=0.1",
                facecolor=color, alpha=0.1,
                edgecolor=color, linewidth=1
            )
            ax.add_patch(box2)
            ax.text(xi, y, item, ha='center', va='center',
                    fontsize=7.5, wrap=True)
            # Arrow from category to item
            ax.annotate(
                '', xy=(xi - 0.9, y), xytext=(4.0, y),
                arrowprops=dict(arrowstyle='->', color=color, lw=1, alpha=0.4)
            )

    ax.set_title('Production Enhancement Roadmap',
                 fontsize=14, fontweight='bold', pad=15)
    fig.tight_layout()
    fig.savefig(f'{OUT}/07_roadmap.png', dpi=DPI, bbox_inches='tight')
    plt.close()

    # ═══════════════════════════════════════════════════════════
    #  FIGURE 8: Surface Band Comparison
    # ═══════════════════════════════════════════════════════════
    print("Generating Figure 08: Surface band comparison...")
    fig, ax = plt.subplots(figsize=(10, 4))

    files_s = ['pcb_crop1\n(flat CAD)',
               'test_crop\n(sensor 36\u00b0)',
               'pcb_crop_ptcld\n(sensor 36\u00b0)']
    band_a = [1.0, 1.0, 1.0]          # Method A: fixed 1.0 mm
    band_b = [0.060, 0.165, 0.166]     # Method B: adaptive MAD-based

    x = np.arange(3)
    w = 0.3
    ax.bar(x - w / 2, band_a, w, color='steelblue', alpha=0.8,
           label='Method A (fixed)')
    ax.bar(x + w / 2, band_b, w, color='darkorange', alpha=0.8,
           label='Method B (adaptive MAD)')

    ax.set_xticks(x)
    ax.set_xticklabels(files_s, fontsize=9)
    ax.set_ylabel('Surface Band (mm)')
    ax.set_title('Surface Thickness Band Comparison', fontweight='bold')
    ax.legend()
    ax.set_yscale('log')
    ax.set_ylim(0.01, 2)

    fig.tight_layout()
    fig.savefig(f'{OUT}/08_surface_band.png', dpi=DPI, bbox_inches='tight')
    plt.close()

    # ═══════════════════════════════════════════════════════════
    #  FIGURES 9-12: Copied from detection runs
    # ═══════════════════════════════════════════════════════════
    print()
    print("Figures 09-12 are real detection outputs, not generated here.")
    print("They are copied from the Method A / Method B output directories:")
    print("  09_method_a_result.png  <- Method A: 08_final_result.png (test_crop)")
    print("  10_method_b_result.png  <- Method B: test_crop_04_detected_overlay.png")
    print("  11_method_a_montage.png <- Method A: 10_montage.png (test_crop)")
    print("  12_closeups.png         <- Method A: 09_closeups.png (pcb_crop1)")
    print()
    print(f"All synthetic figures saved to: {OUT}/")
    print("Done.")


if __name__ == '__main__':
    main()
