# PCB Hole Detection: Comparative Technical Analysis

## Method A (`detect_holes.py`) vs Method B (`pcb_hole_detector.py`)

---

## 1. Executive Summary

Both methods solve the same problem — detecting through-holes in PCB point clouds — and both achieve correct hole counts on all three test files (21, 6, and 5 holes respectively). However, they differ substantially in their mathematical approach at nearly every pipeline stage, producing different diameter measurements, different filtering strategies, and different robustness characteristics. Method A consistently reports larger diameters (mean ≈ 1.1–1.3 mm) than Method B (mean ≈ 0.86–0.88 mm) due to morphological compensation in A versus raw Kasa circle fitting in B.

---

## 2. Pipeline Architecture Comparison

### Method A — 7-stage pipeline (527 lines)

1. Load XYZ
2. RANSAC plane fitting (sklearn, multi-axis)
3. Project to 2D via orthonormal basis
4. Adaptive binary rasterisation
5. Morphological closing + flood-fill
6. Connected-component filtering (solidity + axis ratio)
7. Back-project to 3D

### Method B — 5-stage pipeline (267 lines)

1. Load XYZ
2. SVD plane fitting with MAD-based outlier rejection
3. Project to 2D via orthonormal basis
4. Rasterise + dilate + close → build board mask → subtract
5. Connected-component filtering (circularity + Kasa circle fit)

Both share the same conceptual flow (fit plane → project 2D → rasterise → morphology → filter), but every stage has different implementation details.

---

## 3. Plane Fitting — The Foundation

### 3.1 Method A: RANSAC via Linear Regression

Method A parametrises the plane as a linear function of two coordinates:

    Z_dep = a · X_indep1 + b · X_indep2 + c

It tries all three choices of dependent axis (X, Y, Z) and picks whichever yields the most inliers. This is an sklearn `RANSACRegressor` with:

- Residual threshold: 0.3 mm
- Max trials: 3000
- Min samples: 3 (minimum to define a plane)
- Subsample: 100K points for speed

The normal is recovered as **n** = (−a, −b, 1) (normalised), with the sign depending on which axis was dependent. After RANSAC, all points within 1.0 mm of the plane are kept as surface inliers.

**Mathematical model:** RANSAC is a consensus-based estimator. At each iteration it samples 3 points, fits a plane through them via least-squares (solving a 3×2 linear system), and counts how many of the remaining points lie within ε = 0.3 mm. After T iterations, the model with the most inliers is refined by least-squares over all inliers.

**Strengths:** Very robust to arbitrary outlier fractions (works even if 50%+ of the data is non-planar). The multi-axis strategy avoids the degeneracy that occurs when the plane is nearly perpendicular to the chosen dependent axis.

**Weaknesses:** Slower (3000 trials × 3 axes = 9000 fits). The 1.0 mm thickness band is generous and may include non-surface points on thick boards.

### 3.2 Method B: SVD with MAD-Based Inlier Selection

Method B uses a two-pass SVD approach:

**Pass 1 — Coarse fit:**
1. Subtract the median of all points (robust centering).
2. Subsample 50K points, compute SVD of the centered matrix.
3. The last right-singular vector of SVD gives the normal direction (the direction of minimum variance).
4. Compute signed distances d = (p − median) · n.
5. Build a 256-bin histogram of d, find the mode (peak bin).
6. Compute MAD (Median Absolute Deviation) from the mode.
7. Define inlier band as: |d − mode| ≤ max(0.06, 3.0 × 1.4826 × MAD).

**Pass 2 — Refined fit:**
8. Keep only inlier points from pass 1.
9. Re-center on the inlier mean, re-compute SVD for a refined normal.
10. Recompute the distance histogram, mode, MAD, and final band.

The factor 1.4826 converts MAD to an estimate of the standard deviation under a Gaussian assumption (MAD × 1.4826 ≈ σ). The 3× multiplier gives a ≈ 99.7% confidence band.

**Mathematical model:** SVD of a centered point matrix X = UΣVᵀ finds the principal directions of variance. The last column of V corresponds to the smallest singular value — this is the direction of minimum spread, i.e., the plane normal. This is equivalent to PCA (Principal Component Analysis) applied to the covariance matrix XᵀX.

**Strengths:** Elegant statistical framework. The MAD-based band is adaptive — it automatically narrows for clean CAD data (MAD → 0, band → 0.06 mm floor) and widens for noisy sensor data. Two-pass refinement improves accuracy.

**Weaknesses:** SVD minimises total variance, so it's sensitive to outlier clusters (unlike RANSAC which ignores them). The 0.06 mm floor may be too tight for rough surfaces. The histogram-based mode estimation depends on bin count (256 bins = quantisation of ≈ 0.06 mm for a 15 mm range).

### 3.3 Comparison: Resulting Surface Bands

| File | Method A band | Method B band |
|------|--------------|--------------|
| pcb_crop1 (flat CAD) | 1.000 mm | 0.060 mm |
| test_crop (tilted sensor) | 1.000 mm | 0.165 mm |
| pcb_crop_ptcld (tilted sensor) | 1.000 mm | 0.166 mm |

Method A uses a fixed 1.0 mm band (keeps 100% of points in all cases). Method B adapts: 0.06 mm for the nearly-perfect CAD data, ~0.17 mm for sensor data. Method B's tighter band excludes more off-surface points, which could help with multi-layer boards but might lose legitimate surface points near components.

---

## 4. 2D Projection

Both methods construct an orthonormal basis {**û**, **v̂**} lying in the fitted plane using cross products, but they choose different arbitrary vectors to break the symmetry.

### Method A:

    if |n_z| < 0.9: arb = (0,0,1)
    else:            arb = (1,0,0)
    û = n × arb (normalised)
    v̂ = n × û

### Method B:

    if |n_x| < 0.9: arb = (1,0,0)
    else:            arb = (0,1,0)
    û = n × arb (normalised)
    v̂ = n × û

The mathematical operation is identical — Gram-Schmidt orthogonalisation — but the seed vector choice means the resulting UV axes will be rotated relative to each other. This has no effect on hole detection (circles are rotationally invariant), but the UV coordinates reported in CSV files will differ between methods.

Method B also applies explicit normal orientation logic (ensuring n_z > 0 or n_x > 0), while Method A relies on the RANSAC parametrisation sign convention.

---

## 5. Rasterisation — Where the Methods Diverge Most

### 5.1 Method A: Adaptive Resolution + Binary Occupancy

Method A computes the average point spacing:

    s = 1 / √(density)  where density = N_points / (u_span × v_span)

and sets the grid resolution as:

    r = min(s/2, d_min/20)

This ensures both Nyquist sampling of the point spacing and at least 20 pixels across the smallest hole. For the test data this yields r ≈ 0.019–0.040 mm/px depending on point density. Grid sizes are 1200–2000 px per side.

Each point maps to exactly one pixel: `grid[row, col] = 255`. No dilation, no smoothing — pure binary presence.

### 5.2 Method B: Fixed Resolution + Dilate + Close

Method B uses a fixed, user-specified resolution (default 0.04 mm/px, passed via `--grid-mm`). After rasterisation, it applies:

1. **Dilation** with a 3×3 square kernel (1 iteration): every occupied pixel expands by 1 px in all directions.
2. **Morphological closing** with a 5×5 square kernel (1 iteration).

This is a different strategy: instead of adaptive closing later, Method B aggressively builds up the surface at rasterisation time, making each point occupy a 3×3 neighbourhood.

**Key difference:** Method A's rasterisation produces a sparse image (20% occupancy), then adaptively closes it later. Method B's rasterisation produces a denser image immediately (dilate + close), then builds an even larger board mask to subtract from.

### 5.3 Resolution Impact

| File | Method A resolution | Method B resolution | Method A grid | Method B grid |
|------|-------------------|-------------------|--------------|--------------|
| pcb_crop1 | 0.0400 mm/px | 0.0400 mm/px | 1352×1265 | ~1301×1214 |
| test_crop | 0.0188 mm/px | 0.0400 mm/px | 1278×1955 | ~602×920 |
| pcb_crop_ptcld | 0.0191 mm/px | 0.0400 mm/px | 1284×1294 | ~631×610 |

For the sensor data files, Method A uses approximately 2× finer resolution (adapting to the higher point density), giving ~4× more pixels — substantially more spatial detail for circularity and diameter estimation.

---

## 6. Hole Isolation Strategy

This is the most architecturally different stage between the two methods.

### 6.1 Method A: Single Morphological Closing + Flood Fill

**Step 1 — Adaptive closing:**
Computes kernel radius from point spacing:

    close_r = max(2, ⌈avg_spacing_px × 1.8⌉)

Uses an elliptical structuring element. Closing = Dilate(B) then Erode(B). This bridges inter-point gaps without filling holes.

**Step 2 — Flood fill:**
Inverts the closed image (voids → white, surface → black). Flood-fills from every border pixel with sentinel value 128. Remaining 255-valued pixels are interior voids (holes).

This is a bottom-up approach: start from the raw surface, seal small gaps, then find what's left inside.

### 6.2 Method B: Board Mask Construction + Subtraction

**Step 1 — Build board mask:**
Applies an aggressive morphological closing with a large kernel:

    k = int(round(max_diam * 2.2 / res))   # e.g., 138 px for 2.5mm at 0.04mm/px
    k = max(9, k | 1)                        # ensure odd, ≥ 9

This is ~138×138 px — enormous. It closes every hole in the image, producing a solid board silhouette. Then morphological opening (5×5) cleans edges, connected-component analysis finds the largest region (the board), and `binary_fill_holes` ensures it's completely solid.

**Step 2 — Subtraction:**
    holes = (board > 0) AND (occupancy == 0)

Where the board mask says "board should be here" but the occupancy says "no points" — that's a hole.

**Step 3 — Open (3×3):**
Removes small noise speckles.

This is a top-down approach: first establish where the board is (filling everything including holes), then subtract the actual surface to reveal holes.

### 6.3 Comparison

The board-mask strategy (Method B) is conceptually elegant: it cleanly separates exterior (no board) from interior (board with holes). It handles irregular board outlines well because it explicitly finds the board contour.

The flood-fill strategy (Method A) achieves the same separation but relies on the surface being continuous enough (after closing) that all exterior voids connect to the border. If the board has a large notch or irregular edge, the flood fill correctly reaches it from the border — but if the closing is too aggressive, it might also seal a hole near the edge.

Both approaches work correctly on all test files, but they have different failure modes:
- Method A can fail if a hole is very close to the board edge and the closing bridges it to the exterior.
- Method B can fail if the board mask's huge closing kernel creates artefacts at concave board edges or multi-board scenes.

---

## 7. Shape Filtering — Different Metrics for the Same Goal

### 7.1 Method A: Solidity + Axis Ratio

Method A uses `skimage.regionprops` to compute:

**Solidity** = Area / Convex_Hull_Area
- Perfect circle: S ≈ 1.0
- Star-shaped or concave region: S < 1.0
- Threshold: S ≥ 0.80

**Axis Ratio** = Minor_Axis_Length / Major_Axis_Length
- Perfect circle: AR ≈ 1.0
- Elongated shape: AR → 0
- Threshold: AR ≥ 0.60

These metrics are area-based and geometry-based respectively. Neither depends on perimeter, making them robust to boundary roughness in sensor data.

### 7.2 Method B: Perimeter-Based Circularity + Kasa Circle Fit

Method B uses OpenCV contour analysis:

**Circularity** = 4π × Area / Perimeter²
- Perfect circle: C = 1.0
- Jagged boundary: C << 1.0 (perimeter inflates)
- Threshold: C ≥ 0.45

**Kasa Circle Fit** for diameter measurement:
Given N contour points (xᵢ, yᵢ), Kasa's method solves:

    [2x₁  2y₁  1]   [cx]     [x₁² + y₁²]
    [2x₂  2y₂  1] × [cy]  =  [x₂² + y₂²]
    [  ⋮    ⋮   ⋮]   [ c]     [    ⋮     ]

in the least-squares sense, where the circle has center (cx, cy) and radius r = √(c + cx² + cy²).

The Kasa fit is a linearisation of the circle equation:

    (x − cx)² + (y − cy)² = r²

expanded to:

    x² + y² = 2·cx·x + 2·cy·y + (r² − cx² − cy²)

This is algebraically elegant but has a known bias: it systematically underestimates the radius for noisy contours because it minimises algebraic distance rather than geometric distance.

### 7.3 Impact on Diameter Measurement

This is the single biggest numerical difference between the methods. Method A's diameter comes from the equivalent diameter of the connected component:

    d_raw = √(4·Area/π) × resolution

then adds morphological compensation:

    d_compensated = d_raw + 2 × close_r × resolution

Method B's diameter comes from the Kasa circle fit on the contour pixels.

| File | Method A mean ⌀ | Method B mean ⌀ | Difference |
|------|----------------|----------------|------------|
| pcb_crop1 | 1.297 mm | 0.882 mm | +0.415 mm |
| test_crop | 1.128 mm | 0.865 mm | +0.263 mm |
| pcb_crop_ptcld | 1.141 mm | 0.887 mm | +0.254 mm |

Method A consistently reports ~0.25–0.42 mm larger diameters. This comes from:
1. **Morphological compensation** in Method A adds 2 × 4 × 0.04 = 0.32 mm (pcb_crop1) or 2 × 4 × 0.019 = 0.15 mm (sensor files).
2. **Kasa bias** in Method B slightly underestimates the radius.
3. **Different occupancy images** — Method B's pre-dilation makes the surface thicker around holes, leaving smaller hole regions.

Neither method's diameter can be verified as "ground truth" without CAD reference dimensions, but Method A's compensated diameters (~1.2–1.3 mm) are more consistent with typical PCB via hole sizes (50 mil = 1.27 mm), while Method B's raw diameters (~0.88 mm) are systematically smaller.

---

## 8. Filtering Thresholds and Their Physics

### Method A thresholds:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Min solidity | 0.80 | Convex-hull based — robust to jagged edges |
| Min axis ratio | 0.60 | Rejects elongated voids |
| Min/max diameter | 0.8–2.5 mm | Physical hole size range with margin |
| Border margin | 3 px | Rejects truncated holes at crop edge |

### Method B thresholds:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Min circularity | 0.45 | Perimeter-based — less robust to noise |
| Min area | π × (d_min × 0.28 / res)² | Very generous lower bound |
| Max area | π × (d_max × 0.75 / res)² | Tighter upper bound |
| Min bbox side | max(4, d_min × 0.30 / res) | Rejects tiny fragments |
| Diameter range | 0.45×min to 1.10×max | Applied to Kasa-fit diameter |

Method B's circularity threshold of 0.45 is much looser than the traditional 0.7+ used in most literature. This compensates for the fact that perimeter-based circularity is degraded by sensor noise — the same insight that led Method A to abandon circularity entirely in favour of solidity. On the test data, Method B's sensor-data holes have circularity 0.60–0.84 (comfortably above 0.45), while Method A's circularity for the same holes was 0.28–0.57 (would fail a standard 0.7 threshold, hence the switch to solidity).

The difference is explained by Method B's pre-dilation smoothing the contour, improving perimeter-based circularity somewhat, while Method A operates on a raw binary grid with rougher boundaries.

---

## 9. Hole Centre Localisation

### Method A: Connected-Component Centroid
Uses `skimage.regionprops.centroid` — the area-weighted centre of all pixels in the connected component:

    cx = Σ(col_i) / N_pixels,  cy = Σ(row_i) / N_pixels

This is the geometric centroid, optimal for uniform-density regions.

### Method B: Kasa Circle Centre
Fits a least-squares circle to the contour pixels, then uses the fitted centre (cx, cy). This is mathematically different — it's the centre of the best-fit circle through the boundary, not the area centroid.

For a perfect circle, both give the same answer. For irregular shapes, Kasa's centre is pulled toward the denser part of the boundary, while the area centroid treats all interior pixels equally.

The practical difference in the 3D back-projected centres is small (typically < 0.1 mm), because the holes are nearly circular in both methods.

---

## 10. Coordinate Frames and Back-Projection

Both methods back-project 2D hole centres to 3D using:

    P_3d = plane_centre + u_mm × û + v_mm × v̂ + offset × n̂

Method A omits the normal offset (places centres exactly on the plane), while Method B adds `mode_d × n̂` — the signed distance from the plane centre to the histogram mode. The practical effect is tiny (< 0.002 mm Z difference) since both plane fits converge to nearly the same surface.

---

## 11. Output and Diagnostics

### Method A:
- 10 debug images (3D scatter, 2D projection, raw grid, closed, inverted, interior holes, candidates, final overlay, close-ups, montage)
- CSV with 3D centres, 2D centres, diameter, solidity, axis ratio, circularity
- Timestamped console logging at every step

### Method B:
- 4 images (occupancy, board mask, holes mask, overlay)
- JSON with full metadata (plane parameters, resolution, per-hole data)
- CSV with centres and measurements
- Debug text file with per-component reject/accept reasoning
- Summary JSON across multiple input files

Method B's JSON output is more machine-readable and better suited for automated pipelines. Method A's visual diagnostics are more comprehensive for human debugging.

---

## 12. Performance and Scalability

| Aspect | Method A | Method B |
|--------|----------|----------|
| Dependencies | sklearn, skimage, scipy, cv2, matplotlib | cv2, scipy, numpy |
| Code size | 527 lines | 267 lines |
| Plane fitting | ~7 sec (9000 RANSAC trials) | ~1 sec (2× SVD) |
| Grid resolution | Adaptive (finer for denser data) | Fixed (user parameter) |
| Max grid cap | 6000 px | No cap |
| Batch processing | Single file per run | Multiple files natively |
| Matplotlib dependency | Required (heavy) | Not required |

Method B is significantly faster due to SVD plane fitting (~7× faster than RANSAC) and lower-resolution grids for sensor data. Method A uses approximately 4× more pixels for sensor files, giving better spatial resolution but slower processing.

---

## 13. Robustness Analysis

### Scenario: Clean CAD Data (pcb_crop1)
Both work perfectly. The data is flat (tilt 0°) with uniform point density and clean hole boundaries. All filtering metrics are well above thresholds.

### Scenario: Tilted Sensor Data (test_crop, pcb_crop_ptcld)
Both detect the correct number of holes. However:
- Method B's circularity values (0.60–0.84) are higher than Method A's (0.28–0.57) because Method B's pre-dilation smooths boundaries.
- Method A would fail on these files if using circularity (which is why it switched to solidity). Method B's lower threshold (0.45) and pre-smoothing make circularity viable.

### Scenario: Multi-Layer or Complex Board Geometry
Method B's board-mask approach (largest connected component) assumes a single dominant board. If two separate boards are in the point cloud, only the larger one is kept. Method A's flood-fill approach would detect holes in both boards, but might create false positives at the boundary between them.

### Scenario: Holes Near Board Edge
Method B's board mask has a large morphological kernel (138 px) that may extend the board boundary beyond the actual edge, potentially creating false holes at concave board corners. Method A's flood fill naturally handles this since edge voids connect to the exterior.

---

## 14. Summary of Key Differences

| Aspect | Method A | Method B |
|--------|----------|----------|
| Plane fitting | RANSAC (multi-axis) | SVD + MAD refinement |
| Surface band | Fixed 1.0 mm | Adaptive (0.06–0.17 mm) |
| Resolution | Adaptive (0.019–0.04 mm/px) | Fixed (0.04 mm/px) |
| Surface building | Binary → morph close | Binary → dilate → close |
| Hole isolation | Flood-fill from border | Board mask subtraction |
| Shape filter | Solidity ≥ 0.80, axis ratio ≥ 0.60 | Circularity ≥ 0.45 |
| Diameter method | √(4A/π) + morph compensation | Kasa algebraic circle fit |
| Mean ⌀ (pcb_crop1) | 1.297 mm | 0.882 mm |
| Centre method | Area centroid | Circle-fit centre |
| Normal orientation | Implicit from RANSAC | Explicit sign logic |
| Debug output | 10 images + CSV | 4 images + JSON + CSV |
| Speed | ~18 sec | ~5 sec |
| Batch support | No | Yes |

Both methods are correct implementations that achieve 100% accuracy on the given test files. Method A prioritises measurement accuracy (diameter compensation, adaptive resolution, area-based metrics) and visual diagnostics. Method B prioritises computational efficiency (SVD, fixed resolution, lighter dependencies) and machine-readable output. The choice between them depends on whether the downstream task values accurate diameters (choose A) or fast batch processing (choose B).
