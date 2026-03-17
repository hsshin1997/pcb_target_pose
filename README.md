# PCB Target Pose Pipeline

## Steps for hole detection
1. Build plane using RANSAC
2. Build Occupancy grid to detect for holes
3. Fit circle and get hole center pose

## Steps for target pose detection
### Method 1. Transformation
Assume PCB lay flat on the surface and find relative poses to each other

### Method 2. Template matching 
Find center pose for each target and run template matching to find all the poses



## How to run

find_holes_precise.py

```
python3 src/detect_holes.py ptcld/pipeline_test_part_ptcld_cropped.xyz --min_diam 1.0 --max_diam 2.0
```


### Working version
pcb_hole_detector.py 
```
python3 src/pcb_hole_detector.py ptcld/test_crop.xyz --out-dir hole_detection_output
```
detect_holes.py
```
python3 src/detect_holes.py ptcld/pcb_crop_ptcld.xyz ./hole_detection_output2/pcb_crop_ptcld
```