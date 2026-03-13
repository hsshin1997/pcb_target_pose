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