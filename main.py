#!/usr/bin/env python3

import numpy as np

from computeExtrinsicsModule import computeExtrinsics
from auxFunctions import getRvecTvec

calibDataDir = "./private/cameras_intrinsics_parameters/calib_data.txt"
imgRoot = "./private/images/"

# Specifying the transformation that transforms a point in the world to one in a reference frame, in order to get
# results in the latter
E = np.eye(4)

# Arguments are passed as Rodrigues rotation vector and translation vector
rvec, tvec = getRvecTvec(E)

# Call the function to compute the extrinsic calibration
transformRefFrame2Cameras, transformBetweenCameras = computeExtrinsics(imgRoot, calibDataDir, rvec, tvec)

## Demo to access returned structures

# Accessing transform that takes a point in the reference frame to the reference frame of camera 1
print(transformRefFrame2Cameras[0]["E"])
print(transformRefFrame2Cameras[0]["rvec"])
print(transformRefFrame2Cameras[0]["tvec"])

# Accessing transform that takes a point in the reference frame of camera 2 to camera 3
print(transformBetweenCameras[1][2]["E"])
print(transformBetweenCameras[1][2]["rvec"])
print(transformBetweenCameras[1][2]["tvec"])

# Accessing transform that takes a point in the reference frame of camera 3 to camera 1
print(transformBetweenCameras[2][0]["E"])
print(transformBetweenCameras[2][0]["rvec"])
print(transformBetweenCameras[2][0]["tvec"])

