import cv2 as cv
from cv2 import aruco
import numpy as np
import matplotlib.pyplot as plt
import glob

from auxFunctions import *

def computeExtrinsics(imgRoot, calibDataDir, rvec=[], tvec=[], debug=False):

    # Parameters rvec and tvec should give the a transformation from the world to an arbitrary reference frame
    if len(rvec) == 0:
        rvec = np.zeros(3).reshape(3,1)
    if len(tvec) == 0:
        tvec = np.zeros(3).reshape(3,1)
    tWorld2Ref = {"rvec": rvec, "tvec": tvec, "E": getRigBodyTransfMatrix(rvec, tvec)}

    ## Load camera calibration
    fs = cv.FileStorage(calibDataDir, cv.FILE_STORAGE_READ)

    K = fs.getNode("intrinsics").mat()
    distCoeffs = fs.getNode("distortion").mat()
    imgSz = fs.getNode("image_size").mat()
    imgSz = tuple(imgSz.reshape(1,2)[0].astype(int))

    fs.release()

    ## Define aruco cube coordinates
    # Information given/known:
    # Aruco markers dimensions = 0.69m X 0.69m
    # Edge width = 0.061
    # height of top face of the Aruco cube (to the floor) = 0.88
    d = 0.69/2
    e = 0.061
    h = 0.88

    # Transformation that converts a point in the cube reference frame to the world
    tCube2World = {"rvec": np.zeros(3).reshape(3,1), "tvec": np.array( [[0], [0], [h-d-e]])}
    tCube2World["E"] = getRigBodyTransfMatrix(tCube2World["rvec"], tCube2World["tvec"])

    # Aruco reference frame is in the center of the cube, with:
    # Z pointing upwards
    # X pointing to marker 1
    # Y pointing to marker 4

    # Aruco corner order is clockwise.
    # Aruco marker points in the world's reference frame
    arucoMk0 = np.array( [[d, d, h],
                     [d, -d, h],
                     [-d, -d, h],
                     [-d, d, h]])
    arucoMk1 = np.array( [[d+e, d, h-2*d-e],
                     [d+e, -d, h-2*d-e],
                     [d+e, -d, h-e],
                     [d+e, d, h-e]])
    arucoMk2 = np.array( [[d, -d-e, h-e],
                     [d, -d-e, h-2*d-e],
                     [-d, -d-e, h-2*d-e],
                     [-d, -d-e, h-e]])
    arucoMk3 = np.array( [[],
                     [],
                     [],
                     []])
    arucoMk4 = np.array( [[d, d+e, h-2*d-e],
                     [d, d+e, h-e],
                     [-d, d+e, h-e],
                     [-d, d+e, h-2*d-e]])
    arucoMarkers = {0:arucoMk0, 1:arucoMk1, 2:arucoMk2, 3:arucoMk3, 4:arucoMk4}


    ## Apply given transformation to work the given reference frame
    for key in arucoMarkers.keys():
        if arucoMarkers[key].size > 0:
            arucoMarkers[key] = applyRigBodyTransformation(arucoMarkers[key], tWorld2Ref["rvec"], tWorld2Ref["tvec"])
    tCube2Ref = {"E": tWorld2Ref["E"].dot(tCube2World["E"]) }
    tCube2Ref["rvec"], tCube2Ref["tvec"] = getRvecTvec( tCube2Ref["E"] )


    ## Aruco library config
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_1000)
    parameters = aruco.DetectorParameters_create()

    # Plot aruco marker
    if debug:
        sampleNum = 0
        sampleMarker = aruco.drawMarker(aruco_dict, sampleNum, 500, None)
        cv.imshow("Sample aruco marker nr {}".format(sampleNum), sampleMarker)
        cv.imwrite("./img/sample-aruco-marker-{}.png".format(sampleNum), sampleMarker)
        cv.waitKey()

    # Configuration tuning of the aruco library can yield:
    mode = 2
    if mode == 1:
        # Good-ish results
        parameters.adaptiveThreshWinSizeMin = 5
        parameters.adaptiveThreshWinSizeMax = 25
        parameters.adaptiveThreshWinSizeStep = 5

        parameters.minMarkerPerimeterRate = 0.2
        parameters.maxMarkerPerimeterRate = 4.0

        parameters.minMarkerDistanceRate = 0.01
    elif mode == 2:
        # Great results
        parameters.cornerRefinementMethod = aruco.CORNER_REFINE_APRILTAG
    else:
        # compared to standard values
        pass

    ## Compute camera extrinsic calibration

    tRef2Cam = {}

    # For each camera
    for fname in glob.glob(imgRoot + "*"):
        print(fname)
        # camNum indexing starts from 0
        camNum = int(fname[-5])-1
        img = cv.imread(fname)
        #cv.imshow("img{}".format(camNum+1), img)

        # Detect aruco markers
        imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = aruco.detectMarkers(imgGray, aruco_dict, parameters=parameters, cameraMatrix=K,
                                                              distCoeff=distCoeffs)
        print("Detected {} markers.".format(len(ids)))

        # Refine corners. Not good, worse performance
        # auxCorners = np.concatenate( np.concatenate( corners, axis=0 ), axis=0 )
        # criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 50, 0.01)
        # auxCorners = cv.cornerSubPix(imgGray, auxCorners, winSize=(5,5), zeroZone=(-1,-1), criteria=criteria)
        # drawPointsInImage("img{} :  refined markers".format(camNum+1), img, auxCorners)

        if debug:
            # Draw detected markers
            frame_markers = aruco.drawDetectedMarkers(img.copy(), corners, ids)
            cv.imshow("img{} :  detected markers".format(camNum+1), frame_markers)

            frame_markers_rejected = aruco.drawDetectedMarkers(img.copy(), rejectedImgPoints)
            cv.imshow("img{} :  rejected markers".format(camNum+1), frame_markers_rejected)

        # Build correspondence between image points and object points
        imagePoints = np.empty( [len(ids)*4, 2] )
        objectPoints = np.empty( [len(ids)*4, 3] )
        for i in range(len(ids)):
            id = ids[i][0]
            imagePoints[i*4:(i+1)*4] = corners[i][0]
            objectPoints[i*4:(i+1)*4] = arucoMarkers[id]

        # Find camera pose
        flags = cv.SOLVEPNP_ITERATIVE
        retval, rvec, tvec = cv.solvePnP(objectPoints, imagePoints, cameraMatrix=K, distCoeffs=distCoeffs, flags=flags)
        tRef2Cam[camNum] = {"rvec": rvec, "tvec": tvec, "E": getRigBodyTransfMatrix(rvec, tvec)}

        # Refine?
        #solvePnPRefineLM()
        #solvePnPRefineVVS()

        if debug:
            # Project 3D points back to the camera with the estimated camera pose
            imagePointsEst, _ = cv.projectPoints(objectPoints, tRef2Cam[camNum]["rvec"], tRef2Cam[camNum]["tvec"],
                                                  cameraMatrix=K, distCoeffs=distCoeffs)
            imagePointsEst = imagePointsEst.round()
            annotatedImg = drawPointsInImage(img.copy(), imagePointsEst)
            # Draw cube reference frame
            rvec, tvec = getRvecTvec( tRef2Cam[camNum]["E"].dot(tCube2Ref["E"]) )
            annotatedImg = addReferenceMarker(annotatedImg, rvec, tvec, K, distCoeffs, d+e)
            cv.imshow("Aruco edge points projected back to cam{} with the estimated pose and cube reference frame".format(camNum+1),
                      annotatedImg)

            cv.waitKey()

    ## Compute transformation between cameras

    transfI2J = []
    for i in range(len(tRef2Cam)):
        auxt = []
        for j in range(len(tRef2Cam)):
            # Shown 2 alternatives to report the transformations between camera reference frames

            # Option 1: Euclidean rigid body transformation matrix (it's simpler to write and read)
            # Get Euclidean rigid body transformation
            Ei = tRef2Cam[i]["E"]
            Ej = tRef2Cam[j]["E"]
            # Find the transformation that converts a point in camera i reference frame back to the world
            retval, invEi = cv.invert(Ei)
            if not retval == 1:
                print("Failed to invert camera {} extrinsic calibration".format(i))
                exit(-1)
            # Compose both transformations to get a transformation from reference i to j
            dic = {"E": Ej.dot(invEi) }

            # Option 2: rotation matrix/vector and translation vector
            Ri, _ = cv.Rodrigues(tRef2Cam[i]["rvec"])
            Ti = tRef2Cam[i]["tvec"]
            Rj, _ = cv.Rodrigues(tRef2Cam[j]["rvec"])
            Tj = tRef2Cam[j]["tvec"]
            # Find the transformation that converts a point in camera i reference frame in camera j
            retval, invRi = cv.invert(Ri)
            if not retval == 1:
                print("Failed to invert camera {} rotation matrix".format(i))
                exit(-1)
            dic["rvec"] = cv.Rodrigues(Rj.dot(invRi))[0]
            dic["tvec"] = -Rj.dot(invRi).dot(Ti) + Tj

            if debug:
                # Checking if reported transforms are equivalent
                print(np.sum(dic["E"] - getRigBodyTransfMatrix(dic["rvec"], dic["tvec"])))

            auxt.append(dic)

        transfI2J.append( auxt )

    if debug:
        ## Quick tests
        # Sanity check. Transformation from camera i to j is the inverse of the one from camera j to i
        for i in range(len(transfI2J)):
            for j in range(i, len(transfI2J)):
                print( np.all( transfI2J[i][j]["E"].dot(transfI2J[j][i]["E"]) - np.eye(4) <
                               np.finfo(transfI2J[i][j]["E"].dtype).eps * 10 ) )

        # Test: Plot camera reference frames. Not working! Needs specific code to deal with points outside camera FOV
        # Standard opencv function does not deal with this well
        # annotatedImg2 = img.copy()
        # for i in range(len(transfI2J)):
        #     rvec, tvec = getRvecTvec( transfI2J[i][camNum] )
        #     annotatedImg = addReferenceMarker(annotatedImg2, rvec, tvec, K, distCoeffs, 4)
        # cv.imshow("annotated img with camera reference frames", annotatedImg2)

        cv.waitKey()

    return tRef2Cam, transfI2J

## Might come in handy

# sift = cv.SIFT_create()
# kp = sift.detect(imgGray,None)
# img=cv.drawKeypoints(imgGray,kp,img)
# cv.imshow("sift", img)
#
# cv.waitKey()