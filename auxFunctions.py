import cv2 as cv
import numpy as np

## Auxiliary function definitions

def drawPointsInImage(img, pts):
    for pt in pts:
        pt = pt[0].astype("int")
        img = cv.circle(img, (pt[0], pt[1]), radius=3, color=(0, 255, 0))
    return img

def addReferenceMarker(img, rvec, tvec, cameraMatrix, distCoeffs, scale=0.3):

    marker = scale * np.array([ [0,0,0], [1,0,0], [0,0,0], [0,1,0], [0,0,0], [0,0,1] ])

    imgPts, _ = cv.projectPoints(marker, rvec, tvec, cameraMatrix=cameraMatrix, distCoeffs=distCoeffs)
    imgPts = imgPts.astype("int")

    img = cv.line(img, tuple(imgPts[0].ravel()), tuple(imgPts[1].ravel()), (255, 0, 0), 4)
    img = cv.line(img, tuple(imgPts[2].ravel()), tuple(imgPts[3].ravel()), (0, 255, 0), 4)
    img = cv.line(img, tuple(imgPts[4].ravel()), tuple(imgPts[5].ravel()), (0, 0, 255), 4)

    return img

def getRigBodyTransfMatrix(rvec, tvec):
    return np.block( [[cv.Rodrigues(rvec)[0],   tvec],
                      [np.zeros(3),             1]] )

def getRvecTvec(E):
    rvec = cv.Rodrigues( E[0:3,0:3] )[0]
    tvec = E[0:3,3].reshape(3,1)
    return rvec, tvec

def transformJafterI(rvecJ, tvecJ, rvecI, tvecI):
    EJ = getRigBodyTransfMatrix(rvecJ, tvecJ)
    EI = getRigBodyTransfMatrix(rvecI, tvecI)
    return getRvecTvec( EJ.dot(EI) )

def applyRigBodyTransformation(X, rvec, tvec):
    E = getRigBodyTransfMatrix(rvec, tvec)

    auxX = np.block( [[ X.transpose() ],
                      [ np.ones(len(X)) ]] )
    return E.dot(auxX)[0:3].transpose()

def project2Cam(X, K):
    auxx = K.dot(X.transpose())
    auxx = auxx / auxx[2,:]
    return auxx[0:2,:].transpose()

def filterPointsOutsideCamFOV(X, x, sz):
    i = 0
    for j in range(len(x)):
        if 0 <= x[j][0] and x[j][0] <= sz[0] and 0 <= x[j][1] and x[j][1] <= sz[1]:
            X[i,:] = X[j,:]
            i = i + 1
    return X[:i,:]