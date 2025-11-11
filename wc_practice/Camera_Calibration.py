#What is Camera Calibration?

#Camera Calibration is used to obtain the following camera parameters
#Intrinsics such as focal length(f), principal point(cx, cy) and distortion(k1, k2, p1, p2, k3)
#Extrinsics such as rotation and translation

import cv2
import numpy as np

#from wc_practice.Create_Aruco_Tags import tag_size

cap = cv2.VideoCapture(0) #use the camera

while True:
    ret, frame = cap.read()

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release() #closes video file or capturing device
cv2.destroyAllWindows() #releases all windows

#arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

## ------------------- Creating the Charuco Board -------------------- ##
#CharucoBoard takes in (number of chessboard squares in x and y, size of each square(m), the size of the aruco tag, dictionary)
#board = cv2.aruco.CharucoBoard((6,6), 0.025,0.02, arucoDict)

#tag_size = 300
#tag = np.zeros((tag_size, tag_size), dtype='uint8')
#img = board.generateImage(6, 6))
