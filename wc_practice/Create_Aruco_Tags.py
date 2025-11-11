import numpy as np
import cv2

## ----------------------- Custom Dictionary to Store Types of Aruco Tags ----------------------- ##
##Here we created a dictionary that stores the different types of aruco tags we may use
ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000
}
##Here we defined the type of aruco tag we might want to use
aruco_type = "DICT_4X4_50"

## ------------------------ Creating the Aruco Dictionary  -------------------------- ##
##getPredefinedDictionary is a dictionary of dictionaries of aruco tags
##We pass in an index(the type of aruco we want) and returns a dictionary of aruco tags of that type
#arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50) <-- also works
arucoDict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[aruco_type])

## ------------------------- Generating an Aruco Tag ----------------------------- ##
##generateImageMarker is a function that creates an aruco marker
##We pass in the (dictionary of aruco tags, the index of the dictionary, size of the aruco in pixels, OPTIONAL numpy array to fill, the border color)
##If the numpy array parameter is not passed, the function will create and store one itself
##This can cause issues if used in a loop. Each iteration will allocate memory for each new array rather than reusing the same one

tag_size = 300 #<-- aruco size in pixels
tag = np.zeros((tag_size, tag_size), dtype='uint8') #<-- empty numpy array, this will become our "Aruco Tag or Image"

for id in range(3):
    cv2.aruco.generateImageMarker(arucoDict, id , tag_size, tag, 1)
    print("AruCo type '{}' with ID '{}'".format(aruco_type, id))

    ## ------------------------- Displaying and Saving the Generated Tags -------------------------- ##
    tag_name = "arucoMarkers/" + aruco_type + "_" + str(id) + ".png"
    cv2.imwrite(tag_name, tag) #<-- storing and saving the tags

    cv2.imshow("ArUCO", tag) #<-- showing the tag

cv2.waitKey(0)
cv2.destroyAllWindows()