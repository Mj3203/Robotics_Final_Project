import cv2
import numpy as np
import os
import glob

def display_image(image):
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 0

def save_pictures(folder_name: str, image_name: str, image_file: np.ndarray):
    #Here you need to specify the type for the inputs
    #cv2.write expects a string and a ndarray
    os.makedirs(folder_name, exist_ok=True)
    full_path = os.path.join(folder_name, image_name)
    cv2.imwrite(full_path, image_file)
    return 0

def get_dictionary():
    #Custom dictionary - stores the different types of aruco tags we could potentially use
    custom_aruco_dict = {
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
    #Defining which aruco tag we will use
    aruco_tag_type = "DICT_4X4_50"
    aruco_dictionary_to_use = cv2.aruco.getPredefinedDictionary(custom_aruco_dict[aruco_tag_type])
    return aruco_dictionary_to_use, aruco_tag_type

def generate_aruco_tags(aruco_dict, aruco_tag_type, number_of_tags):
    #We create our aruco tags with generateImageMarkers
    #Inputs: dictionary, tag id # or index, size of displayed image(pixels), OPTIONAL numpy array to fill, border color)

    #When we create our tags, if the numpy array parameter is not passed, the function will create and store one
    #This can cause issues storage/memory issues
    #Each iteration will allocate memory for each new array rather than reusing the same one

    image_tag_size = 300  # <-- aruco size in pixels
    empty_image = np.zeros((image_tag_size, image_tag_size),
                   dtype='uint8')  # <-- empty numpy array, this will become our image that displays the aruco
    for tag_id in range(number_of_tags):
        created_tag = cv2.aruco.generateImageMarker(aruco_dict, tag_id, image_tag_size, empty_image, 1)
        created_tag_name = aruco_tag_type + "_" + str(tag_id) + ".png"

        # To see image, uncomment
        #display_image(created_tag)

        # To save image, uncomment
        #save_pictures('aruco_marker_images', created_tag_name, created_tag)
    return 0

def generate_charuco_board(aruco_dict):
    #Defining the size of the board
    #Inputs: size of board, size of the chessboard square, size of the aruco marker tag, dictionary
    charuco_board = cv2.aruco.CharucoBoard((5,7), .04, .03, aruco_dict)
    charuco_board_img = charuco_board.generateImage((700, 1000))

    #To see image, uncomment
    #display_image(charuco_board_img)

    #To save image, uncomment
    #save_pictures('charuco_board_images', 'charuco_board.png', charuco_board_img)
    return charuco_board

#def live_charuco_detection(d):
#    board = generate_charuco_board(d)
#
#    cap = cv2.VideoCapture(0)
#    while True:
#        ret, frame = cap.read()
#        if not ret:
#            break
#
#        charuco_detector = cv2.aruco.CharucoDetector(board)
#        charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(frame)
#        if charuco_corners is not None and len(charuco_corners) > 0:
#            cv2.aruco.drawDetectedCornersCharuco(frame, charuco_corners, charuco_ids)
#            cv2.aruco.drawDetectedMarkers(frame, marker_corners, marker_ids)
#        else:
#            cv2.putText(frame, "No ChArUco detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#
#        cv2.imshow("Detecting_Charuco_Board", frame)
#        if cv2.waitKey(1) == ord('q'):
#            break
#
#    cv2.destroyAllWindows()

def camera_calibration(aruco_dict):
    #Here we want to access our calibration images
    #Ideally, you want 10-20 images taken from different angles
    calibration_image_directory = 'calibration_images'
    file_path = os.path.join(calibration_image_directory, "*.jpg")
    calibration_images = glob.glob(file_path)

    #Define the charuco board
    calibration_board = generate_charuco_board(aruco_dict)

    all_obj_points = []
    all_img_points = []
    image_size = None

    #This essentially says for each image in the directory
    #Loops through each index of the list, which is the image inside the directory
    for img_name in calibration_images:
        current_image = cv2.imread(img_name)
        cv2.waitKey(0)
        if current_image is None:
            print("Could not read .jpg — check file path")

        #This detector is initialized with the information about our board
        #It tells the detector what shape and size of the checkerboard to expect, and what dictionary to look for
        charuco_detector = cv2.aruco.CharucoDetector(calibration_board)

        #This is the main detection function. It attempts to locate the relevant features of the board
        charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(current_image)

        #This function organizes the 2D points with the corresponding 3D world coordinates
        obj_points, img_points = calibration_board.matchImagePoints(charuco_corners, charuco_ids)

        #This checks to see if our matchImagePoints function was able to validate and map 1 corner to its known world coords
        if len(obj_points) > 0 and len(img_points) > 0:
            all_obj_points.append(obj_points)
            all_img_points.append(img_points)
            if image_size is None:
                image_size = current_image.shape[:2][::-1]
        else:
            print("Could not detect a corner")

    reprojection_error, camera_matrix, distortion_coefficients, rvecs, tvecs = cv2.calibrateCamera(all_obj_points, all_img_points, image_size, None, None)

    print("Reprojection Error", reprojection_error)
    print("Camera Matrix", camera_matrix)
    print("Distortion Matrix", distortion_coefficients)
    return reprojection_error, camera_matrix, distortion_coefficients, rvecs, tvecs

def aruco_detector(aruco_dict, camera_matrix, dist_coeffs):
    test_img = cv2.imread("ooga.jpg")

    #It performs filtering, corner refinement, and thresholding to find the corners in image
    parameters = cv2.aruco.DetectorParameters()
    #Uses dict to know what to look for. Uses the parameters know how to look for the tags
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    # Detect aruco tags from an image, outputs the corners and aruco ids from the image plane
    # Corners gives all 4 corners of the tag from the image
    corners, ids, rejected = detector.detectMarkers(test_img)

    #We want to perform a check that we actually identified a tag
    if ids is not None and len(ids) > 0:
        print(f" Detected {len(ids)} markers: {ids.flatten()}")

        marker_length = 0.04  #this is the size of our tag (in charuco tag we defined this to be 0.04m)

        #We define the marker's shape and size in real wold coordinates and set the origin
        #We need this to be able to find the pose that allows this shape to be projected onto the observed corners in the image plane
        obj_points = np.array([
            [-marker_length / 2, marker_length / 2, 0],
            [marker_length / 2, marker_length / 2, 0],
            [marker_length / 2, -marker_length / 2, 0],
            [-marker_length / 2, -marker_length / 2, 0]
        ], dtype=np.float32)

        # Estimate pose for each marker
        for i in range(len(ids)):
            success, rvec, tvec = cv2.solvePnP(obj_points, corners[i], camera_matrix, dist_coeffs)
            if success:
                # Draw axes for each tag
                cv2.drawFrameAxes(test_img, camera_matrix, dist_coeffs, rvec, tvec, marker_length * 0.5)
        # Draw marker borders and IDs for reference
        cv2.aruco.drawDetectedMarkers(test_img, corners, ids)
    else:
        print("No ArUco markers detected.")

    # Show the image with all poses drawn
    scale = 0.5
    resized = cv2.resize(test_img, (0, 0), fx=scale, fy=scale)
    cv2.imshow("Resized Image", resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_chessboard_corner_obj_points(L):
    obj_points_TL = np.array([
        [0, 0, 0],  # Top-Left (Origin)
        [L, 0, 0],
        [L, -L, 0],
        [0, -L, 0]
    ], dtype=np.float32)

    obj_points_TR = np.array([
        [-L, 0, 0],
        [0, 0, 0],  # Top-Right (Origin)
        [0, -L, 0],
        [-L, -L, 0]
    ], dtype=np.float32)

    obj_points_BR = np.array([
        [-L, L, 0],
        [0, L, 0],
        [0, 0, 0],  # Bottom-Right (Origin)
        [-L, 0, 0]
    ], dtype=np.float32)

    obj_points_BL = np.array([
        [0, L, 0],
        [L, L, 0],
        [L, 0, 0],
        [0, 0, 0]  # Bottom-Left (Origin)
    ], dtype=np.float32)

    chessboard_corner_obj_points = [obj_points_BL, obj_points_TL, obj_points_BR, obj_points_TR]
    return chessboard_corner_obj_points

def chessboard_corner_detection(aruco_dict, camera_matrix, dist_coeffs):
    image = cv2.imread("ooga.jpg")

    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, rejected = detector.detectMarkers(image)

    marker_length = 0.04
    obj_points = get_chessboard_corner_obj_points(marker_length)

    for i in range(len(ids)):
        success, rvec, tvec = cv2.solvePnP(obj_points[i], corners[i], camera_matrix, dist_coeffs)
        if success:
            cv2.drawFrameAxes(image, camera_matrix, dist_coeffs, rvec, tvec, marker_length * 0.5)
            cv2.aruco.drawDetectedMarkers(image, corners, ids)
        origin_3d = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)

        # Use the pose and intrinsics to find the 2D pixel coordinates
        img_coords, jacobian = cv2.projectPoints(
            origin_3d,
            rvec,
            tvec,
            camera_matrix,
            dist_coeffs
        )

        # Extract and convert the pixel coordinates to integers
        origin_x = int(img_coords[0][0][0])
        origin_y = int(img_coords[0][0][1])

        print(f"Tag ID {i} Origin is at pixel: ({origin_x}, {origin_y})")
    else:
        print("No ArUco markers detected.")

    # Show the image with all poses drawn
    scale = 0.5
    resized = cv2.resize(image, (0, 0), fx=scale, fy=scale)
    cv2.imshow("Resized Image", resized)
    cv2.waitKey(0)

def main():
    #Calls the dictionary we want to use
    aruco_dict_to_use, aruco_tag_type = get_dictionary()

    #Generates aruco tags, saves them to aruco_markers_images
    #number_of_tags = 2
    #generate_aruco_tags(aruco_dict_to_use, aruco_tag_type, number_of_tags)

    #Generates a charuco board, saves it to charuco_board_images
    #generate_charuco_board(aruco_dict_to_use)

    #Calibrates camera using test images
    retval, camera_matrix, dist_coeffs, rvecs, tvecs = camera_calibration(aruco_dict_to_use)

    aruco_detector(aruco_dict_to_use, camera_matrix, dist_coeffs)
    chessboard_corner_detection(aruco_dict_to_use, camera_matrix, dist_coeffs)
if __name__ == "__main__":
    main()

#dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
#img_charuco_detection(dictionary)
