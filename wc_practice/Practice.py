import cv2
import numpy as np
import glob

def generate_charuco_board(d):
    charuco_board = cv2.aruco.CharucoBoard((5,7), .04, .03, d)
    charuco_board_img = charuco_board.generateImage((700, 1000))
    cv2.imshow("CharucoBoard", charuco_board_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return charuco_board

def live_charuco_detection(d):
    board = generate_charuco_board(d)

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        charuco_detector = cv2.aruco.CharucoDetector(board)
        charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(frame)
        if charuco_corners is not None and len(charuco_corners) > 0:
            cv2.aruco.drawDetectedCornersCharuco(frame, charuco_corners, charuco_ids)
            cv2.aruco.drawDetectedMarkers(frame, marker_corners, marker_ids)
        else:
            cv2.putText(frame, "No ChArUco detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Detecting_Charuco_Board", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()

def img_charuco_detection(d):
    board = generate_charuco_board(d)

    test_images = glob.glob("*.jpg")

    all_obj_points = []
    all_img_points = []
    image_size = None

    for img_name in test_images:
        test_img = cv2.imread(img_name)
        if test_img is None:
            print("⚠️  Could not read dummy.jpg — check file path!")
            return

        charuco_detector = cv2.aruco.CharucoDetector(board)
        charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(test_img)

        obj_points, img_points = board.matchImagePoints(charuco_corners, charuco_ids)

        if len(obj_points) > 0:
            all_obj_points.append(obj_points)
            all_img_points.append(img_points)
            if image_size is None:
                image_size = test_img.shape[:2][::-1]

        #if charuco_corners is not None and len(charuco_corners) > 0:
        #    all_charuco_corners.append(charuco_corners)
        #    all_charuco_ids.append(charuco_ids)

        #    if image_size is None:
        #        image_size = test_img.shape[:2][::-1]

        #    copy = test_img.copy()
        #    cv2.aruco.drawDetectedCornersCharuco(copy, charuco_corners, charuco_ids)
        #    cv2.aruco.drawDetectedMarkers(copy, marker_corners, marker_ids)
        #    cv2.imshow("Detecting_Charuco_Board", copy)
        #    if cv2.waitKey(0) == ord('q'):
        #        break
        #else:
        #    print("❌ No ChArUco corners detected")

    retval, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(all_obj_points, all_img_points, image_size, None, None)

    #np.savez("cam_calibration.txt", camera_matrix, dist_coeffs)
    print("Reprojection Error", retval)
    print("camera matrix", camera_matrix)
    print("distortion matrix", dist_coeffs)

    # === Pose estimation on a specific image ===
    test_img = cv2.imread("test_2.jpg")
    charuco_detector = cv2.aruco.CharucoDetector(board)
    charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(test_img)

    # Choose one image for pose estimation (you can loop through if you want)
    test_img = cv2.imread("test_3.jpg")
    if test_img is None:
        print("⚠️ Could not read test_3.jpg")
        return

    # Create the ArUco detector (same dictionary as used in your board)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(d, parameters)

    # Detect all individual ArUco tags in this image
    corners, ids, rejected = detector.detectMarkers(test_img)

    if ids is not None and len(ids) > 0:
        print(f"✅ Detected {len(ids)} markers: {ids.flatten()}")

        marker_length = 0.04  # meters (use your board’s known tag size)
        # Define 3D points for one square marker in object space
        obj_points = np.array([
            [-marker_length / 2, marker_length / 2, 0],
            [marker_length / 2, marker_length / 2, 0],
            [marker_length / 2, -marker_length / 2, 0],
            [-marker_length / 2, -marker_length / 2, 0]
        ], dtype=np.float32)

        # Estimate pose for each marker
        for i, corner in enumerate(corners):
            success, rvec, tvec = cv2.solvePnP(obj_points, corner, camera_matrix, dist_coeffs)
            if success:
                # Draw axes for each tag
                cv2.drawFrameAxes(test_img, camera_matrix, dist_coeffs, rvec, tvec, marker_length * 0.5)
                print(f"Marker {ids[i][0]} Pose:")
                print("Rotation Vector:\n", rvec)
                print("Translation Vector:\n", tvec)
        # Draw marker borders and IDs for reference
        cv2.aruco.drawDetectedMarkers(test_img, corners, ids)

    else:
        print("❌ No ArUco markers detected.")

    # Show the image with all poses drawn
    cv2.imshow("Aruco Marker Poses", test_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
img_charuco_detection(dictionary)
