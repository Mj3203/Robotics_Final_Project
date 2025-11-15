import cv2
import numpy as np

# === 1. Normalized Center Positions (from your second snippet) ===

# Size of one square (1/8 of the normalized board width)
S = 1 / 8
# Board length in normalized units (1.0)
NORMALIZED_BOARD_LENGTH = 1.0

# Array of center positions (0.0625, 0.1875, ..., 0.9375)
CENTER_POSITIONS = [(k + 0.5) * S for k in range(8)]

# Files ('a' through 'h') and Ranks ('8' through '1')
FILES = 'abcdefgh'
RANKS = '87654321'

SQUARE_CENTERS = {}

# Populate the dictionary with normalized (x, y) coordinates
for j, rank in enumerate(RANKS):  # j is the Y-index (0 to 7)
    for i, file in enumerate(FILES):  # i is the X-index (0 to 7)
        X_center = CENTER_POSITIONS[i]
        Y_center = CENTER_POSITIONS[j]
        square_name = file + rank
        SQUARE_CENTERS[square_name] = (X_center, Y_center)

# === 2. Mocking Calibration Data and Utility Functions ===

# IMPORTANT: You MUST replace these with your actual calibration data.
# Placeholder Camera Matrix (Intrinsic Parameters)
# Assuming a focal length of 800 pixels and image center (cx, cy)
CAMERA_MATRIX = np.array([
    [800.0, 0.0, 320.0],
    [0.0, 800.0, 240.0],
    [0.0, 0.0, 1.0]
])

# Placeholder Distortion Coefficients (K1, K2, P1, P2, K3)
DIST_COEFFS = np.zeros((5, 1))

# The length of one ArUco marker side in meters
MARKER_LENGTH = 0.04

# The total board side length in meters (8 squares * 0.04m/square)
BOARD_LENGTH_M = 8 * MARKER_LENGTH  # 0.32 meters


# Mock function for getting 3D object points of the ArUco marker.
# This is highly dependent on your ArUco pattern generation.
# Assuming a simple 4x4 ArUco tag where the origin (0,0,0) is at a specific corner
# and the obj_points define the corner locations relative to that origin.
def get_chessboard_corner_obj_points(marker_len):
    """Mocks the object points for four corners of one ArUco marker."""
    # Assuming origin (0,0,0) is top-left of the marker, Z-axis points out.
    half_len = marker_len / 2.0
    obj_points = np.array([
        [[-half_len, half_len, 0], [half_len, half_len, 0],
         [half_len, -half_len, 0], [-half_len, -half_len, 0]]
    ], dtype=np.float32)
    # If using multiple markers, this structure would be more complex (list of 4x3 arrays)
    return [obj_points[0]]


# === 3. Projection Function ===

def project_square_center(image_path, target_square, aruco_dict, camera_matrix, dist_coeffs):
    """
    Detects ArUco markers, estimates the board pose, and projects
    the center of a target chess square onto the image.

    Args:
        image_path (str): Path to the image file (e.g., "ooga.jpg").
        target_square (str): Chess notation (e.g., 'e4').
        aruco_dict (cv2.aruco.Dictionary): Dictionary to use for detection.
        camera_matrix (np.array): Camera intrinsic matrix.
        dist_coeffs (np.array): Distortion coefficients.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return

    # Check if the target square name is valid
    if target_square not in SQUARE_CENTERS:
        print(f"Error: Square '{target_square}' not found in defined centers.")
        return

    # Normalized center point (0 to 1)
    x_norm, y_norm = SQUARE_CENTERS[target_square]

    # ArUco Detection Setup
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, rejected = detector.detectMarkers(image)

    if ids is not None and len(ids) > 0:

        # --- A. Determine the Board's World Coordinates (3D) ---

        # We assume the coordinate system is defined by the first detected marker.
        marker_length = MARKER_LENGTH  # 0.04 m

        # Calculate the 3D world point of the target center.
        # This point is relative to the origin (0,0,0) defined by your ArUco object points.

        # The normalized coordinates (0 to 1) are scaled by the full board length in meters (0.32m).
        X_w = x_norm * BOARD_LENGTH_M
        Y_w = y_norm * BOARD_LENGTH_M
        Z_w = 0.0  # All points are on the Z=0 board plane

        # The target 3D point in the world frame (relative to the ArUco origin)
        target_3d_point = np.array([[X_w, Y_w, Z_w]], dtype=np.float32)

        # --- B. Get the Pose (rvec, tvec) ---

        # Use the pose of the first detected marker (ID 0) as the board's pose.
        # This requires the obj_points to be defined for that specific marker's ID and position.

        # IMPORTANT: This step requires that your `get_chessboard_corner_obj_points`
        # defines the 3D position of the marker relative to the desired board origin (A8).
        # We will use a mock to continue the process.

        # In a real application, you would ensure the marker used here establishes the
        # consistent X-Y plane for the entire board.

        obj_points_list = get_chessboard_corner_obj_points(marker_length)

        # Using the first detected marker's image corners and its corresponding 3D object points
        # to calculate the board's pose.
        success, rvec, tvec = cv2.solvePnP(
            obj_points_list[0],  # 3D points of the object (marker)
            corners[0],  # 2D points of the marker in the image
            camera_matrix,
            dist_coeffs
        )

        if success:
            # --- C. Project the 3D Point to 2D Pixel Coordinates ---

            img_coords, jacobian = cv2.projectPoints(
                target_3d_point,
                rvec,
                tvec,
                camera_matrix,
                dist_coeffs
            )

            # Extract the pixel coordinates
            pixel_x = int(img_coords[0][0][0])
            pixel_y = int(img_coords[0][0][1])

            print(
                f"Square {target_square} (Normalized: {x_norm:.4f}, {y_norm:.4f}) projected to Pixel: ({pixel_x}, {pixel_y})")

            # --- D. Draw and Display ---

            # Draw a circle at the calculated center
            cv2.circle(image, (pixel_x, pixel_y), 5, (0, 255, 0), -1)  # Green circle

            # Put the square name text next to the center
            cv2.putText(image, target_square, (pixel_x + 10, pixel_y + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)  # Blue text

            # Draw the pose axis for debugging (optional)
            #cv2.drawFrameAxes(image, camera_matrix, dist_coeffs, rvec, tvec, marker_length * 1.5)
            cv2.aruco.drawDetectedMarkers(image, corners, ids)

        else:
            print("PnP solve was unsuccessful.")

    else:
        print("No ArUco markers detected. Cannot estimate board pose.")

    # Show the image with all markings
    cv2.imshow("Projected Chess Center", image)
    cv2.imwrite("projected_center_output.png", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# === 4. Execution (Example) ===

if __name__ == '__main__':
    # Initialize ArUco dictionary (choose the one you used to print your markers)
    # Example: cv2.aruco.DICT_6X6_250
    try:
        ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    except AttributeError:
        # Fallback for older OpenCV versions
        ARUCO_DICT = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)

    # Define the square you want to find the center for
    TARGET_SQUARE_NAME = 'e4'

    # Execute the function
    project_square_center(
        image_path="ooga.jpg",
        target_square=TARGET_SQUARE_NAME,
        aruco_dict=ARUCO_DICT,
        camera_matrix=CAMERA_MATRIX,
        dist_coeffs=DIST_COEFFS
    )