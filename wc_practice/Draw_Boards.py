import cv2
import numpy as np

# Assume you already have this from your calibration setup:
# H_board2img  — homography mapping from board-plane (0..8,0..8) → image pixels
#                e.g. from cv2.findHomography(board_pts, img_pts)
# Each "square" on the board has coordinates in board space:
# (i, j), (i+1, j), (i+1, j+1), (i, j+1)

files = "ABCDEFGH"  # chess file letters
square_polys = {}  # dictionary: { "A1": np.array([[x,y],...]), ... }

for i in range(8):  # file (column)
    for j in range(8):  # rank (row)
        # 1️⃣ corners in board coordinate units
        corners_board = np.float32([[[i, j]], [[i + 1, j]], [[i + 1, j + 1]], [[i, j + 1]]])

        # 2️⃣ project into image using homography
        corners_img = cv2.perspectiveTransform(corners_board, H_board2img)
        corners_img = corners_img.reshape(-1, 2)  # flatten to (4,2)

        # 3️⃣ assign algebraic name, rank 1–8 bottom→top
        square_name = f"{files[i]}{j + 1}"
        square_polys[square_name] = corners_img
