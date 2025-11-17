import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("ChessModel.pt")

#results is a list
results = model("chessboard_w_homography.jpg")
res = results[0]  # <-- FIX: get the first (and only) Results object

res.show()
