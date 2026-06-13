import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

baseOption = python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(base_options=baseOption, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

image = mp.Image.create_from_file("hand.jpg")
result = detector.detect(image)

annotatedImage = draw_landmarks_on_image(image.numpy_view(), result)
cv2.imshow(cv2.cvtColor(annotatedImage, cv2.COLOR_RGB2BGR))