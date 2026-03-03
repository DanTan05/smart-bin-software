import cv2
import numpy as np


class ImageProcessor:
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size

    def is_too_dark(self, image, threshold=15):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return np.mean(gray) < threshold

    def is_blurry(self, image, threshold=0):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var() < threshold

    def preprocess(self, image):
        if image is None:
            raise ValueError("Input image is None")

        if self.is_too_dark(image):
            raise ValueError("Image too dark")

        if self.is_blurry(image):
            raise ValueError("Image too blurry")

        image = cv2.resize(image, self.target_size)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # MobileNetV2 requires rescale to 0-1.
        # V2 does NOT have built-in preprocessing unlike V3Small.
        image = image.astype("float32") / 255.0

        return image
