import cv2
import numpy as np


class ImageProcessor:
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size

    def is_too_dark(self, image, threshold=40):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        return mean_brightness < threshold

    def is_blurry(self, image, threshold=100):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        return variance < threshold

    def preprocess(self, image):

        if image is None:
            raise ValueError("Input image is None")

        # Quality checks
        if self.is_too_dark(image):
            raise ValueError("Image too dark")

        if self.is_blurry(image):
            raise ValueError("Image too blurry")

        # Resize
        image = cv2.resize(image, self.target_size)

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1]
        image = image.astype("float32") / 255.0

        return image

