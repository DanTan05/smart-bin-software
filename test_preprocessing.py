import cv2
from preprocessing.image_preprocessor import ImageProcessor

processor = ImageProcessor()

image = cv2.imread(r'test_images\plastic15.jpg')

try:
    processed = processor.preprocess(image)
    print("Preprocessing successful")
    print("Processed image shape:", processed.shape)
    print("Pixel range:", processed.min(), processed.max())
except ValueError as e:
    print("Preprocessing failed:", e)
