import cv2
import os
import shutil

class ImageSource:
    def __init__(self, image_folder, processed_folder="processed_images"):
        self.image_folder = image_folder
        self.processed_folder = processed_folder

        # Create processed folder if it doesn't exist
        os.makedirs(self.processed_folder, exist_ok=True)

    def get_latest_image(self):
        # Get all image files in the folder
        files = [
            os.path.join(self.image_folder, f)
            for f in os.listdir(self.image_folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        if not files:
            raise ValueError("No images found in the image folder")

        # Pick the latest by modification time
        latest_file = max(files, key=os.path.getmtime)

        # Load the image
        image = cv2.imread(latest_file)
        if image is None:
            raise ValueError(f"Failed to load image: {latest_file}")

        # Move to processed folder so it won't be used again
        dest_path = os.path.join(self.processed_folder, os.path.basename(latest_file))
        shutil.move(latest_file, dest_path)

        return image, latest_file
