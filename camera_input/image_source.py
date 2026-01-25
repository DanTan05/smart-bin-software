import cv2
import os


class ImageSource:
    def __init__(self, image_folder):
        self.image_folder = image_folder
        self.image_files = self._load_image_list()
        self.index = 0

    def _load_image_list(self):
        files = []
        for file in os.listdir(self.image_folder):
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                files.append(os.path.join(self.image_folder, file))

        if not files:
            raise ValueError("No images found in the image folder")

        return files

    def get_next_image(self):
        """
        Returns the next image from the folder.
        Loops back to the start when all images are used.
        """
        image_path = self.image_files[self.index]
        image = cv2.imread(image_path)

        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        self.index = (self.index + 1) % len(self.image_files)
        return image, image_path
