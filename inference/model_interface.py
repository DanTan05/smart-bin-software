import random


class ModelInterface:
    def __init__(self):
        # Temporary class labels (replace later with real ones)
        self.classes = ["plastic", "can", "organic", "glass", "mixed"]

    def predict(self, processed_image):
        """
        Input: preprocessed image (224, 224, 3)
        Output: (predicted_class, confidence)
        """

        if processed_image is None:
            raise ValueError("Processed image is None")
        

        # ---- TEMPORARY STUB LOGIC ----


        predicted_class = random.choice(self.classes)
        confidence = round(random.uniform(0.5, 0.99), 2)
        return predicted_class, confidence

'''        predicted_class = "mixed"
        confidence = 0.99
        '''
