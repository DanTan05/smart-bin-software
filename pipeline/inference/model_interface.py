import os
import numpy as np
import tensorflow as tf


class ModelInterface:
    def __init__(self, tflite_model_path=None):
        if tflite_model_path is None:
            tflite_model_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "waste_classifier.tflite"
            )

        # Class order matches train.py class indices exactly:
        # {'battery': 0, 'cans': 1, 'organic': 2, 'paper': 3, 'plastic': 4}
        self.classes = ["battery", "cans", "organic", "paper", "plastic"]

        # Battery always triggers alert
        self.alert_classes = {"battery"}

        # Temperature scaling
        # MobileNetV2 on black-background images is well-calibrated at T=1.0.
        # Raise to 1.1 or 1.2 only if evaluate.py shows avg correct confidence > 0.95.
        self.temperature = 1.2

        self.interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
        self.interpreter.allocate_tensors()
        self.input_details  = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def _apply_temperature(self, probs):
        if self.temperature == 1.0:
            return probs
        eps    = 1e-7
        log_p  = np.log(np.clip(probs, eps, 1.0))
        scaled = log_p / self.temperature
        scaled -= scaled.max()
        exp_s  = np.exp(scaled)
        return exp_s / exp_s.sum()

    def predict(self, processed_image):
        """
        Input:  preprocessed image (224, 224, 3) float32 scaled 0-1.
                ImageProcessor divides by 255 before passing here.
                MobileNetV2 expects values in 0-1 range.

        Output: (predicted_class, confidence, requires_alert, all_probs)
                all_probs is a dict mapping each class name to its probability.
        """
        if processed_image is None:
            raise ValueError("Processed image is None")

        img = np.expand_dims(processed_image, axis=0).astype(np.float32)
        self.interpreter.set_tensor(self.input_details[0]['index'], img)
        self.interpreter.invoke()
        raw_probs = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

        probs           = self._apply_temperature(raw_probs)
        predicted_idx   = int(np.argmax(probs))
        confidence      = float(probs[predicted_idx])
        predicted_class = self.classes[predicted_idx]
        requires_alert  = predicted_class in self.alert_classes
        all_probs       = dict(zip(self.classes, probs.tolist()))

        return predicted_class, confidence, requires_alert, all_probs
