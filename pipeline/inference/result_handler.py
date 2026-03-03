class ResultHandler:
    def __init__(self, confidence_threshold=0.60):
        """
        confidence_threshold:
            Minimum confidence to accept a prediction.
            Lowered from 0.7 to 0.60 to match evaluate.py and inference_pipeline.py.
            Our model's wrong predictions average 0.482 confidence, so 0.60 safely
            catches uncertain predictions without over-rejecting correct ones.
        """
        self.confidence_threshold = confidence_threshold

        # Classes that always route to mixed bin regardless of confidence
        self.mixed_classes = {"battery"}

    def handle(self, predicted_class, confidence, requires_alert=False):
        """
        Input:
            predicted_class (str)
            confidence      (float)
            requires_alert  (bool) -- passed through from ModelInterface

        Output:
            final_class  (str)   -- bin slot to route item to
            accepted     (bool)  -- True if prediction was confident enough
            alert        (bool)  -- True if battery detected (notify app)
        """

        # Battery: always route to mixed AND always alert, regardless of confidence
        # A low-confidence battery guess is still safer to treat as battery
        if predicted_class == "battery":
            return "mixed", True, True

        # Low confidence → mixed bin, no alert
        if confidence < self.confidence_threshold:
            return "mixed", False, False

        # Normal accepted prediction
        return predicted_class, True, False
