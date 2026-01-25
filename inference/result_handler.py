class ResultHandler:
    def __init__(self, confidence_threshold=0.7):
        """
        confidence_threshold:
        Minimum confidence required to accept an AI prediction.
        """
        self.confidence_threshold = confidence_threshold

    def handle(self, predicted_class, confidence):
        """
        Input:
            predicted_class (str)
            confidence (float)

        Output:
            final_class (str)
            accepted (bool)     
        """

        if confidence >= self.confidence_threshold:
            return predicted_class, True
        else:
            # Low confidence → treat as general waste
            return "mixed", False
