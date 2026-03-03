import csv
import os
from datetime import datetime


class ClassificationLogger:
    """Logs every classification to a CSV file for analysis and debugging."""

    COLUMNS = [
        "timestamp",
        "image_path",
        "preprocessing",
        "predicted_class",
        "confidence",
        "prob_battery",
        "prob_cans",
        "prob_organic",
        "prob_paper",
        "prob_plastic",
        "final_class",
        "accepted",
        "alert",
        "inference_time_ms",
    ]

    def __init__(self, log_dir=None):
        if log_dir is None:
            log_dir = os.path.dirname(os.path.abspath(__file__))

        self.log_file = os.path.join(log_dir, "classifications.csv")

        # Write header only if the file does not exist yet
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(self.COLUMNS)

    def log(
        self,
        image_path,
        preprocessing,
        predicted_class,
        confidence,
        all_probs,
        final_class,
        accepted,
        alert,
        inference_time_ms,
    ):
        """
        Append one classification row to the CSV log.

        Parameters:
            image_path       (str)   : path to the original image
            preprocessing    (str)   : "passed" or the rejection reason
            predicted_class  (str)   : raw model prediction before threshold
            confidence       (float) : top confidence after temperature scaling
            all_probs        (dict)  : {"battery": 0.01, "cans": 0.02, ...}
            final_class      (str)   : routed class after business logic
            accepted         (bool)  : True if prediction cleared threshold
            alert            (bool)  : True if battery alert was triggered
            inference_time_ms (float): model inference time in milliseconds
        """
        row = [
            datetime.now().isoformat(timespec="seconds"),
            image_path,
            preprocessing,
            predicted_class,
            f"{confidence:.4f}" if confidence is not None else "",
            f"{all_probs.get('battery', 0):.4f}" if all_probs else "",
            f"{all_probs.get('cans', 0):.4f}" if all_probs else "",
            f"{all_probs.get('organic', 0):.4f}" if all_probs else "",
            f"{all_probs.get('paper', 0):.4f}" if all_probs else "",
            f"{all_probs.get('plastic', 0):.4f}" if all_probs else "",
            final_class,
            accepted,
            alert,
            f"{inference_time_ms:.1f}" if inference_time_ms is not None else "",
        ]

        try:
            with open(self.log_file, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(row)
            print(f"[LOG] Saved to {self.log_file}")
        except PermissionError:
            print(f"[LOG] WARNING: Could not write to {self.log_file} "
                  f"— file may be open in another application. "
                  f"Close it and re-run to log this entry.")
