import os
import time
from camera_input.image_source import ImageSource
from preprocessing.image_preprocessor import ImageProcessor
from inference.model_interface import ModelInterface
from inference.result_handler import ResultHandler
from communication.api_client import ApiClient
from logs.classification_logger import ClassificationLogger

# ---------------- CONFIG ----------------
PIPELINE_DIR  = os.path.dirname(os.path.abspath(__file__))
BIN_ID        = "BIN_001"
API_ENDPOINT  = "https://ingestbinevent-t4vkrtxd5q-uc.a.run.app"
SEND_API_EVENTS = False   # set True only when testing with backend
# ----------------------------------------

def main():
    image_source    = ImageSource(
        os.path.join(PIPELINE_DIR, "test_images"),
        os.path.join(PIPELINE_DIR, "processed_images")
    )
    image_processor = ImageProcessor()
    model           = ModelInterface()
    result_handler  = ResultHandler()
    api_client      = ApiClient(API_ENDPOINT)
    logger          = ClassificationLogger()

    print("\nStarting BIN pipeline for latest image...\n")

    image, path = image_source.get_latest_image()
    print(f"Image loaded: {path}")

    # ── Vision + fallback ────────────────────────────────────────────────
    preprocessing   = "passed"
    predicted_class = None
    confidence      = None
    all_probs       = {}
    final_class     = "mixed"
    accepted        = False
    alert           = False
    inference_ms    = 0.0

    try:
        processed = image_processor.preprocess(image)

        t_start = time.time()
        predicted_class, confidence, requires_alert, all_probs = model.predict(processed)
        inference_ms = (time.time() - t_start) * 1000

        final_class, accepted, alert = result_handler.handle(
            predicted_class, confidence, requires_alert
        )

        print(f"AI result:    {predicted_class} ({confidence:.2f} confidence)")
        print(f"Final class:  {final_class}  |  Accepted: {accepted}")
        print(f"Inference:    {inference_ms:.1f} ms")

        if alert:
            print(f"🚨 ALERT: Battery detected — hazardous item")

    except ValueError as e:
        preprocessing = str(e)
        print(f"Preprocessing issue: {e}")
        print("Fallback → MIXED waste")

    # ── Log classification ───────────────────────────────────────────────
    logger.log(
        image_path=path,
        preprocessing=preprocessing,
        predicted_class=predicted_class or "",
        confidence=confidence,
        all_probs=all_probs,
        final_class=final_class,
        accepted=accepted,
        alert=alert,
        inference_time_ms=inference_ms,
    )

    # ── PIECE COLLECTED event ────────────────────────────────────────────
    piece_event = {
        "binId":     BIN_ID,
        "subBin":    final_class,
        "eventType": "PIECE_COLLECTED"
    }

    print(f"Event → PIECE_COLLECTED | subBin: {final_class}")

    if SEND_API_EVENTS:
        api_client.send_event(piece_event)
    else:
        print("[API] Dry run (not sent)")

    # ── BATTERY DETECTED event ──────────────────────────────────────────
    if alert:
        battery_event = {
            "binId":     BIN_ID,
            "eventType": "BATTERY_DETECTED",
            "subBin":    "mixed"
        }
        print("🚨 Event → BATTERY_DETECTED | subBin: mixed")
        if SEND_API_EVENTS:
            api_client.send_event(battery_event)
        else:
            print("[API] Dry run (not sent)")

    print("\nPipeline run completed.\n")


if __name__ == "__main__":
    main()
