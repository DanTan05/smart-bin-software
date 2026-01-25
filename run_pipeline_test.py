from camera_input.image_source import ImageSource
from preprocessing.image_preprocessor import ImageProcessor
from inference.model_interface import ModelInterface
from inference.result_handler import ResultHandler
from logic.event_manager import EventManager
from communication.api_client import ApiClient
import time

# ---------------- CONFIG ----------------
BIN_ID = "BIN_001"
API_ENDPOINT = "https://ingestbinevent-t4vkrtxd5q-uc.a.run.app"
SEND_API_EVENTS = False   # ⚠️ KEEP FALSE unless app team confirms
# ---------------------------------------


def main():
    image_source = ImageSource("test_images")
    image_processor = ImageProcessor()
    model = ModelInterface()
    result_handler = ResultHandler()
    event_manager = EventManager(BIN_ID)
    api_client = ApiClient(API_ENDPOINT)

    # Simulated sensor values
    simulated_fill_levels = [25, 70, 100, 100, 0]

    print("\nStarting FULL bin pipeline simulation...\n")

    for fill_level in simulated_fill_levels:
        print(f"--- Fill Level: {fill_level}% ---")

        image, path = image_source.get_next_image()
        print(f"Image loaded: {path}")

        # -------------------------------
        # Vision + fallback handling
        # -------------------------------
        try:
            processed = image_processor.preprocess(image)
            predicted_class, confidence = model.predict(processed)
            final_class, _ = result_handler.handle(predicted_class, confidence)
            print(f"AI result: {final_class}")

        except ValueError as e:
            print(f"Preprocessing issue: {e}")
            print("Fallback → MIXED waste")
            final_class = "mixed"

        # -------------------------------
        # Event generation
        # -------------------------------
        events = event_manager.evaluate(
            sub_bin=final_class,
            fill_level=fill_level
        )

        for event in events:
            print(f"Event → {event['eventType']} | Fill: {event.get('fillLevel')}")
            if SEND_API_EVENTS:
                api_client.send_event(event)
            else:
                print("[API] Dry run (not sent)")

        print()
        time.sleep(1)

    print("Simulation completed.\n")


if __name__ == "__main__":
    main()
