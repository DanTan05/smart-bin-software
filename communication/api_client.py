import requests
import json
from datetime import datetime


class ApiClient:
    def __init__(self, endpoint_url, timeout=5):
        """
        endpoint_url: Backend API endpoint
        timeout: HTTP timeout in seconds
        """
        self.endpoint_url = endpoint_url
        self.timeout = timeout
        self.headers = {
            "Content-Type": "application/json"
        }

    def send_event(self, event_payload):
        """
        Sends a bin event to the backend.

        event_payload must strictly follow the API contract.
        """

        if not isinstance(event_payload, dict):
            raise ValueError("Event payload must be a dictionary")

        try:
            response = requests.post(
                self.endpoint_url,
                headers=self.headers,
                data=json.dumps(event_payload),
                timeout=self.timeout
            )

            if response.status_code == 200:
                print(f"[API] Event sent successfully: {event_payload['eventType']}")
                return True
            else:
                print(
                    f"[API] Failed to send event "
                    f"(status={response.status_code}): {response.text}"
                )
                return False

        except requests.exceptions.RequestException as e:
            print(f"[API] Connection error: {e}")
            return False
