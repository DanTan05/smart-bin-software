class EventManager:
    def __init__(self, bin_id):
        self.bin_id = bin_id
        self.last_state = "NORMAL"

    def evaluate(self, sub_bin, fill_level, hardware_error=None, piece_collected=False):
        """
        Returns a LIST of events to send (can be empty).

        Rules:
        - PIECE_COLLECTED is emitted per item (if enabled)
        - LEVEL_UPDATE only for 1–99%
        - BIN_FULL only at 100%
        - BIN_EMPTIED resets state
        - HARDWARE_ERROR triggers alert only
        """
        events = []


        # ---------------------------------
        # 1️⃣ HARDWARE ERROR (ALERT ONLY)
        # ---------------------------------
        if hardware_error is not None:
            if self.last_state != "ERROR":
                self.last_state = "ERROR"
                events.append({
                    "binId": self.bin_id,
                    "eventType": "HARDWARE_ERROR",
                    "errorCode": hardware_error
                })
            return events

        # -------------------------
        # 2️⃣ BIN EMPTIED (RESET)
        # -------------------------
        if fill_level == 0 and self.last_state in ["FULL", "ERROR"]:
            self.last_state = "NORMAL"
            events.append({
                "binId": self.bin_id,
                "subBin": sub_bin,
                "eventType": "BIN_EMPTIED",
                "fillLevel": 0
            })
            return events

        # -----------------------
        # 3️⃣ BIN FULL (ALERT)
        # -----------------------
        if fill_level >= 100 and self.last_state != "FULL":
            self.last_state = "FULL"
            events.append({
                "binId": self.bin_id,
                "subBin": sub_bin,
                "eventType": "BIN_FULL",
                "fillLevel": 100
            })
            return events

        # -----------------------
        # 4️⃣ LEVEL UPDATE (1–99)
        # -----------------------
        if 0 < fill_level < 100:
            events.append({
                "binId": self.bin_id,
                "subBin": sub_bin,
                "eventType": "LEVEL_UPDATE",
                "fillLevel": fill_level
            })

        return events
