class EventManager:
    def __init__(self, bin_id, full_threshold=90):
        self.bin_id = bin_id
        self.full_threshold = full_threshold
        self.last_state = "NORMAL"

    def evaluate(self, sub_bin, fill_level, hardware_error=None):
        """
        Returns a LIST of events to send (can be empty).
        """
        events = []

        # -------------------------------
        # 1️⃣ LEVEL UPDATE (ALWAYS SENT)
        # -------------------------------
        if fill_level is not None:
            events.append({
                "binId": self.bin_id,
                "subBin": sub_bin,
                "eventType": "LEVEL_UPDATE",
                "fillLevel": fill_level,
                "errorCode": None
            })

        # ---------------------------------
        # 2️⃣ HARDWARE ERROR (PRIORITY)
        # ---------------------------------
        if hardware_error is not None:
            if self.last_state != "ERROR":
                self.last_state = "ERROR"
                events.append({
                    "binId": self.bin_id,
                    "subBin": None,
                    "eventType": "HARDWARE_ERROR",
                    "fillLevel": None,
                    "errorCode": hardware_error
                })
            return events

        # -----------------------
        # 3️⃣ BIN FULL EVENT
        # -----------------------
        if fill_level >= self.full_threshold and self.last_state != "FULL":
            self.last_state = "FULL"
            events.append({
                "binId": self.bin_id,
                "subBin": sub_bin,
                "eventType": "BIN_FULL",
                "fillLevel": fill_level,
                "errorCode": None
            })
            return events

        # -------------------------
        # 4️⃣ BIN EMPTIED EVENT
        # -------------------------
        if self.last_state == "FULL" and fill_level == 0:
            self.last_state = "NORMAL"
            events.append({
                "binId": self.bin_id,
                "subBin": sub_bin,
                "eventType": "BIN_EMPTIED",
                "fillLevel": 0,
                "errorCode": None
            })

        return events
