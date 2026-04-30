import os
import logging
from gradio_client import Client, handle_file

logger = logging.getLogger(__name__)

class ShuttleDetector:
    def __init__(self):
        # Space ID should be in your .env or HF Secrets
        self.space_id = os.getenv("HF_SHUTTLE_SPACE", "briii6/shuttle_detactor")
        self.token = os.getenv("HF_TOKEN")
        self.client = Client(self.space_id)

    def detect(self, video_path):
        """Calls the HF Space API to get shuttle coordinates."""
        logger.info(f"Sending video to HF Space: {self.space_id}")
        try:
            # Matches the api_name in your z4.py and app.py snippets
            result = self.client.predict(
                video_file=handle_file(video_path),
                api_name="/detect_shuttlecock"
            )
            return result
        except Exception as e:
            logger.error(f"Shuttle API Error: {e}")
            return None