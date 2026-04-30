import logging

logger = logging.getLogger(__name__)

class ShotSegmentor:
    def segment_via_model(self, video_path):
        """Placeholder for future model-based segmentation."""
        logger.info("Running dummy model segmentation.")
        return [20, 60, 110] # Example hit frames

    def get_intervals(self, hit_frames):
        """Converts specific hits into start-end intervals."""
        if len(hit_frames) < 2: return []
        return [(hit_frames[i], hit_frames[i+1]) for i in range(len(hit_frames)-1)]