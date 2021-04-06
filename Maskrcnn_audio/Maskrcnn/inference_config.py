from trainconfig import TrainConfig
import numpy as np

class InferenceConfig(TrainConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    POST_NMS_ROIS_INFERENCE = 2000
    DETECTION_MAX_INSTANCES = 2000


    # Minimum probability value to accept a detected instance
    # ROIs below this threshold are skipped
    DETECTION_MIN_CONFIDENCE = 0.4

    # Non-maximum suppression threshold for detection
    DETECTION_NMS_THRESHOLD = 0.4
    MEAN_PIXEL = np.array([0.,0.,0.])

inference_config = InferenceConfig()