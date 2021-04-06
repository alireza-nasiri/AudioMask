from config import Config
import numpy as np
class TrainConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "bowl2018"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1 # background + nuclei

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64)  # anchor side in pixels
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]
    TRAIN_ROIS_PER_IMAGE = 256
    STEPS_PER_EPOCH = 125
    # use small validation steps since the epoch is small

    VALIDATION_STEPS = 50
    RPN_NMS_THRESHOLD = 0.4
    RPN_TRAIN_ANCHORS_PER_IMAGE =256
    LEARNING_RATE = 0.01
    USE_MINI_MASK = True
    MEAN_PIXEL = np.array([0,0,0])
    VERBOSE = 2
    MULTI = True
