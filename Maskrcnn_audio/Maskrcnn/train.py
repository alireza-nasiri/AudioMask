
import os
import time
import numpy as np
import skimage.io
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
import datetime
# Download and install the Python COCO tools from https://github.com/waleedka/coco
# That's a fork from the original https://github.com/pdollar/coco with a bug
# fix for Python 3.
# I submitted a pull request https://github.com/cocodataset/cocoapi/pull/50
# If the PR is merged then use the original repo.
# Note: Edit PythonAPI/Makefile and replace "python" with "python3".
import shutil
from  trainconfig import *
from datasets import *
import utils
from  MaskRCNN import *
import sys
# Root directory of the project



    # The following two functions are from pycocotools with a few changes.

############################################################
#  Training
############################################################


def train(train_dataset='', val_dataset='', pretrain_model='', logs_path='', num_train=300,
          num_val=50, epochs = 50):
    if pretrain_model == '':
        ROOT_DIR = os.getcwd()
        # Path to trained weights file
        model_path = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
    else:
        model_path = pretrain_model

    if os.path.exists(logs_path):
        pass
    else:
        os.mkdir(logs_path)

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    set_session(tf.Session(config=tf_config))
    # Configurations
    config = TrainConfig()
    # Create model
    model = MaskRCNN(mode="training", config=config, model_dir=logs_path)
    print(model.log_dir)
    log_path = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
    log_path = os.path.join(logs_path, log_path)

    if os.path.exists(log_path):
        pass
    else:
        os.mkdir(log_path)

    with open(os.path.join(log_path, 'configuration.txt'), 'w') as f:
        orig_stdout = sys.stdout
        sys.stdout = f
        print('train_image_path:', train_dataset)
        print('pretrain_model:', pretrain_model)
        config.display()
        shutil.copyfile('config.py', os.path.join(log_path, 'config.py'))
        shutil.copyfile('utils.py', os.path.join(log_path, 'utils.py'))
        shutil.copyfile('model.py', os.path.join(log_path, 'model.py'))
        shutil.copyfile('MaskRCNN.py', os.path.join(log_path, 'MaskRCNN.py'))
        shutil.copyfile('train.py', os.path.join(log_path, 'train.py'))
        shutil.copyfile('run_train.py', os.path.join(log_path, 'run_train.py'))
        sys.stdout = orig_stdout

    # Load weights
    print("Loading weights ", model_path)
    model.load_weights(model_path, by_name=True,
    exclude = ["mrcnn_class_logits", "mrcnn_bbox_fc",
               "mrcnn_bbox", "mrcnn_mask"])

    # Training dataset. Use the training set and 35K from the
    # validation set, as as in the Mask RCNN paper.
    with open(os.path.join(log_path, 'train_log.txt'), 'w', 1) as f:
        orig_stdout = sys.stdout
        sys.stdout = f
        dataset_train = NucleiDataset()
        dataset_train.load_nuclei(train_dataset, num_train)
        dataset_train.prepare()

        # Validation dataset
        dataset_val = NucleiDataset()
        dataset_val.load_nuclei(val_dataset, num_val)
        dataset_val.prepare()

        # *** This training schedule is an example. Update to your needs ***

        # Training - Stage 1
        print('********************************************')
        print('*                                          *')
        print("Training network heads")
        print('*                                          *')
        print('********************************************')
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=epochs,
                    layers='all')


        sys.stdout = orig_stdout





