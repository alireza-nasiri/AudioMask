import model as modellib
import os
import numpy as np
#from tqdm import tqdm
from inference_config import inference_config
#from utils import rle_encode, rle_decode, rle_to_string
import visualize
from datasets import *
import MaskRCNN
import skimage
import skimage.color
import warnings
import imageio
import utils
import sys
import shutil
from tqdm import tqdm

def get_filename_map(filemap):
    d = {}
    with open(filemap, 'r') as f:
        i = 0
        for line in f.readlines():
            j = i % 10
            filename, index = line.split(',')
            d[(int(index), j)] = filename
            i += 1
    return d

def predict(test_image_path, result_dir, model_path='', event = '', generate_prediction_image=True,
            generate_prediction_rle=True, submission_file='submission.csv'):
    ROOT_DIR = os.getcwd()
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")

    # Recreate the model in inference mode
    model = MaskRCNN.MaskRCNN(mode="inference",
                              config=inference_config,
                              model_dir=MODEL_DIR)
    inference_config.display()
    # Get path to saved weights
    # Either set a specific path or find last trained weights
    # model_path = os.path.join(ROOT_DIR, ".h5 file name here")
    if model_path == '':
        model_path = model.find_last()[1]
    # Load trained weights (fill in path to trained weights here)
    assert model_path != "", "Provide path to trained weights"
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)
    if os.path.exists(result_dir):
        pass
    else:
        os.mkdir(result_dir)

    with open(os.path.join(result_dir, 'configuration.txt'), 'w') as f:
        orig_stdout = sys.stdout
        sys.stdout = f
        print('test_image_path:', test_image_path)
        print('model_path:', model_path)
        inference_config.display()
        shutil.copyfile('config.py', os.path.join(result_dir, 'config.py'))
        shutil.copyfile('utils.py', os.path.join(result_dir, 'utils.py'))
        shutil.copyfile('model.py', os.path.join(result_dir, 'model.py'))
        shutil.copyfile('MaskRCNN.py', os.path.join(result_dir, 'MaskRCNN.py'))
        sys.stdout = orig_stdout

    image_ids = []
    
    #for devtest use this:
    #d = get_filename_map('../datasets/test_500_' + event + 'filename_mapping.txt')
    
    #for evaluation use this:
    d = get_filename_map('../datasets/test_set_3_gunshotfilename_mapping.txt')   
    f = open(submission_file, 'w')
    for file in tqdm(os.listdir(test_image_path)):
        if file.endswith('.png'):
            
            
            image_path = os.path.join(test_image_path, file)
            image_id = os.path.splitext(file)[0]
            image_ids.append(image_id)
            i = int(image_id)
            original_image = imageio.imread(image_path)
            #print(original_image)
            if len(original_image.shape) == 2:
                original_image = skimage.color.gray2rgb(original_image)
                
            results = model.detect([original_image], verbose=0)
            r = results[0]

            masks = r['masks']
            #pick = utils.non_max_suppression_mask(masks, r['scores'], 0.3)
            #masks = masks[:, :, pick]
            #rois = r['rois'][pick]
            #class_ids = r['class_ids'][pick]
            #scores = r['scores'][pick]
            
            '''
            #This is the part of the code that would only write the most probable ROI into the text file
            for j in range(10):
                pick = -1
                curmax = 0
                for k in range(r['rois'].shape[0]):
                    #print(r['rois'][k])
                    if j*128 < r['rois'][k][0] < j*128 and r['scores'][k] > curmax:
                        pick = k
                        curmax = r['scores'][k]
                if pick > -1:
                    f.write(d[(i, j)][:-4] + ';' + str(r['rois'][pick][1]/1305.0*30) + ';'
                            +str(r['rois'][pick][3]/1305.0*30) + ';' + event + '\n')
                else:
                    f.write(d[(i, j)][:-4] + ';' + '0.0;' + '0.0;' + 'no event' + '\n')
            '''
            for j in range(10):
                pick = []
                pick.clear()
                
                curmax = 0
                for k in range(r['rois'].shape[0]):
                    #print(r['rois'][k])
                    if j*128 - 20 < r['rois'][k][0] < j*128 + 20 and r['scores'][k] > curmax:
                        pick.append(k)
                        #scores.append(r['scores'][k])
                        #curmax = r['scores'][k]
                if pick:
                    for ind in pick:
                        f.write(d[(i, j)][:-4] + ';' + str(r['rois'][ind][1]/1305.0*30) + ';'
                            +str(r['rois'][ind][3]/1305.0*30) + ';' + event +';'+ str(r['scores'][ind])+ '\n')
                else:
                    f.write(d[(i, j)][:-4] + ';' + '0.0;' + '0.0;' + 'no event' + '\n')
            
            if generate_prediction_image:
                visualize.display_instances((original_image).astype(np.uint8), r['rois'], masks, r['class_ids'],
                                            ['bg', 'cell'], r['scores'] , filename = os.path.join(result_dir, file))
                #raw_predictions.append((masks, scores))

    f.close()


