import numpy as np
import skimage.io as io
import os
import skimage.filters as f
from skimage import img_as_ubyte as ubyte
import shutil
import skimage.filters as f

base_dataset  = 'test_500_glassbreak_ground_truth'
preprocessing = False
sample_num = 500

path = './' + base_dataset + '/'
des_path = './{0}_{1}/'.format('combine', base_dataset)

if not os.path.exists(des_path):
    os.mkdir(des_path)
else:
    shutil.rmtree(des_path)
    os.mkdir(des_path)

i = 0
files = os.listdir(path)
files = sorted(files)
while i < len(files):
    combined_image = np.zeros((1280, 1305), dtype=np.uint8)
    for j in range(10):
        img = io.imread(path + files[i+j])
        combined_image[128*j:128*(j+1), :] = img
    img = combined_image
    io.imsave(des_path + str(int(i / 10)) + '.png', img)
    i+=10



