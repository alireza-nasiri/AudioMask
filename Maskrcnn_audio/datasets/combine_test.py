import numpy as np
import skimage.io as io
import os
import skimage.filters as f
from skimage import img_as_ubyte as ubyte
import shutil
import skimage.filters as f

base_dataset  = 'test_set_3_babycry'
preprocessing = True
sample_num = 500

path = './' + base_dataset + '/'
if preprocessing:
    des_path = './{0}_{1}_{2}/'.format('combine', base_dataset, 'preprocessing')
else:
    des_path = './{0}_{1}/'.format('combine', base_dataset)

if not os.path.exists(des_path):
    os.mkdir(des_path)
else:
    shutil.rmtree(des_path)
    os.mkdir(des_path)

i = 0
files = os.listdir(path)
files = sorted(files)
mapping = open(base_dataset + 'filename_mapping.txt', 'w')
while i < len(files):
    combined_image = np.zeros((1280, 1305))
    for j in range(10):
        img = io.imread(path + files[i+j])
        combined_image[128*j:128*(j+1), :] = img
        mapping.write(files[i+j] + ',' + str(int(i/10)) + '\n')
    img = combined_image
    if preprocessing:
        img = f.gaussian(img, 1)/255.0
        img = 1 - img
        m = np.mean(img, axis=1)
        img = np.transpose(img, (1, 0))
        img = img - m
        img[img < 0] = 0
        img = np.transpose(img, (1, 0))
        img = ubyte(img)
    else:
        img = img/255.0
        img = ubyte(img)

    io.imsave(des_path + str(int(i / 10)) + '.png', img)
    i+=10
mapping.close()



