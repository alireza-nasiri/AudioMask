import numpy as np
import skimage.io as io
import os
import skimage.filters as f
from skimage import img_as_ubyte as ubyte
import shutil

base_dataset  = 'val_1000_babycry_44100_128'
preprocessing = True
sample_num = 1000

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
for i in range(0, sample_num, 10):
    combined_image = np.zeros((1280, 1305))
    gt_dir = des_path + str(int(i/10))
    os.mkdir(gt_dir)
    k = 0
    for j in range(10):
        img = io.imread(path + str(i+j) + '.png')
        combined_image[128*j:128*(j+1), :] = img
        new_gt = np.zeros((1280, 1305), dtype=np.uint8)
        gt = io.imread(path + str(i+j) + '/' + str(i+j) + '.png')
        if np.sum(gt) > 0:
            new_gt[128*j:128*(j+1), :] = gt
            io.imsave(des_path + str(int(i/10)) + '/' + str(k) + '.png', new_gt)
            k += 1
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



