import numpy as np
import skimage.io as io
import os
import skimage.filters as f
from skimage import img_as_ubyte as ubyte
import shutil

base_dataset  = 'train_2500_gunshot_1'
sample_num = 2500

path = './' + base_dataset + '/'

des_path = './{0}_{1}/'.format('show_combine_1', base_dataset)

if not os.path.exists(des_path):
    os.mkdir(des_path)
else:
    shutil.rmtree(des_path)
    os.mkdir(des_path)
for i in range(0, sample_num, 10):
    combined_image = np.zeros((1280, 1305), np.uint8)
    combined_gt_image = np.zeros((1280, 1305), dtype=np.uint8)
    k = 0
    for j in range(10):
        img = io.imread(path + str(i+j) + '.png')
        combined_image[128*j:128*(j+1), :] = img
        gt = io.imread(path + str(i+j) + '/' + str(i+j) + '.png')
        combined_gt_image[128*j:128*(j+1), :] = gt
    img = combined_image

    # img = f.gaussian(img, 1) / 255.0
    # img = 1 - img
    # m = np.mean(img, axis=1)
    # img = np.transpose(img, (1, 0))
    # img = img - m
    # img[img < 0] = 0
    # img = np.transpose(img, (1, 0))
    # img = ubyte(img)

    io.imsave(des_path + str(int(i / 10)) + '.png', img)
    io.imsave(des_path + str(int(i/10)) + '_gt.png', combined_gt_image)



