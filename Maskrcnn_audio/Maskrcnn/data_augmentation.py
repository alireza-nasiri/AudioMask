import os
import imageio
import numpy as np
from skimage.transform import rotate
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage import distance_transform_edt
from scipy.ndimage.filters import gaussian_filter
import scipy.ndimage
import cv2
import random

source_path = '/home/easycui/HU/kaggle_2018/dataset/kaggle_2018_data_science/clean_transform2gray_reverse/train/'
dest_path = '/home/easycui/RCI/data/kaggle_2018/dataset/kaggle_2018_data_science/clean_transform2gray_reverse_argument/train/'



def elastic_transform(img, gts, alpha, sigma, alpha_affine, random_state=None):
    # the img and gt must be in uint8 data type
    if random_state is None:
        random_state = np.random.RandomState(None)
    img_shape = img.shape
    shape_size = img_shape
    # print(shape)
    # print(img)
    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size],
                       center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)

    img = cv2.warpAffine(img, M, shape_size[::-1], borderMode=cv2.BORDER_CONSTANT)
    img = np.reshape(img, img_shape)
    for i in range(gts.shape[2]):
        gts[:,:,i] = cv2.warpAffine(gts[:,:,i], M, shape_size[::-1], borderMode=cv2.BORDER_CONSTANT)
    ax = random_state.rand(*shape_size) * 2 - 1
    ay = random_state.rand(*shape_size) * 2 - 1

    # print(img_shape, gt_shape)

    dx = gaussian_filter(ax, sigma) * alpha
    dy = gaussian_filter(ay, sigma) * alpha
    x, y = np.meshgrid(np.arange(img_shape[1]), np.arange(img_shape[0]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
    img = map_coordinates(img, indices, order=1, mode='constant').reshape(img_shape)

    gt_shape = gts.shape


    dx = gaussian_filter(ax, sigma) * alpha
    dy = gaussian_filter(ay, sigma) * alpha
    x, y= np.meshgrid(np.arange(gt_shape[1]), np.arange(gt_shape[0]))
    indices_gt = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
    for i in range(gts.shape[2]):
        gts[:,:,i] = map_coordinates(gts[:,:,i], indices_gt, order=1, mode='constant').reshape(gt_shape[:2])
    gts[gts > 100] = 255
    gts[gts <= 100] = 0
    #print(np.max(gts))
    return img, gts


def augment(path, image_num):
    image = imageio.imread(path + str(image_num) + '.png')
    if len(image.shape) == 3:
        image = image[:,:,0]
    gts = []
    for file in os.listdir(path + str(image_num)):
        if file.endswith('.png'):
            gt_image = imageio.imread(path + str(image_num) + '/' + file)
            #print(gt_image.shape)
            gts.append(gt_image)
    gts = np.asarray(gts)
    gts = np.transpose(gts, (1,2, 0))
    img, gts = elastic_transform(image, gts, 200, 12, 10)
    return img, gts

#img = imageio.imread('../dataset/kaggle_2018_data_science/clean_transform2gray_reverse_argument/train/1548/11.png')
#print(np.sum(img))



i = 0
index_img = 1
while index_img < 5000:
    print(i)
    if os.path.exists(dest_path + str(index_img)):
        pass
    else:
        os.mkdir(dest_path + str(index_img))
    img, gts = augment(source_path, i%488+1)
    angle = int(random.random() * 360)
    img = (rotate(img, angle, order = 0) * 255).astype(np.uint8)
    gts = (rotate(gts, angle, order = 0) * 255).astype(np.uint8)
    gts[gts > 100] = 255
    gts[gts <= 100] = 0

    j = 0
    index = 1
    scale = min(512.0 / gts.shape[0], 512.0 / gts.shape[1])
    mask = scipy.ndimage.zoom(gts, zoom=[scale, scale, 1], order=0)
    while j < gts.shape[2]:
        if np.sum(mask[:,:,j]) > 255*9:
            imageio.imwrite(dest_path + str(index_img) + '/' + str(index) + '.png', gts[:,:,j])
            index += 1
        j += 1

    if index > 1:
        imageio.imwrite(dest_path + str(index_img) + '.png', img)
        index_img += 1
    i += 1
