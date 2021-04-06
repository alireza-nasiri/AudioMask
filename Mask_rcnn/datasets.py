import os
import time
import numpy as np
import skimage.io
import utils
import imageio
import skimage
import skimage.filters as f

class NucleiDataset(utils.Dataset):
    def load_nuclei(self, dataset_dir,image_num):
        """Load a subset of the COCO dataset.
        dataset_dir: The root directory of the COCO dataset.
        subset: What to load (train, val, minival, valminusminival)
        """

        image_dir = "{}/".format(dataset_dir)

        self.add_class("nuclei", 1, "cell")
        self.gt= {}
        # Add images
        for i in range(0, image_num):
            filename = str(i)
            self.add_image(
                "nuclei", image_id=i,
                path=os.path.join(image_dir, filename + '.png'),
                anno_path=os.path.join(image_dir, filename))

    def load_image(self, image_id):
        info = self.image_info[image_id]
        image = imageio.imread(info["path"])
        if len(image.shape) == 3:
            if image.shape[2] > 3:
                image = image[:,:,:3]
        else:
            image = skimage.color.gray2rgb(image)
            #print(image)
        #image = self.preprocess(image)
        image = image.astype('float32')
        return image


    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        path = self.image_info[image_id]['anno_path']
        annotations = os.listdir(path)
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        masks = []
        count = len(annotations)
        for annotation in annotations:
            mask = imageio.imread(path + '/' + annotation)
            mask = mask.astype('float32')
            masks.append(mask)
        masks = np.asarray(masks)
        masks[masks > 0.] = 1.
        masks = np.transpose(masks, (1, 2, 0))

        #occlusion = np.logical_not(masks[:, :, -1]).astype(np.uint8)
        #for i in range(count - 2, -1, -1):
        #    masks[:, :, i] = masks[:, :, i] * occlusion
        #    occlusion = np.logical_and(occlusion, np.logical_not(masks[:, :, i]))

        class_ids = np.ones(count)
        return masks, class_ids.astype(np.int32)

        #return self.gt[image_id][0], self.gt[image_id][1]

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "nuclei":
            return info["nuclei"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def preprocess(self, img):
        #if not (img[:,:,0] == img[:,:,1]).all():
        #    gray = skimage.color.rgb2gray(img.astype('uint8'))
        #    gray = 1-gray
        #else:
        #gray = skimage.color.rgb2gray(img.astype('uint8'))
        #img = skimage.color.gray2rgb(gray)
        img = f.gaussian(img, 1)
        img = 1 - img
        m = np.mean(img, axis = 1)
        img = np.transpose(img, (1,0,2))
        img = img - m
        img = np.transpose(img, (1, 0, 2))
        img *= 255.
        return img
