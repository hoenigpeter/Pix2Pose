import os
import skimage
from skimage.filters import gaussian
from skimage import io
from skimage.transform import resize,rotate
from skimage.filters import gaussian
from imgaug import augmenters as iaa
import imgaug.augmenters as iaa  # noqa
from imgaug.augmenters import (Sequential, SomeOf, OneOf, Sometimes, WithColorspace, WithChannels, Noop,
                                Lambda, AssertLambda, AssertShape, Scale, CropAndPad, Pad, Crop, Fliplr,
                                Flipud, Superpixels, ChangeColorspace, PerspectiveTransform, Grayscale,
                                GaussianBlur, AverageBlur, MedianBlur, Convolve, Sharpen, Emboss, EdgeDetect,
                                DirectedEdgeDetect, Add, AddElementwise, AdditiveGaussianNoise, Multiply,
                                MultiplyElementwise, Dropout, CoarseDropout, Invert, ContrastNormalization,
                                Affine, PiecewiseAffine, ElasticTransformation, pillike, LinearContrast)  # noqa

from matplotlib import pyplot as plt

import numpy as np
import random
import cv2

import torch
from torch.utils.data import Dataset, DataLoader

class data_generator():
    def __init__(self,data_dir,rgb_dir,xyz_dir,batch_size=50,gan=True,imsize=128,
                 res_x=640,res_y=480,prob=1.0,
                 **kwargs):
        '''
        data_dir: Folder that contains cropped image+xyz
        back_dir: Folder that contains random background images
            batch_size: batch size for training
        gan: if False, gt for GAN is not yielded
        '''
        self.data_dir = data_dir
        self.rgb_dir = rgb_dir
        self.xyz_dir = xyz_dir
        self.imsize=imsize
        self.batch_size = batch_size
        self.gan = gan

        #data_list = os.listdir(data_dir)
        rgb_data_list = os.listdir(rgb_dir)
        xyz_data_list = os.listdir(xyz_dir)
        self.rgb_datafiles=[]
        self.xyz_datafiles=[]
        self.res_x=res_x
        self.res_y=res_y

        for file in rgb_data_list:
            if(file.endswith(".png")):
                self.rgb_datafiles.append(file)
        for file in xyz_data_list:
            if(file.endswith(".png")):
                self.xyz_datafiles.append(file)

        self.n_data = len(self.rgb_datafiles)
        self.indices = np.arange(self.n_data)
        self.idx = 0

        print("Total training views:", self.n_data)

        self.seq_syn= iaa.Sequential([        
                                    Sometimes(0.5 * prob, CoarseDropout( p=0.2, size_percent=0.05) ),
                                    Sometimes(0.5 * prob, GaussianBlur(1.2*np.random.rand())),
                                    Sometimes(0.5 * prob, Add((-25, 25), per_channel=0.3)),
                                    Sometimes(0.3 * prob, Invert(0.2, per_channel=True)),
                                    Sometimes(0.5 * prob, Multiply((0.6, 1.4), per_channel=0.5)),
                                    Sometimes(0.5 * prob, Multiply((0.6, 1.4))),
                                    Sometimes(0.5 * prob, LinearContrast((0.5, 2.2), per_channel=0.3))
                                    ], random_order = False)
        
        # self.seq_syn= iaa.Sequential([
        #                             iaa.WithChannels(0, iaa.Add((-15, 15))),
        #                             iaa.WithChannels(1, iaa.Add((-15, 15))),
        #                             iaa.WithChannels(2, iaa.Add((-15, 15))),
        #                             iaa.ContrastNormalization((0.8, 1.3)),
        #                             iaa.Multiply((0.8, 1.2),per_channel=0.5),
        #                             iaa.GaussianBlur(sigma=(0.0, 0.5)),
        #                             iaa.Sometimes(0.1, iaa.AdditiveGaussianNoise(scale=10, per_channel=True)),
        #                             iaa.Sometimes(0.5, iaa.ContrastNormalization((0.5, 2.2), per_channel=0.3)),
        #                             ], random_order=True)

    def get_patch_pair(self,v_id):
        #imgs = np.load(os.path.join(self.data_dir,self.datafiles[v_id])).astype(np.float32)

        rgb_imgs = cv2.imread(os.path.join(self.rgb_dir,self.rgb_datafiles[v_id]))
        xyz_imgs = cv2.imread(os.path.join(self.xyz_dir,self.xyz_datafiles[v_id]))

        rgb_imgs = rgb_imgs[...,::-1].astype(np.float32)
        xyz_imgs = xyz_imgs[...,::-1].astype(np.float32)

        real_img =rgb_imgs/255
        p_xyz =xyz_imgs/255

        p_height = p_xyz.shape[0]
        p_width = p_xyz.shape[1]
        p_mask_no_occ = np.sum(p_xyz,axis=2)>0
        p_xyz[np.invert(p_mask_no_occ)]=[0.5,0.5,0.5]

        real_img_uint8 = np.array((real_img * 255).clip(0, 255), dtype=np.uint8)
        img_augmented = self.seq_syn.augment_image(real_img_uint8) / 255

        base_image = img_augmented
        tgt_image = p_xyz

        #mask_area_crop = rotate(p_mask_no_occ.astype(np.float),r_angle)
        mask_area_crop = p_mask_no_occ
        # Calculate scaling factors for maintaining aspect ratio
        scaling_factor = min(self.imsize / real_img.shape[0], self.imsize / real_img.shape[1])

        # Calculate new dimensions after maintaining aspect ratio
        new_height = int(real_img.shape[0] * scaling_factor)
        new_width = int(real_img.shape[1] * scaling_factor)

        # Calculate the amount of padding needed to center the object
        pad_height_top = max(0, (self.imsize - new_height) // 2)
        pad_height_bottom = self.imsize - new_height - pad_height_top
        pad_width_left = max(0, (self.imsize - new_width) // 2)
        pad_width_right = self.imsize - new_width - pad_width_left

        # Resize base_image, tgt_image, and mask_image
        src_image_resized = resize(base_image, (new_height, new_width), order=1, mode='reflect')
        tgt_image_resized = resize(tgt_image, (new_height, new_width), order=1, mode='reflect')
        mask_area_resized = resize(mask_area_crop.astype(np.float), (new_height, new_width), order=1, mode='reflect')

        # Create a new canvas of the desired size (128x128) and fill with gray
        src_image_final = np.full((self.imsize, self.imsize, 3), 0.5, dtype=np.float32)
        tgt_image_final = np.full((self.imsize, self.imsize, 3), 0.5, dtype=np.float32)
        mask_area_final = np.full((self.imsize, self.imsize), 0, dtype=np.float32)

        # Place the resized images at the center of the canvas
        src_image_final[pad_height_top:pad_height_top + new_height, pad_width_left:pad_width_left + new_width] = src_image_resized
        tgt_image_final[pad_height_top:pad_height_top + new_height, pad_width_left:pad_width_left + new_width] = tgt_image_resized
        mask_area_final[pad_height_top:pad_height_top + new_height, pad_width_left:pad_width_left + new_width] = mask_area_resized

        src_image_final = src_image_final.astype(np.float64)
        src_image_final = (src_image_final - 0.5) * 2.0

        tgt_image_final = tgt_image_final.astype(np.float64)
        tgt_image_final = (tgt_image_final - 0.5) * 2.0

        mask_area_final = mask_area_final.astype(np.float64)

        return src_image_final,tgt_image_final,mask_area_final    

    def generator(self):
        while True:
            if self.idx >= self.n_data:
                self.idx = 0
                np.random.shuffle(self.indices)

            batch_indices = self.indices[self.idx:self.idx + self.batch_size]
            self.idx += self.batch_size

            batch_src = np.zeros((len(batch_indices), self.imsize, self.imsize, 3))
            batch_tgt = np.zeros((len(batch_indices), self.imsize, self.imsize, 3))

            for i, idx in enumerate(batch_indices):
                v_id = idx
                s_img, t_img, mask_area = self.get_patch_pair(v_id)
                batch_src[i] = s_img
                batch_tgt[i] = t_img

            batch_tgt_disc = np.zeros(len(batch_indices))
            batch_tgt_disc[:] = 1
            batch_prob = np.zeros((len(batch_indices), self.imsize, self.imsize, 1))
            yield batch_src, batch_tgt, batch_tgt_disc, batch_prob

class CustomDataset(Dataset):
    def __init__(self, data_dir, rgb_dir, xyz_dir, batch_size=50, gan=True, imsize=128, res_x=640, res_y=480, prob=1.0):
        self.data_dir = data_dir
        self.rgb_dir = rgb_dir
        self.xyz_dir = xyz_dir
        self.imsize = imsize
        self.batch_size = batch_size
        self.gan = gan
        self.prob = prob

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.rgb_data_list = [file for file in os.listdir(rgb_dir) if file.endswith(".png")]
        self.xyz_data_list = [file for file in os.listdir(xyz_dir) if file.endswith(".png")]

        self.data = list(zip(self.rgb_data_list, self.xyz_data_list))
        self.n_data = len(self.data)
        self.indices = list(range(self.n_data))
        self.seq_syn= iaa.Sequential([        
                                    Sometimes(0.5 * prob, CoarseDropout( p=0.2, size_percent=0.05) ),
                                    Sometimes(0.5 * prob, GaussianBlur(1.2*np.random.rand())),
                                    Sometimes(0.5 * prob, Add((-25, 25), per_channel=0.3)),
                                    Sometimes(0.3 * prob, Invert(0.2, per_channel=True)),
                                    Sometimes(0.5 * prob, Multiply((0.6, 1.4), per_channel=0.5)),
                                    Sometimes(0.5 * prob, Multiply((0.6, 1.4))),
                                    Sometimes(0.5 * prob, LinearContrast((0.5, 2.2), per_channel=0.3))
                                    ], random_order = False)

    def shuffle_indices(self):
        np.random.shuffle(self.indices)
        
    def __len__(self):
        return self.n_data // self.batch_size

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size: (idx + 1) * self.batch_size]

        batch_src = []
        batch_tgt = []
        batch_tgt_disc = []
        prob_gt = []

        for i in batch_indices:
            v_id = i
            s_img, t_img, mask_gt = self.get_patch_pair(v_id)
            batch_src.append(s_img)
            batch_tgt.append(t_img)
            prob_gt.append(mask_gt)
            batch_tgt_disc.append(1)

        src_tensor = torch.stack([torch.from_numpy(img).to(self.device) for img in batch_src])
        tgt_tensor = torch.stack([torch.from_numpy(img).to(self.device) for img in batch_tgt])
        prob_gt_tensor = torch.stack([torch.from_numpy(img).to(self.device) for img in prob_gt])
        tgt_disc_tensor = torch.tensor(batch_tgt_disc, dtype=torch.float32).to(self.device)

        src_tensor = src_tensor.permute(0, 3, 1, 2) 
        tgt_tensor = tgt_tensor.permute(0, 3, 1, 2)
        prob_gt_tensor = prob_gt_tensor.unsqueeze(3)
        prob_gt_tensor = prob_gt_tensor.permute(0, 3, 1, 2) 

        # print("src_tensor: ", src_tensor.shape)
        # print("tgt_tensor: ", tgt_tensor.shape)
        # print("prob_gt_tensor: ", prob_gt_tensor.shape)
        # print("tgt_disc_tensor: ", tgt_disc_tensor.shape)

        return src_tensor, tgt_tensor, tgt_disc_tensor, prob_gt_tensor

    def get_patch_pair(self,v_id):
        rgb_imgs = cv2.imread(os.path.join(self.rgb_dir,self.rgb_data_list[v_id]))
        xyz_imgs = cv2.imread(os.path.join(self.xyz_dir,self.xyz_data_list[v_id]))

        rgb_imgs = rgb_imgs[...,::-1].astype(np.float32)
        xyz_imgs = xyz_imgs[...,::-1].astype(np.float32)

        real_img =rgb_imgs/255
        p_xyz =xyz_imgs/255

        p_height = p_xyz.shape[0]
        p_width = p_xyz.shape[1]
        p_mask_no_occ = np.sum(p_xyz,axis=2)>0
        p_xyz[np.invert(p_mask_no_occ)]=[0.5,0.5,0.5]

        real_img_uint8 = np.array((real_img * 255).clip(0, 255), dtype=np.uint8)
        img_augmented = self.seq_syn.augment_image(real_img_uint8) / 255

        base_image = img_augmented
        tgt_image = p_xyz

        #mask_area_crop = rotate(p_mask_no_occ.astype(np.float),r_angle)
        mask_area_crop = p_mask_no_occ
        # Calculate scaling factors for maintaining aspect ratio
        scaling_factor = min(self.imsize / real_img.shape[0], self.imsize / real_img.shape[1])

        # Calculate new dimensions after maintaining aspect ratio
        new_height = int(real_img.shape[0] * scaling_factor)
        new_width = int(real_img.shape[1] * scaling_factor)

        # Calculate the amount of padding needed to center the object
        pad_height_top = max(0, (self.imsize - new_height) // 2)
        pad_height_bottom = self.imsize - new_height - pad_height_top
        pad_width_left = max(0, (self.imsize - new_width) // 2)
        pad_width_right = self.imsize - new_width - pad_width_left

        # Resize base_image, tgt_image, and mask_image
        src_image_resized = resize(base_image, (new_height, new_width), order=1, mode='reflect')
        tgt_image_resized = resize(tgt_image, (new_height, new_width), order=1, mode='reflect')
        mask_area_resized = resize(mask_area_crop.astype(np.float), (new_height, new_width), order=1, mode='reflect')

        # Create a new canvas of the desired size (128x128) and fill with gray
        src_image_final = np.full((self.imsize, self.imsize, 3), 0.5, dtype=np.float32)
        tgt_image_final = np.full((self.imsize, self.imsize, 3), 0.5, dtype=np.float32)
        mask_area_final = np.full((self.imsize, self.imsize), 0, dtype=np.float32)

        # Place the resized images at the center of the canvas
        src_image_final[pad_height_top:pad_height_top + new_height, pad_width_left:pad_width_left + new_width] = src_image_resized
        tgt_image_final[pad_height_top:pad_height_top + new_height, pad_width_left:pad_width_left + new_width] = tgt_image_resized
        mask_area_final[pad_height_top:pad_height_top + new_height, pad_width_left:pad_width_left + new_width] = mask_area_resized

        src_image_final = src_image_final.astype(np.float32)
        src_image_final = (src_image_final - 0.5) * 2.0

        tgt_image_final = tgt_image_final.astype(np.float32)
        tgt_image_final = (tgt_image_final - 0.5) * 2.0

        mask_area_final = mask_area_final.astype(np.float32)

        return src_image_final,tgt_image_final,mask_area_final 