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

class data_generator():
    def __init__(self,data_dir,batch_size=50,gan=True,imsize=128,
                 res_x=640,res_y=480,prob=1.0,
                 **kwargs):
        '''
        data_dir: Folder that contains cropped image+xyz
        back_dir: Folder that contains random background images
            batch_size: batch size for training
        gan: if False, gt for GAN is not yielded
        '''
        self.data_dir = data_dir
        self.imsize=imsize
        self.batch_size = batch_size
        self.gan = gan
        data_list = os.listdir(data_dir)
        self.datafiles=[]
        self.res_x=res_x
        self.res_y=res_y

        for file in data_list:
            if(file.endswith(".npy")):
                self.datafiles.append(file)

        self.n_data = len(self.datafiles)
        print("Total training views:", self.n_data)

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
        
        # self.seq_syn= iaa.Sequential([
        #                             # iaa.Sometimes(0.5, PerspectiveTransform(0.05)),
        #                             # iaa.Sometimes(0.5, CropAndPad(percent=(-0.05, 0.1))),
        #                             # iaa.Sometimes(0.5, Affine(scale=(1.0, 1.2))),
        #                             Sometimes(0.5 * prob, CoarseDropout( p=0.2, size_percent=0.05) ),
        #                             Sometimes(0.4 * prob, GaussianBlur((0., 3.))),
        #                             Sometimes(0.3 * prob, pillike.EnhanceSharpness(factor=(0., 50.))),
        #                             Sometimes(0.3 * prob, pillike.EnhanceContrast(factor=(0.2, 50.))),
        #                             Sometimes(0.5 * prob, pillike.EnhanceBrightness(factor=(0.1, 6.))),
        #                             Sometimes(0.3 * prob, pillike.EnhanceColor(factor=(0., 20.))),
        #                             Sometimes(0.5 * prob, Add((-25, 25), per_channel=0.3)),
        #                             Sometimes(0.3 * prob, Invert(0.2, per_channel=True)),
        #                             Sometimes(0.5 * prob, Multiply((0.6, 1.4), per_channel=0.5)),
        #                             Sometimes(0.5 * prob, Multiply((0.6, 1.4))),
        #                             Sometimes(0.1 * prob, AdditiveGaussianNoise(scale=10, per_channel=True)),
        #                             Sometimes(0.5 * prob, iaa.contrast.LinearContrast((0.5, 2.2), per_channel=0.3)),
        #                             Sometimes(0.5 * prob, Grayscale(alpha=(0.0, 1.0))),
        #                             ], random_order=True)

        self.seq_syn= iaa.Sequential([        
                                    Sometimes(0.5 * prob, CoarseDropout( p=0.2, size_percent=0.05) ),
                                    Sometimes(0.5 * prob, GaussianBlur(1.2*np.random.rand())),
                                    Sometimes(0.5 * prob, Add((-25, 25), per_channel=0.3)),
                                    Sometimes(0.3 * prob, Invert(0.2, per_channel=True)),
                                    Sometimes(0.5 * prob, Multiply((0.6, 1.4), per_channel=0.5)),
                                    Sometimes(0.5 * prob, Multiply((0.6, 1.4))),
                                    Sometimes(0.5 * prob, LinearContrast((0.5, 2.2), per_channel=0.3))
                                    ], random_order = False)

    def get_patch_pair(self,v_id,batch_count):
        imgs = np.load(os.path.join(self.data_dir,self.datafiles[v_id])).astype(np.float32)
        is_real=False
        if imgs.shape[2]==7:
            #this is real image
            p_vis_mask = imgs[:,:,6]>0
            is_real=True
            
        real_img =imgs[:,:,:3]/255
        p_xyz =imgs[:,:,3:6]/255

        p_height = p_xyz.shape[0]
        p_width = p_xyz.shape[1]
        p_mask_no_occ = np.sum(p_xyz,axis=2)>0
        p_xyz[np.invert(p_mask_no_occ)]=[0.5,0.5,0.5]

        # plt.imshow(p_xyz)
        # plt.title('before resizing')
        # plt.show()
        real_img_uint8 = np.array((real_img * 255).clip(0, 255), dtype=np.uint8)
        img_augmented = self.seq_syn.augment_image(real_img_uint8) / 255

        #rotate -15 to 15 degree
        #random flip and rotation is also possible --> good for you!

        r_angle = random.random()*30-15
        base_image = rotate(img_augmented, r_angle,mode='reflect')
        tgt_image =  rotate(p_xyz, r_angle,mode='reflect')

        mask_area_crop = rotate(p_mask_no_occ.astype(np.float),r_angle)
        
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

        # print("src_image_final: ", src_image_final.dtype)
        # print("tgt_image_final: ", tgt_image_final.dtype)
        # print("mask_area_final: ", mask_area_final.dtype)

        # plt.imshow(src_image_final)
        # plt.title('after resizing')
        # plt.show()
        # plt.imshow(tgt_image_final)
        # plt.title('after resizing')
        # plt.show()
        # plt.imshow(mask_area_final)
        # plt.title('after resizing')
        # plt.show()

        return src_image_final,tgt_image_final,mask_area_final    

    def generator(self):
        scene_seq = np.arange(self.n_data)
        np.random.shuffle(scene_seq)
        idx=0
        batch_index=0
        batch_count=0
        batch_src =np.zeros((self.batch_size,self.imsize,self.imsize,3)) #templates
        batch_tgt =np.zeros((self.batch_size,self.imsize,self.imsize,3)) #templates
        batch_tgt_disc =np.zeros((self.batch_size))
        batch_prob = np.zeros((self.batch_size,self.imsize,self.imsize,1))
        batch_mask = np.zeros((self.batch_size,self.imsize,self.imsize,1))

        batch_tgt_disc[:]=1
        while True:
            v_id = scene_seq[idx]
            idx+=1
            if(idx >= scene_seq.shape[0]):
                idx=0
                np.random.shuffle(scene_seq)

            s_img,t_img,mask_area = self.get_patch_pair(v_id,batch_count)
            batch_src[batch_index] = s_img
            batch_tgt[batch_index] =t_img
            batch_prob[batch_index,:,:,0] =mask_area
            batch_index+=1
            if(batch_index >= self.batch_size):
                batch_index=0
                batch_count+=1
                if(batch_count>=100):
                    batch_count=0
                if(self.gan):
                    yield batch_src, batch_tgt ,batch_tgt_disc,batch_prob
                else:
                    yield batch_src, batch_tgt