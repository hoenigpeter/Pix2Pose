import os
import skimage
from skimage.filters import gaussian
from skimage import io
from skimage.transform import resize,rotate
from skimage.filters import gaussian
from imgaug import augmenters as iaa

import numpy as np
import random

class data_generator():
    def __init__(self,data_dir, back_dir,
                 batch_size=50,gan=True,imsize=128,
                 res_x=640,res_y=480,
                 **kwargs):
        '''
        data_dir: Folder that contains cropped image+xyz
        back_dir: Folder that contains random background images
            batch_size: batch size for training
        gan: if False, gt for GAN is not yielded
        '''
        self.data_dir = data_dir
        self.back_dir = back_dir
        self.imsize=imsize
        self.batch_size = batch_size
        self.gan = gan
        self.backfiles = os.listdir(back_dir)
        data_list = os.listdir(data_dir)
        self.datafiles=[]
        self.res_x=res_x
        self.res_y=res_y

        for file in data_list:
            if(file.endswith(".npy")):
                self.datafiles.append(file)

        self.n_data = len(self.datafiles)
        self.n_background = len(self.backfiles)
        print("Total training views:", self.n_data)

        self.seq_syn= iaa.Sequential([
                                    iaa.WithChannels(0, iaa.Add((-15, 15))),
                                    iaa.WithChannels(1, iaa.Add((-15, 15))),
                                    iaa.WithChannels(2, iaa.Add((-15, 15))),
                                    iaa.ContrastNormalization((0.8, 1.3)),
                                    iaa.Multiply((0.8, 1.2),per_channel=0.5),
                                    iaa.GaussianBlur(sigma=(0.0, 0.5)),
                                    iaa.Sometimes(0.1, iaa.AdditiveGaussianNoise(scale=10, per_channel=True)),
                                    iaa.Sometimes(0.5, iaa.ContrastNormalization((0.5, 2.2), per_channel=0.3)),
                                    ], random_order=True)

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
        back_fn = self.backfiles[int(random.random()*(self.n_background-1))]
        back_img = io.imread(self.back_dir+"/"+back_fn)
        if back_img.ndim != 3:
            back_img = skimage.color.gray2rgb(back_img)
        back_img = back_img.astype(np.float32)/255
        need_resize=False
        desired_size_h=desired_size_w=0
        if(back_img.shape[0] < p_xyz.shape[0]*2):
            desired_size_h=p_xyz.shape[0]*2
            need_resize=True
        if(back_img.shape[1] < p_xyz.shape[1]*2):
            desired_size_w=p_xyz.shape[1]*2
            need_resize=True
        if(need_resize):
            back_img = resize(back_img,(max(desired_size_h,back_img.shape[0]),max(desired_size_w,back_img.shape[1])),order=1,mode='reflect')

        img_augmented = self.seq_syn.augment_image(real_img*255)/255
        v_limit = back_img.shape[0]-real_img.shape[0]-20
        u_limit = back_img.shape[1]-real_img.shape[1]-20
        v_ref =int(random.random()*v_limit+10)
        u_ref =int(random.random()*u_limit+10)
        
        #rotate -15 to 15 degree
        r_angle = random.random()*30-15
        base_image = rotate(img_augmented, r_angle,mode='reflect')
        tgt_image =  rotate(p_xyz, r_angle,mode='reflect')
        mask_area_crop = rotate(p_mask_no_occ.astype(np.float),r_angle)
        
        src_image_resized = resize(base_image,(self.imsize,self.imsize),order=1,mode='reflect')
        tgt_image_resized = resize(tgt_image,(self.imsize,self.imsize),order=1,mode='reflect')
        mask_area_resized = resize(mask_area_crop,(self.imsize,self.imsize),order=1,mode='reflect')

        return src_image_resized,tgt_image_resized,mask_area_resized    

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