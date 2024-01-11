import yaml
import skimage
from skimage import io
from skimage.transform import resize,rotate
import os,sys

ROOT_DIR = os.path.abspath(".")
sys.path.append(ROOT_DIR) 
sys.path.append("./bop_toolkit")

from rendering import utils as renderutil
from rendering.renderer_xyz import Renderer
from rendering.model import Model3D

import matplotlib.pyplot as plt
import transforms3d as tf3d
import numpy as np
import time
import cv2
from PIL import Image

from bop_toolkit_lib import inout,dataset_params
from tools import bop_io

def get_sympose(rot_pose,sym):
    rotation_lock=False
    if(np.sum(sym)>0): #continous symmetric
        axis_order='s'
        multiply=[]
        for axis_id,axis in enumerate(['x','y','z']):
            if(sym[axis_id]==1):
                axis_order+=axis
                multiply.append(0)
        for axis_id,axis in enumerate(['x','y','z']):
            if(sym[axis_id]==0):
                axis_order+=axis
                multiply.append(1)

        axis_1,axis_2,axis_3 =tf3d.euler.mat2euler(rot_pose,axis_order)
        axis_1 = axis_1*multiply[0]
        axis_2 = axis_2*multiply[1]
        axis_3 = axis_3*multiply[2]            
        rot_pose =tf3d.euler.euler2mat(axis_1,axis_2,axis_3,axis_order) #
        sym_axis_tr = np.matmul(rot_pose,np.array([sym[:3]]).T).T[0]
        z_axis = np.array([0,0,1])
        #if symmetric axis is pallell to the camera z-axis, lock the rotaion augmentation
        inner = np.abs(np.sum(sym_axis_tr*z_axis))
        if(inner>0.8):
            rotation_lock=True #lock the in-plane rotation                            

    return rot_pose,rotation_lock
def get_rendering(obj_model,rot_pose,tra_pose, ren):
    ren.clear()
    M = np.eye(4)
    M[:3, :3] = rot_pose
    M[:3, 3] = tra_pose
    ren.draw_model(obj_model, M)
    img_r, depth_rend = ren.finish()
    img_r = img_r[:, :, ::-1] * 255

    vu_valid = np.where(depth_rend > 0)
    if vu_valid[0].size == 0 or vu_valid[1].size == 0:
        bbox_gt = np.zeros(4, dtype=np.int)
    else:
        bbox_gt = np.array([np.min(vu_valid[0]), np.min(vu_valid[1]), np.max(vu_valid[0]), np.max(vu_valid[1])])
    
    return img_r, depth_rend, bbox_gt

if len(sys.argv)<3:
    print("rendering 3d coordinate images using a converted ply file, format of 6D pose challange(http://cmp.felk.cvut.cz/sixd/challenge_2017/) can be used")
    print("python3 tools/2_2_render_pix2pose_training_custom.py [cfg_fn] [dataset_name]")    
else:
    cfg_fn = sys.argv[1] #"cfg/cfg_bop2019.json"
    cfg = inout.load_json(cfg_fn)

    dataset=sys.argv[2]
    bop_dir,source_dir,model_plys,model_info,model_ids,rgb_files,\
        depth_files,mask_files,mask_visib_files,gts,cam_param_global,scene_cam =\
             bop_io.get_dataset(cfg,dataset,incl_param=True)
    
    xyz_target_dir = bop_dir+"/train_xyz"

    im_width,im_height =cam_param_global['im_size'] 
    cam_K = cam_param_global['K']
    #check if the image dimension is the same
    rgb_fn = rgb_files[0]
    img_temp = inout.load_im(rgb_fn)
    if(img_temp.shape[0]!=im_height or img_temp.shape[1]!=im_width):
        print("the size of training images is different from test images")
        im_height = img_temp.shape[0]
        im_width = img_temp.shape[1]
     
    ren = Renderer((im_width,im_height),cam_K)

t_model=-1
for m_id,model_id in enumerate(model_ids):
        if(t_model!=-1 and model_id!=t_model):
            continue
        m_info = model_info['{}'.format(model_id)]
        ply_fn =bop_dir+"/models_xyz/obj_{:06d}.ply".format(int(model_id))
        obj_model = Model3D()
        obj_model.load(ply_fn, scale=0.001)
        keys = m_info.keys()
        sym_continous = [0,0,0,0,0,0]
        if('symmetries_discrete' in keys):
            print(model_id,"is symmetric_discrete")
            print("During the training, discrete transform will be properly handled by transformer loss")
        if('symmetries_continuous' in keys):
            print(model_id,"is symmetric_continous")
            print("During the rendering, rotations w.r.t to the symmetric axis will be ignored")
            sym_continous[:3] = m_info['symmetries_continuous'][0]['axis']
            sym_continous[3:]= m_info['symmetries_continuous'][0]['offset']
            print("Symmetric axis(x,y,z):", sym_continous[:3])

        xyz_dir = xyz_target_dir+"/{:02d}".format(int(model_id))
        xyz_sub_dir = xyz_dir + "/xyz_images"
        rgb_sub_dir = xyz_dir + "/rgb_images"

        if not(os.path.exists(xyz_dir)):
            os.makedirs(xyz_dir)
        if not(os.path.exists(xyz_sub_dir)):
            os.makedirs(xyz_sub_dir)
        if not(os.path.exists(rgb_sub_dir)):
            os.makedirs(rgb_sub_dir)
            
        for img_id in range(len(rgb_files)):
            gt_img = gts[img_id]
            for gt in gt_img:            
                obj_id = int(gt['obj_id'])
                if(obj_id !=int(model_id)):
                    continue   

                rgb_fn = rgb_files[img_id]  
                string_without_extension = rgb_fn[:-4]
                img_string = string_without_extension[-6:]
                print(img_string)
                scene_string = string_without_extension[-17:-11]
                print(scene_string)

                if(dataset not in ['hb','ycbv','itodd']):
                    #for dataset that has gvien training images.
                    cam_K = np.array(scene_cam[img_id]["cam_K"]).reshape(3,3)
                ren.set_cam(cam_K)
                tra_pose = np.array((gt['cam_t_m2c']/1000))[:,0]
                rot_pose = np.array(gt['cam_R_m2c']).reshape(3,3)
                
                mask = inout.load_im(mask_files[img_id])>0     
                mask_visib = inout.load_im(mask_visib_files[img_id])>0
                rot_pose,rotation_lock = get_sympose(rot_pose,sym_continous)
                img_r,depth_rend,bbox_gt = get_rendering(obj_model,rot_pose,tra_pose,ren)
                
                img = inout.load_im(rgb_fn)

                print("mask_area: ", mask.shape)
                print("mask_visib_area: ", mask_visib.shape)
                
                # Calculate the number of non-zero pixels in the visibility mask
                visible_pixels = np.count_nonzero(mask_visib)

                # Calculate the number of non-zero pixels in the total mask
                pixels = np.count_nonzero(mask)

                print("visible_pixels", visible_pixels)
                print("pixels: ", pixels)

                # Calculate the percentage of visibility
                if pixels > 0:
                    visibility_percentage = float(visible_pixels) / float(pixels)
                else:
                    visibility_percentage = 0.0

                print("visibility_percentage: ", visibility_percentage)

                def pad_and_resize_image(img, img_r, bbox_gt):
                    bbox_width = bbox_gt[3] - bbox_gt[1]
                    bbox_height = bbox_gt[2] - bbox_gt[0]

                    center_x = (bbox_gt[1] + bbox_gt[3]) // 2
                    center_y = (bbox_gt[0] + bbox_gt[2]) // 2

                    enlarged_bbox_size = int(max(bbox_width, bbox_height) * 1.5)

                    crop_xmin = max(center_x - enlarged_bbox_size // 2, 0)
                    crop_xmax = min(center_x + enlarged_bbox_size // 2, img.shape[1])
                    crop_ymin = max(center_y - enlarged_bbox_size // 2, 0)
                    crop_ymax = min(center_y + enlarged_bbox_size // 2, img.shape[0])

                    cropped_data = np.zeros((enlarged_bbox_size, enlarged_bbox_size, 6), np.uint8)

                    crop_width = crop_xmax - crop_xmin
                    crop_height = crop_ymax - crop_ymin

                    cropped_data[
                        (enlarged_bbox_size - crop_height) // 2 : (enlarged_bbox_size - crop_height) // 2 + crop_height,
                        (enlarged_bbox_size - crop_width) // 2 : (enlarged_bbox_size - crop_width) // 2 + crop_width,
                        :3
                    ] = img[crop_ymin:crop_ymax, crop_xmin:crop_xmax]
                    cropped_data[
                        (enlarged_bbox_size - crop_height) // 2 : (enlarged_bbox_size - crop_height) // 2 + crop_height,
                        (enlarged_bbox_size - crop_width) // 2 : (enlarged_bbox_size - crop_width) // 2 + crop_width,
                        3:6
                    ] = img_r[crop_ymin:crop_ymax, crop_xmin:crop_xmax]

                    return cropped_data
                
                data = pad_and_resize_image(img, img_r, bbox_gt)
                max_axis=max(data.shape[0],data.shape[1])
                if(max_axis>128):
                    #resize to 128 
                    scale = 128.0/max_axis #128/200, 200->128
                    new_shape=np.array([data.shape[0]*scale+0.5,data.shape[1]*scale+0.5]).astype(np.int) 
                    new_data = np.zeros((new_shape[0],new_shape[1],data.shape[2]),data.dtype)
                    new_data[:,:,:3] = resize( (data[:,:,:3]/255).astype(np.float32),(new_shape[0],new_shape[1]))*255
                    new_data[:,:,3:6] = resize( (data[:,:,3:6]/255).astype(np.float32),(new_shape[0],new_shape[1]))*255
                    if(data.shape[2]>6):
                        new_data[:,:,6:] = (resize(data[:,:,6:],(new_shape[0],new_shape[1]))>0.5).astype(np.uint8)
                else:
                    new_data= data

                if new_data[:,:,3:6].shape[0] > 15 and new_data[:,:,3:6].shape[1] > 15 and new_data[:,:,:3].shape[0] > 15 and new_data[:,:,:3].shape[1] > 15 and visibility_percentage > 0.1:
                    # print("Either the first or the second number in the shape array is 0")
                    # plt.imshow(new_data[:,:,:3])
                    # plt.title('after resizing')
                    # plt.show()
                    # plt.imshow(new_data[:,:,3:6])
                    # plt.title('after resizing')
                    # plt.show()
                    #np.save(xyz_fn,new_data)
                    rgb_data = new_data[:,:,:3]
                    xyz_data = new_data[:,:,3:6]

                    xyz_fn = os.path.join(xyz_dir,scene_string+"_"+img_string+".npy")
                    xyz_sub_fn = os.path.join(xyz_sub_dir,scene_string+"_"+img_string+".png")
                    rgb_sub_fn = os.path.join(rgb_sub_dir,scene_string+"_"+img_string+".png")

                    cv2.imwrite(xyz_sub_fn, xyz_data[:, :, ::-1])
                    cv2.imwrite(rgb_sub_fn, rgb_data[:, :, ::-1])
                    #Image.fromarray(new_data[:,:,:3]).save(xyz_fn+".jpg") 
                    #if(augment_inplane>0 and not(rotation_lock)):
                    #    augment_inplane_gen(xyz_id,img,img_r,depth_rend,mask,isYCB=False,step=augment_inplane)
                #xyz_id+=1
