import yaml
import skimage
from skimage import io
from skimage.transform import resize, rotate
import os
import sys

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

from bop_toolkit_lib import inout, dataset_params
from tools import bop_io

def save_image_as_jpg(image, filename):
    #io.imsave(filename, image)
    cv2.imwrite(filename, image)
    
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
    img_r = img_r[:, :, ::-1]

    vu_valid = np.where(depth_rend > 0)
    if vu_valid[0].size == 0 or vu_valid[1].size == 0:
        bbox_gt = np.zeros(4, dtype=np.int)
    else:
        bbox_gt = np.array([np.min(vu_valid[0]), np.min(vu_valid[1]), np.max(vu_valid[0]), np.max(vu_valid[1])])
    
    return img_r, depth_rend, bbox_gt

if len(sys.argv)<3:
    print("rendering 3d coordinate images using a converted ply file, format of 6D pose challange(http://cmp.felk.cvut.cz/sixd/challenge_2017/) can be used")
    print("python3 tools/2_2_render_pix2pose_training.py [cfg_fn] [dataset_name]")    
else:
    cfg_fn = sys.argv[1] #"cfg/cfg_bop2019.json"
    cfg = inout.load_json(cfg_fn)

    dataset=sys.argv[2]
    bop_dir,source_dir,model_plys,model_info,model_ids,rgb_files,\
        depth_files,mask_files,mask_visib_files,gts,cam_param_global,scene_cam =\
             bop_io.get_dataset(cfg,dataset,incl_param=True)
    
    xyz_target_dir = bop_dir+"/train_xyz_images_test"

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
            sym_continous[3:] = m_info['symmetries_continuous'][0]['offset']
            print("Symmetric axis(x,y,z):", sym_continous[:3])

        xyz_dir =xyz_target_dir+"/{:02d}".format(int(model_id))
        rgb_jpg_dir = xyz_dir+"/rgb_images"
        xyz_jpg_dir = xyz_dir+"/xyz_images"

        if not(os.path.exists(xyz_dir)):
            os.makedirs(xyz_dir)
        if not(os.path.exists(rgb_jpg_dir)):
            os.makedirs(rgb_jpg_dir)
        if not(os.path.exists(xyz_jpg_dir)):
            os.makedirs(xyz_jpg_dir)

        xyz_id = 0
        for img_id in range(len(rgb_files)):
            gt_img = gts[img_id]
            for gt in gt_img:
                obj_id = int(gt['obj_id'])
                if(obj_id !=int(model_id)):
                    print(obj_id !=int(model_id))
                    continue    

                print("gt: ", gt)
                print("model_id: ", model_id)
                print()        
                xyz_fn = os.path.join(xyz_dir,"{:06d}.npy".format(xyz_id))
                
                rgb_fn = rgb_files[img_id] 

                string_without_extension = rgb_fn[:-4]
                desired_string = string_without_extension[-6:]

                if(dataset not in ['hb','ycbv','itodd']):
                    #for dataset that has gvien training images.
                    cam_K = np.array(scene_cam[img_id]["cam_K"]).reshape(3,3)
                ren.set_cam(cam_K)
                tra_pose = np.array((gt['cam_t_m2c']/1000))[:,0]
                rot_pose = np.array(gt['cam_R_m2c']).reshape(3,3)
                mask = inout.load_im(mask_files[img_id])>0     
                mask_visib = inout.load_im(mask_visib_files[img_id])>0
                rot_pose,rotation_lock = get_sympose(rot_pose,sym_continous)
                img_r, depth_rend, bbox_gt = get_rendering(obj_model,rot_pose,tra_pose,ren)
                
                img = inout.load_im(rgb_fn)
                
                visible_pixels = np.count_nonzero(mask_visib)
                pixels = np.count_nonzero(mask)

                # Calculate the percentage of visibility
                if pixels > 0:
                    visibility_percentage = float(visible_pixels) / float(pixels)
                else:
                    visibility_percentage = 0.0

                ymin, xmin, ymax, xmax = bbox_gt
                cropped_img = img[ymin:ymax, xmin:xmax]
                cropped_img_r = img_r[ymin:ymax, xmin:xmax]

                if cropped_img.shape[0] > 1 and cropped_img.shape[1] > 1 and cropped_img_r.shape[0] > 1 and cropped_img_r.shape[0] > 1 and visibility_percentage > 0.01:
                    rgb_output_path = os.path.join(rgb_jpg_dir, desired_string + ".jpg")
                    save_image_as_jpg(cropped_img, rgb_output_path)

                    # Save XYZ rendering
                    # xyz_output_path = os.path.join(xyz_jpg_dir, desired_string + ".npy")
                    # np.save(xyz_output_path, (img_r*255).astype(np.uint8))

                    # Save XYZ rendering
                    xyz_jpg_output_path = os.path.join(xyz_jpg_dir, desired_string + ".png")
                    #save_image_as_jpg((img_r*255).astype(np.uint8), xyz_jpg_output_path)
                    cv2.imwrite(xyz_jpg_output_path, (img_r*255).astype(np.uint8))
