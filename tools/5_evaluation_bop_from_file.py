import os,sys
from math import radians
import csv
import cv2
from skimage.transform import resize

from matplotlib import pyplot as plt
import time
import random
import numpy as np
import transforms3d as tf3d

import json

ROOT_DIR = os.path.abspath(".")
sys.path.append(ROOT_DIR)  # To find local version of the library
sys.path.append("./bop_toolkit")

if(len(sys.argv)!=4):
    print("python3 tools/5_evaluation_bop_basic.py [gpu_id] [cfg file] [dataset_name]")
    sys.exit()
    
gpu_id = sys.argv[1]
if(gpu_id=='-1'):
    gpu_id=''
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
import tensorflow as tf
from bop_toolkit_lib import inout
from tools import bop_io
from pix2pose_util import data_io as dataio
from pix2pose_model import ae_model as ae
from pix2pose_model import recognition as recog
from pix2pose_util.common_util import get_bbox_from_mask

configuration = tf.compat.v1.ConfigProto()
configuration.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=configuration)

cfg_fn =sys.argv[2]
cfg = inout.load_json(cfg_fn)

#from tools.mask_rcnn_util import BopInferenceConfig
from skimage.transform import resize

# Load the JSON data from the file
print()
det_path = cfg["dataset_dir"] + "/" + sys.argv[3] + "/test/test_bboxes/" + cfg["detection_file"]
print(det_path)
print()

with open(det_path, 'r') as json_file:
    data = json.load(json_file)

# Create a mapping between image paths and detection instances
image_detection_map = {}
for scene_image_id, detections in data.items():
    scene_id, image_id = map(int, scene_image_id.split('/'))
    if sys.argv[3] == "itodd":
        image_path = f"{scene_id:06d}/gray/{image_id:06d}.tif"
    else:
        image_path = f"{scene_id:06d}/rgb/{image_id:06d}.png"
    image_detection_map[image_path] = detections

def get_gt_detection(path):
    print(path)
    if sys.argv[3] == "tless":
        relative_image_path = path.split('primesense/')[1]
    else:
        relative_image_path = path.split('test/')[1]

    print(relative_image_path)
    detections_for_image = image_detection_map.get(relative_image_path, [])
    rois = [[bbox[1], bbox[0], bbox[1] + bbox[3], bbox[0] + bbox[2]] for detection in detections_for_image for bbox in [detection['bbox_est']]]
    
    obj_ids = [detection['obj_id'] for detection in detections_for_image]
    scores = [detection['score'] for detection in detections_for_image]
    obj_ids = np.array(obj_ids)

    if sys.argv[3] == "lmo":
        dataset_mapping = {
            "lmo": {1: 0, 5: 1, 6: 2, 8: 3, 9: 4, 10: 5, 11: 6, 12: 7}
        }
        obj_orders = [dataset_mapping["lmo"].get(detection['obj_id'], detection['obj_id']) for detection in detections_for_image]
    else:
        obj_orders = obj_ids-1

    rois = np.array(rois)
    obj_orders = np.array(obj_orders)
    scores = np.array(scores)

    print()
    print("rois: ", rois)
    print("obj_orders: ", obj_orders)
    print("obj_ids: ", obj_ids)
    print("scores: ", scores)

    return rois,obj_orders,obj_ids,scores

score_type = cfg["score_type"]
#1-scores from a 2D detetion pipeline is used (used for the paper)
#2-scores are caluclated using detection score+overlapped mask (only supported for Mask RCNN, sued for the BOP challenge)

task_type = cfg["task_type"]
#1-Output all results for target object in the given scene
#2-ViVo task (2019 BOP challenge format, take the top-n instances)
cand_factor =float(cfg['cand_factor'])

config_tf = tf.ConfigProto()
config_tf.gpu_options.allow_growth = True
sess = tf.Session(config=config_tf)

dataset=sys.argv[3]
vis=False
output_dir = cfg['path_to_output']
output_img = output_dir+"/" +dataset
if not(os.path.exists(output_img)):
    os.makedirs(output_img)

bop_dir,test_dir,model_plys,\
model_info,model_ids,rgb_files,\
depth_files,mask_files,gts,\
cam_param_global,scene_cam = bop_io.get_dataset(cfg,dataset,incl_param=True,train=False)

im_width,im_height =cam_param_global['im_size'] 
cam_K = cam_param_global['K']
model_params =inout.load_json(os.path.join(bop_dir+"/models_xyz/",cfg['norm_factor_fn']))

if(dataset=='itodd'):
    img_type='gray'
else:
    img_type='rgb'

if("target_obj" in cfg.keys()):
    target_obj = cfg['target_obj']
    remove_obj_id=[]
    incl_obj_id=[]
    for m_id,model_id in enumerate(model_ids):
        if(model_id not in target_obj):
            remove_obj_id.append(m_id)
        else:
            incl_obj_id.append(m_id)
    for m_id in sorted(remove_obj_id,reverse=True):
        print("Remove a model:",model_ids[m_id])
        del model_plys[m_id]
        del model_info['{}'.format(model_ids[m_id])]
        
    model_ids = model_ids[incl_obj_id]
    

print("Camera info-----------------")
print(im_width,im_height)
print(cam_K)
print("-----------------")

'''
standard estimation parameter for pix2pose
'''

th_outlier=cfg['outlier_th']
dynamic_th = True
if(type(th_outlier[0])==list):
    print("Individual outlier thresholds are applied for each object")
    dynamic_th=False
    th_outliers = np.squeeze(np.array(th_outlier))
th_inlier=cfg['inlier_th']
th_ransac=3

dummy_run=False

'''
Load pix2pose inference weights
'''
load_partial=False
obj_pix2pose=[]
obj_names=[]
image_dummy=np.zeros((im_height,im_width,3),np.uint8)
if( 'backbone' in cfg.keys()):
    backbone = cfg['backbone']
else:
    backbone = 'paper'
for m_id,model_id in enumerate(model_ids):
    model_param = model_params['{}'.format(model_id)]
    obj_param=bop_io.get_model_params(model_param)
    weight_dir = bop_dir+"/pix2pose_weights/{:02d}".format(model_id)
    if(backbone=='resnet50'):
        weight_fn = os.path.join(weight_dir,"inference_resnet_model.hdf5")
        if not(os.path.exists(weight_fn)):
            weight_fn = os.path.join(weight_dir,"inference_resnet50.hdf5")
    else:
        weight_fn = os.path.join(weight_dir,"inference.hdf5")
    print("load pix2pose weight for obj_{} from".format(model_id),weight_fn)
    if not(dynamic_th):
        th_outlier = [th_outliers[m_id]] #provid a fixed outlier value
        print("Set outlier threshold to ",th_outlier[0])    
    recog_temp = recog.pix2pose(weight_fn,camK= cam_K,
                                res_x=im_width,res_y=im_height,obj_param=obj_param,
                                th_ransac=th_ransac,th_outlier=th_outlier,
                                th_inlier=th_inlier,backbone=backbone)
    obj_pix2pose.append(recog_temp)    
    obj_names.append(model_id)

test_target_fn = cfg['test_target']
target_list = bop_io.get_target_list(os.path.join(bop_dir,test_target_fn+".json"))

prev_sid=-1
result_dataset=[]

model_ids_list = model_ids.tolist()

for scene_id,im_id,obj_id_targets,inst_counts in target_list:
    print("Recognizing scene_id:{}, im_id:{}".format(scene_id,im_id))
    if(prev_sid!=scene_id):
        cam_path = test_dir+"/{:06d}/scene_camera.json".format(scene_id)
        cam_info = inout.load_scene_camera(cam_path)
        if(dummy_run):
            image_t = np.zeros((im_height,im_width,3),np.uint8)        
            for obj_id_target in obj_id_targets: #refreshing
                _,_,_,_,_,_ = obj_pix2pose[model_ids_list.index(obj_id_target)].est_pose(image_t,np.array([0,0,128,128],np.int))    
    
    prev_sid=scene_id #to avoid re-load scene_camera.json
    cam_param = cam_info[im_id]
    cam_K = cam_param['cam_K']
    depth_scale = cam_param['depth_scale'] #depth/1000 * depth_scale
    
    if(img_type=='gray'):
        rgb_path = test_dir+"/{:06d}/".format(scene_id)+img_type+\
                        "/{:06d}.tif".format(im_id)
        image_gray = inout.load_im(rgb_path)
        #copy gray values to three channels    
        image_t = np.zeros((image_gray.shape[0],image_gray.shape[1],3),dtype=np.uint8)
        image_t[:,:,:]= np.expand_dims(image_gray,axis=2)
    else:
        rgb_path = test_dir+"/{:06d}/".format(scene_id)+img_type+\
                        "/{:06d}.png".format(im_id)
        image_t = inout.load_im(rgb_path)            

    t1=time.time()
    inst_count_est=np.zeros((len(inst_counts)))
    inst_count_pred = np.zeros((len(inst_counts)))
    
    if(img_type=='gray'):
        rgb_path = test_dir+"/{:06d}/".format(scene_id)+img_type+\
                        "/{:06d}.tif".format(im_id)
        image_gray = inout.load_im(rgb_path)
        #copy gray values to three channels    
        image_t = np.zeros((image_gray.shape[0],image_gray.shape[1],3),dtype=np.uint8)
        #image_t[:,:,:]= np.expand_dims(image_gray,axis=2)
        rois,obj_orders,obj_ids,scores = get_gt_detection(rgb_path)

    else:
        rgb_path = test_dir+"/{:06d}/".format(scene_id)+img_type+\
                        "/{:06d}.png".format(im_id)
        image_t = inout.load_im(rgb_path)   
        rois,obj_orders,obj_ids,scores = get_gt_detection(rgb_path)         

    
    result_score=[]
    result_objid=[]
    result_R=[]
    result_t=[]
    result_roi=[]
    result_img=[]
    vis=False

    for r_id,roi in enumerate(rois):
        if(roi[0]==-1 and roi[1]==-1):
            continue
        obj_id = obj_ids[r_id]        
        if not(obj_id in obj_id_targets):
            #skip if the detected object is not in the target object
            continue           
        obj_gt_no = obj_id_targets.index(obj_id)
        if(inst_count_pred[obj_gt_no]>inst_counts[obj_gt_no]*cand_factor):
          continue
        inst_count_pred[obj_gt_no]+=1

        obj_order_id = obj_orders[r_id]
        obj_pix2pose[obj_order_id].camK=cam_K.reshape(3,3)                
        img_pred,mask_pred,rot_pred,tra_pred,frac_inlier,bbox_t =\
        obj_pix2pose[obj_order_id].est_pose(image_t,roi.astype(np.int))            
        if(frac_inlier==-1):
            continue        
        score = scores[r_id]        
        #inst_count_pred[obj_gt_no]+=1
        result_score.append(score)
        result_objid.append(obj_id)
        result_R.append(rot_pred)
        result_t.append(tra_pred)
        
    if len(result_score)>0:
        result_score = np.array(result_score)
        result_score = result_score/np.max(result_score) #normalize
        sorted_id = np.argsort(1-result_score) #sort results
        time_spend =time.time()-t1 #ends time for the computation
        total_inst=0
        n_inst = np.sum(inst_counts)
    else:
        continue    
    
    for result_id in sorted_id:
        obj_id = result_objid[result_id]
        R = result_R[result_id].flatten()
        t = (result_t[result_id]).flatten()
        score = result_score[result_id]
        obj_gt_no = obj_id_targets.index(obj_id)
        inst_count_est[obj_gt_no]+=1
        if(task_type=='2' and inst_count_est[obj_gt_no]>inst_counts[obj_gt_no]):
            #skip if the result exceeds the amount of taget instances for vivo task
            continue
        result_temp ={'scene_id':scene_id,'im_id': im_id,'obj_id':obj_id,'score':score,'R':R,'t':t,'time':time_spend }
        result_dataset.append(result_temp)
        total_inst+=1
        if(task_type=='2' and total_inst>n_inst): #for vivo task
            break        
 


if(dataset=='tless'):
    output_path = os.path.join(output_dir,"pix2pose-iccv19_"+dataset+"-test-primesense.csv")
else:
    output_path = os.path.join(output_dir,"pix2pose-iccv19_"+dataset+"-test.csv")

print("Saving the result to ",output_path)
inout.save_bop_results(output_path,result_dataset)

