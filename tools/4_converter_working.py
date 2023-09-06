import os,sys
import transforms3d as tf3d
from math import radians

if(len(sys.argv)!=5):
    print("python3 tools/4_converter_working.py <gpu_id> <cfg_fn> <dataset> <obj_id>")
    sys.exit()

gpu_id = sys.argv[1]
if(gpu_id=='-1'):
    gpu_id=''
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id


ROOT_DIR = os.path.abspath(".")
sys.path.append(ROOT_DIR)  # To find local version of the library
sys.path.append("./bop_toolkit")

from bop_toolkit_lib import inout,dataset_params

from pix2pose_model import ae_model as ae
import matplotlib.pyplot as plt
import time
import random
import numpy as np

import tensorflow as tf
from keras import losses
from keras import optimizers
from keras.callbacks import TensorBoard, ModelCheckpoint,Callback
from keras import backend as K
from keras.optimizers import Adam
from keras.layers import Input
from keras.models import Model
from keras.utils import GeneratorEnqueuer
from keras.layers import Layer

from pix2pose_util import data_io as dataio
from tools import bop_io

configuration = tf.compat.v1.ConfigProto()
configuration.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=configuration)

def dummy_loss(y_true,y_pred):
    return y_pred

loss_weights = [100,1]
train_gen_first = False
load_recent_weight = True

dataset=sys.argv[3]

cfg_fn = sys.argv[2] #"cfg/cfg_bop2019.json"
cfg = inout.load_json(cfg_fn)

bop_dir,source_dir,model_plys,model_info,model_ids,rgb_files,depth_files,mask_files,mask_visib_files,gts,cam_param_global,scene_cam = bop_io.get_dataset(cfg,dataset,incl_param=True)
im_width,im_height =cam_param_global['im_size'] 
weight_prefix = "pix2pose" 
obj_id = int(sys.argv[4]) #identical to the number for the ply file.
weight_dir = bop_dir+"/pix2pose_weights_no_bg/{:02d}".format(obj_id)
if not(os.path.exists(weight_dir)):
        os.makedirs(weight_dir)

m_info = model_info['{}'.format(obj_id)]
keys = m_info.keys()
sym_pool=[]
sym_cont = False
sym_pool.append(np.eye(3))
if('symmetries_discrete' in keys):
    print(obj_id,"is symmetric_discrete")
    print("During the training, discrete transform will be properly handled by transformer loss")
    sym_poses = m_info['symmetries_discrete']
    print("List of the symmetric pose(s)")
    for sym_pose in sym_poses:
        sym_pose = np.array(sym_pose).reshape(4,4)
        print(sym_pose[:3,:3])
        sym_pool.append(sym_pose[:3,:3])
if('symmetries_continuous' in keys):
    sym_cont=True

optimizer_dcgan =Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08) 
optimizer_disc = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08) 
backbone='paper'
if('backbone' in cfg.keys()):
    if(cfg['backbone']=="resnet50"):
            backbone='resnet50'
if(backbone=='resnet50'):
    generator_train = ae.aemodel_unet_resnet50(p=1.0)
else:
    generator_train = ae.aemodel_unet_prob(p=1.0)
    

discriminator = ae.DCGAN_discriminator()
imsize=128
dcgan_input = Input(shape=(imsize, imsize, 3))
dcgan_target = Input(shape=(imsize, imsize, 3))
prob_gt = Input(shape=(imsize, imsize, 1))
gen_img,prob = generator_train(dcgan_input)
recont_l = ae.transformer_loss(sym_pool)([gen_img,dcgan_target,prob,prob_gt])
discriminator.trainable = False
disc_out = discriminator(gen_img)
dcgan = Model(inputs=[dcgan_input,dcgan_target,prob_gt],outputs=[recont_l,disc_out])

epoch=0
recent_epoch=-1

if load_recent_weight:
    weight_save_gen=""
    weight_save_disc=""
    for fn_temp in sorted(os.listdir(weight_dir)):
        if(fn_temp.startswith(weight_prefix+".")):
                    temp_split  = fn_temp.split(".")
                    epoch_split = temp_split[1].split("-") #"01_real_1.0-0.1752.hdf5"
                    epoch_split2= epoch_split[0].split("_") #01_real_1.0
                    epoch_temp = int(epoch_split2[0])
                    network_part = epoch_split2[1]
                    if(epoch_temp>=recent_epoch):
                        recent_epoch = epoch_temp
                        if(network_part=="gen"):                        
                            weight_save_gen = fn_temp
                        elif(network_part=="disc"):
                            weight_save_disc = fn_temp

    if(weight_save_gen!=""):
        print("load recent weights from", weight_dir+"/"+weight_save_gen)
        generator_train.load_weights(os.path.join(weight_dir,weight_save_gen))
    
    if(weight_save_disc!=""):
        print("load recent weights from", weight_dir+"/"+weight_save_disc)
        discriminator.load_weights(os.path.join(weight_dir,weight_save_disc))
   

dcgan.compile(loss=[dummy_loss, 'binary_crossentropy'],
                loss_weights=loss_weights ,optimizer=optimizer_dcgan)
dcgan.summary()

discriminator.trainable = True
discriminator.compile(loss=['binary_crossentropy'],optimizer=optimizer_disc)
discriminator.summary()

generator_train.save(os.path.join(weight_dir,"inference_resnet_model.hdf5"))
sys.exit()  # Exit the script here          
