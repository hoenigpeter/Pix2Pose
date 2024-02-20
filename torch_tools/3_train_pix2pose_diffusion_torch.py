import os,sys
import transforms3d as tf3d
from math import radians

if(len(sys.argv)!=5):
    print("python3 tools/3_train_pix2pose_wo_background_crop.py <gpu_id> <cfg_fn> <dataset> <obj_id>")
    sys.exit()

gpu_id = sys.argv[1]
if(gpu_id=='-1'):
    gpu_id=''
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

ROOT_DIR = os.path.abspath(".")
sys.path.append(ROOT_DIR)  # To find local version of the library
sys.path.append("./bop_toolkit")

from bop_toolkit_lib import inout,dataset_params

from pix2pose_model import ae_model_diffusion_torch as ae
import matplotlib.pyplot as plt
import time
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter 
from pix2pose_util import data_io_no_background as dataio
from torch_tools import bop_io

log_dir = "logs"  # Set your desired directory for TensorBoard logs
writer = SummaryWriter(log_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_disc_batch(X_src, X_tgt, generator_model, batch_counter, label_smoothing=False,label_flipping=0):    
    if batch_counter % 2 == 0:   
        X_src = X_src.transpose(0,3,1,2)
        X_src = torch.from_numpy(X_src).to(device).float()
        X_disc, prob = generator_model(X_src)       
        y_disc = np.zeros((X_disc.shape[0], 1), dtype=np.uint8)

    else:
        X_disc = X_tgt 
        X_disc = X_disc.transpose(0,3,1,2)
        X_disc = torch.from_numpy(X_disc).to(device).float()        
        y_disc = np.ones((X_disc.shape[0], 1), dtype=np.uint8)

    return X_disc, y_disc

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
weight_dir = bop_dir + "/pix2pose_weights_diffusion/{:02d}".format(obj_id)
if not(os.path.exists(weight_dir)):
        os.makedirs(weight_dir)
data_dir = bop_dir+"/train_xyz/{:02d}".format(obj_id)
rgb_data_dir = data_dir + "/rgb_images"
xyz_data_dir = data_dir + "/xyz_images"

batch_size=50
#datagenerator = dataio.data_generator(data_dir,rgb_data_dir,xyz_data_dir,batch_size=batch_size,res_x=im_width,res_y=im_height,prob=augmentation_prob)
augmentation_prob = 1.0
custom_dataset = dataio.CustomDataset(data_dir, rgb_data_dir, xyz_data_dir, batch_size=batch_size, gan=True, imsize=128, res_x=im_width,res_y=im_height,prob=augmentation_prob)
data_loader = DataLoader(custom_dataset, batch_size=None, shuffle=True)

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

lr = 1e-4
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8

num_steps = 10
in_channels = 3
out_channels = 64
noise_std = 0.1

generator = ae.DiffusionModel(num_steps, in_channels, out_channels, noise_std)
generator.to(device)
transformer_loss = ae.TransformerLoss(sym=sym_pool)

optimizer_generator = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2), eps=epsilon)

imsize = 128
channels = 3

dcgan_input = torch.randn(1, channels, imsize, imsize).to(device)
dcgan_target = torch.randn(1, channels, imsize, imsize).to(device)
prob_gt = torch.randn(1, 1, imsize, imsize).to(device)
gen_img,prob = generator(dcgan_input)

epoch=0
max_epoch=10

N_data = custom_dataset.n_data
batch_counter = 0

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_generator, max_epoch, eta_min=1e-7)

while epoch < max_epoch:

    custom_dataset.shuffle_indices()
    total_iterations = len(data_loader)
    
    for iteration, (X_src, X_tgt, disc_tgt, prob_gt) in enumerate(data_loader):

        start_time_iteration = time.time()  # Record start time for the current iteration

        X_disc, prob = generator(X_src)       
        y_disc = np.zeros((X_disc.shape[0], 1), dtype=np.uint8)

        optimizer_generator.zero_grad()

        y_disc = torch.from_numpy(y_disc).to(device).float() 

        dcgan_loss = transformer_loss([X_disc, X_tgt, prob, prob_gt])
    
        # Update TensorBoard with the loss value
        writer.add_scalar('Transformation Loss', dcgan_loss.item(), iteration + total_iterations * epoch)

        dcgan_loss.backward()
        optimizer_generator.step()

        elapsed_time_iteration = time.time() - start_time_iteration  # Calculate elapsed time for the current iteration

        lr_current = lr_current = optimizer_generator.param_groups[0]['lr']

        print("Epoch {:02d}, Iteration {:03d}/{:03d}, Transformation Loss: {:.4f}, lr: {:.6f}, Time per Iteration: {:.4f} seconds".format(epoch, iteration+1, total_iterations, dcgan_loss, lr_current, elapsed_time_iteration))

        if iteration % 50 == 0: 
            imgfn = weight_dir+"/val_img/"+weight_prefix+"_{:02d}.jpg".format(batch_counter+epoch*100000)
            if not(os.path.exists(weight_dir+"/val_img/")):
                os.makedirs(weight_dir+"/val_img/")

            gen_images,probs = generator(X_src)
            f,ax=plt.subplots(10,4,figsize=(10,20))
            for i in range(10):
                ax[i,0].imshow( ( (X_src[i]+1)/2).detach().cpu().numpy().transpose(1, 2, 0) )
                ax[i,1].imshow( ( (X_tgt[i]+1)/2).detach().cpu().numpy().transpose(1, 2, 0)  )
                ax[i,2].imshow( ( (gen_images[i]+1)/2).detach().cpu().numpy().transpose(1, 2, 0) )
                ax[i,3].imshow( ( (probs[i]+1)/2).detach().cpu().numpy().transpose(1, 2, 0) )
            plt.savefig(imgfn)
            plt.close()

            writer.flush()

        batch_counter+=1

    elapsed_time_batch = time.time() - start_time_iteration  # Calculate elapsed time for the whole batch
    print("Time for the whole batch: {:.4f} seconds".format(elapsed_time_batch))

    scheduler.step()
    torch.save(generator.state_dict(), os.path.join(weight_dir, f'{weight_prefix}_generator_epoch_{epoch}.pth'))
    writer.close()
    epoch += 1