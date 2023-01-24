
#!/usr/bin/env python
# coding: utf-8

# %%

# In[1]:
import argparse
import subprocess
import pprint
import tqdm

# In[2]:

import numpy as np
import torch
# import torch.multiprocessing
# torch.multiprocessing.set_sharing_strategy('file_system')
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
import torchvision
# In[3]:

from COTR.models import build_model
from COTR.utils import debug_utils, utils
# from COTR.datasets import cotr_dataset
# from COTR.trainers.cotr_trainer_Ver3 import COTRTrainer
from COTR.global_configs import general_config
from COTR.options.options import *
from COTR.options.options_utils import *
from DatasetLidarCamera_Ver9_4 import DatasetLidarCameraKittiOdometry
from utils import (mat2xyzrpy, merge_inputs, overlay_imgs, quat2mat,
                   quaternion_from_matrix, rotate_back, rotate_forward,
                   tvector2mat)
from COTR.models.monodepth2_model import MonoDepth
from utils_imgprocessing import colormap , two_images_side_by_side
from utils_tensorboard import push_training_data , push_validation_data
from utils_tensorboard import save_model
# In[4]:


utils.fix_randomness(0)
# np.set_printoptions(threshold=sys.maxsize)


# In[5]:
EPOCH = 1

def train(opt):
    pprint.pprint(dict(os.environ), width=1)
    result = subprocess.Popen(["nvidia-smi"], stdout=subprocess.PIPE)
    print(result.stdout.read().decode())
    # device = torch.cuda.current_device()
    device = torch.device('cuda:1')
    print(f'can see {torch.cuda.device_count()} gpus')
    print(f'current using gpu at {device} -- {torch.cuda.get_device_name(device)}')
    # dummy = torch.rand(3758725612).to(device)
    # del dummy
    torch.cuda.empty_cache()
    model = build_model(opt)
    model = model.to(device)
    model_eval = model.eval()
    monodepth_model = MonoDepth()
    
    dataset_class = DatasetLidarCameraKittiOdometry
#     if opt.enable_zoom:
#         train_dset = cotr_dataset.COTRZoomDataset(opt, 'train')
#         val_dset = cotr_dataset.COTRZoomDataset(opt, 'val')
#     else:
#         train_dset = cotr_dataset.COTRDataset(opt, 'train')
#         val_dset = cotr_dataset.COTRDataset(opt, 'val')

    dataset_train = dataset_class("/mnt/sgvrnas/sjmoon/kitti/kitti_odometry", max_r= 20.0, max_t=1.5,
                                  split='train', use_reflectance=False,
                                  val_sequence= '06')
    
    dataset_val = dataset_class("/mnt/sgvrnas/sjmoon/kitti/kitti_odometry", max_r= 20.0, max_t= 1.5,
                                split='val', use_reflectance=False,
                                val_sequence='06')
    
    train_dataset_size = len(dataset_train)
    val_dataset_size = len(dataset_val)
    print('Number of the train dataset: {}'.format(train_dataset_size))
    print('Number of the val dataset: {}'.format(val_dataset_size))

    train_loader = DataLoader(dataset_train, batch_size=opt.batch_size,
                              shuffle=opt.shuffle_data, num_workers=opt.workers,
                              worker_init_fn=utils.worker_init_fn, collate_fn=merge_inputs,drop_last=False,pin_memory=True)
    val_loader   = DataLoader(dataset_val, batch_size=opt.batch_size,
                              shuffle=opt.shuffle_data, num_workers=opt.workers,
                              worker_init_fn=utils.worker_init_fn,collate_fn=merge_inputs,drop_last=False, pin_memory=True)
    optim_list = [{"params": model.transformer.parameters(), "lr": opt.learning_rate},
                  {"params": model.corr_embed.parameters(), "lr": opt.learning_rate},
                  {"params": model.query_proj.parameters(), "lr": opt.learning_rate},
                  {"params": model.input_proj.parameters(), "lr": opt.learning_rate},
                  ]
    if opt.lr_backbone > 0:
        optim_list.append({"params": model.backbone.parameters(), "lr": opt.lr_backbone})
    optim = torch.optim.Adam(optim_list)
    
    print('Number of the batch train dataset: {}'.format (len(train_loader)))
    print('Number of the batch val dataset: {}'.format (len(val_loader)))
    
    starting_epoch = 0
    epochs = 200
    train_iter = 0
    val_iter = 0
    
    for epoch in range(starting_epoch, epochs + 1):
        
        EPOCH = epoch
        print('This is %d-th epoch' % epoch)

        train_loss_list = []
        # training batch
        for batch_idx, data_pack in tqdm.tqdm(
                enumerate(train_loader), total=len(train_loader),
                desc='Training iteration=%d' % train_iter, ncols=80,
                leave=False):
        
            rgb_input = []
            depth_pred_output_img = []
            dense_depth_img =[]
            corrs =[]
            
            for idx in range(len(data_pack['rgb'])):
                img_input = data_pack['rgb'][idx].cuda()
                dense_depth = data_pack['dense_depth_img'][idx].cuda()
                corr = data_pack['corrs'][idx].cuda()
                # batch stack 
                rgb_input.append(img_input)
                dense_depth_img.append(dense_depth)
                corrs.append(corr)
            
            rgb_input = torch.stack(rgb_input)
            dense_depth_img = torch.stack(dense_depth_img).permute(0,2,3,1).type(torch.float32)
            corrs = torch.stack(corrs)
            query     = corrs[:, :, :2].cuda()
            corr_target = corrs[:, :, 2:].cuda()
            
            optim.zero_grad()
            with torch.no_grad():    
                mono_depth_pred = monodepth_model.forward(rgb_input)
            
            for idx in range(len(data_pack['rgb'])):
                rgb_pred = mono_depth_pred[idx].squeeze(0)
                rgb_pred = colormap(rgb_pred)
                rgb_pred = torch.from_numpy(rgb_pred).type(torch.float32).cuda()
                depth_pred_output_img.append(rgb_pred)
            
            depth_pred_output_img = torch.stack(depth_pred_output_img).permute(0,2,3,1)
            sbs_img =two_images_side_by_side(depth_pred_output_img , dense_depth_img)
            
            # ####### display input signal #########        
            # plt.figure(figsize=(10, 10))
            # # plt.subplot(211)
            # plt.imshow(torchvision.utils.make_grid(sbs_img).permute(1,2,0).cpu().numpy())
            # plt.title("RGB_input", fontsize=22)
            # plt.axis('off')

            # plt.subplot(212)
            # plt.imshow(torchvision.utils.make_grid(depth_pred_output_img).permute(1,2,0).cpu().numpy() , cmap='magma')
            # plt.title("mono_depth_pred", fontsize=22)
            # plt.axis('off')        
            ############# end of display input signal ###################
            
            corrs_pred = model(sbs_img, query)['pred_corrs']
            loss = nn.functional.mse_loss(corrs_pred, corr_target)
            
            ##cyclic loss pre-processing
            img_reverse_input = torch.cat([sbs_img[..., 640:], sbs_img[..., :640]], axis=-1)
            query_reverse = corrs_pred.clone()
            query_reverse[..., 0] = query_reverse[..., 0] - 0.5
            cycle = model(img_reverse_input, query_reverse)['pred_corrs']
            cycle[..., 0] = cycle[..., 0] - 0.5
            mask = torch.norm(cycle - query, dim=-1) < 10 / 1280
            if mask.sum() > 0:
                cycle_loss = torch.nn.functional.mse_loss(cycle[mask], query[mask]) 
                loss += cycle_loss
            loss_data = loss.data.item()
            
            if np.isnan(loss_data):
                print('loss is nan during training')
                optim.zero_grad()
            else:
                loss.backward()
            optim.step()
            
            train_iter += 1
            train_loss_list.append(loss_data)
            
            # if batch_idx % 1000 == 0 and batch_idx != 0:
            #     push_training_data(opt, train_iter, data_pack, corrs_pred, corr_target, loss)

        train_mean_loss = np.array(train_loss_list).mean()
        print ("batch traing loss : " , train_mean_loss)
        
        # validation batch 
        val_loss_list = []
        for batch_idx, data_pack in tqdm.tqdm(
                enumerate(val_loader), total=len(val_loader),
                desc='Valid iteration=%d' % val_iter, ncols=80,
                leave=False):
 
            rgb_input = []
            depth_pred_output_img = []
            dense_depth_img =[]
            corrs =[]
    
            for idx in range(len(data_pack['rgb'])):
                img_input = data_pack['rgb'][idx].cuda()
                dense_depth = data_pack['dense_depth_img'][idx].cuda()
                corr = data_pack['corrs'][idx].cuda()
                # batch stack 
                rgb_input.append(img_input)
                dense_depth_img.append(dense_depth)
                corrs.append(corr)
            
            rgb_input = torch.stack(rgb_input)
            dense_depth_img = torch.stack(dense_depth_img).permute(0,2,3,1).type(torch.float32)
            corrs = torch.stack(corrs)
            query     = corrs[:, :, :2].cuda()
            corr_target = corrs[:, :, 2:].cuda()
            
            optim.zero_grad()
            with torch.no_grad():    
                mono_depth_pred = monodepth_model.forward(rgb_input)
            
            for idx in range(len(data_pack['rgb'])):
                rgb_pred = mono_depth_pred[idx].squeeze(0)
                rgb_pred = colormap(rgb_pred)
                rgb_pred = torch.from_numpy(rgb_pred).type(torch.float32).cuda()
                depth_pred_output_img.append(rgb_pred)
            
            depth_pred_output_img = torch.stack(depth_pred_output_img).permute(0,2,3,1)
            sbs_img =two_images_side_by_side(depth_pred_output_img , dense_depth_img)
            
            with torch.no_grad():
                corrs_pred = model_eval(sbs_img, query)['pred_corrs'] 
                loss = nn.functional.mse_loss(corrs_pred, corr_target)
            
                ##cyclic loss pre-processing
                img_reverse_input = torch.cat([sbs_img[..., 640:], sbs_img[..., :640]], axis=-1)
                query_reverse = corrs_pred.clone()
                query_reverse[..., 0] = query_reverse[..., 0] - 0.5
                cycle = model(img_reverse_input, query_reverse)['pred_corrs']
                cycle[..., 0] = cycle[..., 0] - 0.5
                mask = torch.norm(cycle - query, dim=-1) < 10 / 1280
                if mask.sum() > 0:
                    cycle_loss = torch.nn.functional.mse_loss(cycle[mask], query[mask]) 
                    loss += cycle_loss           
            loss_data = loss.data.item()
            
            if np.isnan(loss_data):
                print('loss is nan while validating')
            val_loss_list.append(loss_data)
            val_iter += 1
        
        val_mean_loss = np.array(val_loss_list).mean()
        validation_data = {'val_loss': val_mean_loss,
                           'pred': corrs_pred,
                           }
        
        push_validation_data(opt, val_iter, data_pack, validation_data ,query , corr_target ,sbs_img)  
        print ("batch validation loss : " , validation_data['val_loss'])
            
        if epoch % 2 == 0 and epoch != 0: 
            save_model (epoch, train_iter , optim, model , opt.out)

    print("train epoch end")

# In[6]:


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    set_general_arguments(parser)
    set_dataset_arguments(parser)
    set_nn_arguments(parser)
    set_COTR_arguments(parser)
    parser.add_argument('--num_kp', type=int,
                        default=1000)
    parser.add_argument('--kp_pool', type=int,
                        default=100)
    parser.add_argument('--enable_zoom', type=str2bool,
                        default=False)
    parser.add_argument('--zoom_start', type=float,
                        default=1.0)
    parser.add_argument('--zoom_end', type=float,
                        default=0.1)
    parser.add_argument('--zoom_levels', type=int,
                        default=10)
    parser.add_argument('--zoom_jitter', type=float,
                        default=0.5)
    parser.add_argument('--out_dir', type=str, default=general_config['out'], help='out directory')
    parser.add_argument('--tb_dir', type=str, default=general_config['tb_out'], help='tensorboard runs directory')
    parser.add_argument('--learning_rate', type=float,
                        default=1e-5, help='learning rate')
    parser.add_argument('--lr_backbone', type=float,
                        default=1e-5, help='backbone learning rate')
    parser.add_argument('--batch_size', type=int,
                        default= 1, help='batch size for training')
    parser.add_argument('--cycle_consis', type=str2bool, default=True,
                        help='cycle consistency')
    parser.add_argument('--bidirectional', type=str2bool, default=False,
                        help='left2right and right2left')
    parser.add_argument('--max_iter', type=int,
                        default=500000, help='total training iterations')
    parser.add_argument('--valid_iter', type=int,
                        default=1000, help='iterval of validation')
    #parser.add_argument('--resume', type=str2bool, default=True,
    #                    help='resume training with same model name')
    parser.add_argument('--resume', type=str2bool, default=False,
                        help='resume training with same model name')
    parser.add_argument('--cc_resume', type=str2bool, default=False,
                        help='resume from last run if possible')
    parser.add_argument('--need_rotation', type=str2bool, default=False,
                        help='rotation augmentation')
    parser.add_argument('--max_rotation', type=float, default=0,
                        help='max rotation for data augmentation')
    parser.add_argument('--rotation_chance', type=float, default=0,
                        help='the probability of being rotated')
    parser.add_argument('--load_weights', type=str, default=None, help='load a pretrained set of weights, you need to provide the model id')
    parser.add_argument('--suffix', type=str, default='', help='model suffix')
#     opt = parser.parse_args()
    opt = parser.parse_args(args=[])
    opt.command = ' '.join(sys.argv)
    layer_2_channels = {'layer1': 256,
                        'layer2': 512,
                        'layer3': 1024,
                        'layer4': 2048, }
    opt.dim_feedforward = layer_2_channels[opt.layer]
    opt.num_queries = opt.num_kp
#     opt.name = get_compact_naming_cotr(opt)
    opt.out_dir = '/home/seongjoo/work/autocalib/COTR'
    opt.name = '/20230124'
    opt.tb_dir = '/home/seongjoo/work/autocalib/COTR/tb'
#     opt.out = os.path.join(opt.out_dir, opt.name)
#     opt.tb_out = os.path.join(opt.tb_dir, opt.name)
    opt.out = '/home/seongjoo/work/autocalib/COTR/out/model/20230124'
    opt.tb_out = '/home/seongjoo/work/autocalib/COTR/out/tb/20230124'
    if opt.cc_resume:
        if os.path.isfile(os.path.join(opt.out, 'checkpoint.pth.tar')):
            print('resuming from last run')
            opt.load_weights = None
            opt.resume = True
        else:
            opt.resume = False
    assert (bool(opt.load_weights) and opt.resume) == False
    if opt.load_weights:
        opt.load_weights_path = os.path.join(opt.out_dir, opt.load_weights, 'checkpoint.pth.tar')
    if opt.resume:
        opt.load_weights_path = os.path.join(opt.out, 'checkpoint.pth.tar')
#     opt.scenes_name_list = build_scenes_name_list_from_opt(opt)
    # if opt.confirm:
    #     confirm_opt(opt)
    else:
        print_opt(opt)
#     save_opt(opt)
    train(opt)

