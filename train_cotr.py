#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse
import subprocess
import pprint


# In[2]:


import numpy as np
import torch
# import torch.multiprocessing
# torch.multiprocessing.set_sharing_strategy('file_system')
from torch.utils.data import DataLoader


# In[3]:


from COTR.models import build_model
from COTR.utils import debug_utils, utils
# from COTR.datasets import cotr_dataset
from COTR.trainers.cotr_trainer import COTRTrainer
from COTR.global_configs import general_config
from COTR.options.options import *
from COTR.options.options_utils import *
from COTR_DatasetLidarCamera_Ver5 import DatasetLidarCameraKittiOdometry
from utils import (mat2xyzrpy, merge_inputs, overlay_imgs, quat2mat,
                   quaternion_from_matrix, rotate_back, rotate_forward,
                   tvector2mat)


# In[4]:


utils.fix_randomness(0)
np.seterr(invalid='ignore')


# In[5]:


def train(opt):
    pprint.pprint(dict(os.environ), width=1)
    result = subprocess.Popen(["nvidia-smi"], stdout=subprocess.PIPE)
    print(result.stdout.read().decode())
    device = torch.cuda.current_device()
    print(f'can see {torch.cuda.device_count()} gpus')
    print(f'current using gpu at {device} -- {torch.cuda.get_device_name(device)}')
    # dummy = torch.rand(3758725612).to(device)
    # del dummy
    torch.cuda.empty_cache()
    model = build_model(opt)
    model = model.to(device)
    dataset_class = DatasetLidarCameraKittiOdometry
#     if opt.enable_zoom:
#         train_dset = cotr_dataset.COTRZoomDataset(opt, 'train')
#         val_dset = cotr_dataset.COTRZoomDataset(opt, 'val')
#     else:
#         train_dset = cotr_dataset.COTRDataset(opt, 'train')
#         val_dset = cotr_dataset.COTRDataset(opt, 'val')

    dataset_train = dataset_class("/mnt/data/kitti_odometry", max_r=10.0, max_t=0.2,
                                  split='train', use_reflectance=False,
                                  val_sequence= '00')
    
    dataset_val = dataset_class("/mnt/data/kitti_odometry", max_r=10.0, max_t=0.2,
                                split='val', use_reflectance=False,
                                val_sequence='00')
    
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
    trainer = COTRTrainer(opt, model, optim, None, train_loader, val_loader)
    trainer.train()
    print("train epoch end")


# In[6]:


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    set_general_arguments(parser)
    set_dataset_arguments(parser)
    set_nn_arguments(parser)
    set_COTR_arguments(parser)
    parser.add_argument('--num_kp', type=int,
                        default=100)
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
                        default=1e-4, help='learning rate')
    parser.add_argument('--lr_backbone', type=float,
                        default=1e-5, help='backbone learning rate')
    parser.add_argument('--batch_size', type=int,
                        default=1, help='batch size for training')
    parser.add_argument('--cycle_consis', type=str2bool, default=True,
                        help='cycle consistency')
    parser.add_argument('--bidirectional', type=str2bool, default=False,
                        help='left2right and right2left')
    parser.add_argument('--max_iter', type=int,
                        default=200000, help='total training iterations')
    parser.add_argument('--valid_iter', type=int,
                        default=1000, help='iterval of validation')
    #parser.add_argument('--resume', type=str2bool, default=True,
    #                    help='resume training with same model name')
    parser.add_argument('--resume', type=str2bool, default=False,
                        help='resume training with same model name')
    parser.add_argument('--cc_resume', type=str2bool, default=True,
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
    opt.out_dir = '/root/work/COTR/out/'
    opt.tb_dir = '/root/work/COTR/out/tb'
    opt.name = 'default'
    opt.out = os.path.join(opt.out_dir, opt.name)
    opt.tb_out = os.path.join(opt.tb_dir, opt.name)
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
    if opt.confirm:
        confirm_opt(opt)
    else:
        print_opt(opt)
#     save_opt(opt)
    train(opt)

