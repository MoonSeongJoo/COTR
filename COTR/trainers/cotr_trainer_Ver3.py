import os
import math
import os.path as osp
import time

import tqdm
import torch
import numpy as np
import torchvision
import torchvision.utils as vutils
from PIL import Image, ImageDraw


from COTR.utils import utils, debug_utils, constants
from COTR.trainers import base_trainer, tensorboard_helper
from COTR.projector import pcd_projector
import matplotlib.pyplot as plt
np.seterr(invalid='ignore')

class COTRTrainer(base_trainer.BaseTrainer):
    def __init__(self, opt, model, optimizer, criterion,
                 train_loader, val_loader):
        super().__init__(opt, model, optimizer, criterion,
                         train_loader, val_loader)

    def validate_batch(self, data_pack):
        assert self.model.training is False
        with torch.no_grad():
            img = []
            corrs =[]

            for idx in range(len(data_pack['rgb'])):
                img_input = data_pack['rgb'][idx].cuda()
        #         img_reverse = data_pack['rgb_reverse'].cuda()
                corrs_input = data_pack['corrs'][idx].cuda()

                # batch stack 
                img.append(img_input)
                corrs.append(corrs_input)

            img = torch.stack(img)
            corrs = torch.stack(corrs)
            img_cpu = img.cpu()
#             print ('img shape' , img.shape)

            img = img.permute(0,2,3,1)
#             print ('img shape' , img.shape)
            query = corrs[:, :, :2]
            target = corrs[:, :, 2:]

#             print ('normalized sbs_img shape' ,  img.shape)
#             plt.figure(figsize=(10, 10))
#             plt.imshow(img.squeeze(dim=0).permute(1,2,0).cpu())
#             plt.title("normalized sbs image", fontsize=22)
#             plt.axis('off')
#             plt.show()

#             plt.figure(figsize=(20, 40))
#             plt.imshow(torchvision.utils.make_grid(img_cpu).permute(2,0,1))
#             plt.title("pred_corrs", fontsize=22)
#             plt.axis('off')
#             plt.show()   
            
            self.optim.zero_grad()
            pred = self.model(img, query)['pred_corrs']
            loss = torch.nn.functional.mse_loss(pred, target)
            if self.opt.cycle_consis and self.opt.bidirectional:
                cycle = self.model(img, pred)['pred_corrs']
                mask = torch.norm(cycle - query, dim=-1) < 10 / constants.MAX_SIZE
                if mask.sum() > 0:
                    cycle_loss = torch.nn.functional.mse_loss(cycle[mask], query[mask])
                    loss += cycle_loss
            elif self.opt.cycle_consis and not self.opt.bidirectional:
                img_reverse = torch.cat([img[..., constants.MAX_SIZE:], img[..., :constants.MAX_SIZE]], axis=-1)
                query_reverse = pred.clone()
                query_reverse[..., 0] = query_reverse[..., 0] - 0.5
                cycle = self.model(img_reverse, query_reverse)['pred_corrs']
                cycle[..., 0] = cycle[..., 0] - 0.5
                mask = torch.norm(cycle - query, dim=-1) < 10 / constants.MAX_SIZE
                if mask.sum() > 0:
                    cycle_loss = torch.nn.functional.mse_loss(cycle[mask], query[mask])
                    loss += cycle_loss
            loss_data = loss.data.item()
            if np.isnan(loss_data):
                print('loss is nan while validating')
            return loss_data, pred , query , target , img

    def validate(self):
        '''validate for whole validation dataset
        '''
        training = self.model.training
        self.model.eval()
        val_loss_list = []
        for batch_idx, data_pack in tqdm.tqdm(
                enumerate(self.val_loader), total=len(self.val_loader),
                desc='Valid iteration=%d' % self.iteration, ncols=80,
                leave=False):
            loss_data, pred ,query ,target ,img = self.validate_batch(data_pack)
            val_loss_list.append(loss_data)
        mean_loss = np.array(val_loss_list).mean()
        validation_data = {'val_loss': mean_loss,
                           'pred': pred,
                           }
        self.push_validation_data(data_pack, validation_data ,query , target ,img)
        self.save_model()
        if training:
            self.model.train()

    def save_model(self):
        torch.save({
            'epoch': self.epoch,
            'iteration': self.iteration,
            'optim_state_dict': self.optim.state_dict(),
            'model_state_dict': self.model.state_dict(),
        }, osp.join(self.out, 'checkpoint.pth.tar'))
        if self.iteration % (10 * self.valid_iter) == 0:
            torch.save({
                'epoch': self.epoch,
                'iteration': self.iteration,
                'optim_state_dict': self.optim.state_dict(),
                'model_state_dict': self.model.state_dict(),
            }, osp.join(self.out, f'{self.iteration}_checkpoint.pth.tar'))

    def draw_corrs(self, imgs, corrs, col=(255, 0, 0)):
        imgs = utils.torch_img_to_np_img(imgs)
        out = []
        for img, corr in zip(imgs, corrs):
            img = np.interp(img, [img.min(), img.max()], [0, 255]).astype(np.uint8)
            img = Image.fromarray(img)
            draw = ImageDraw.Draw(img)
#             corr *= np.array([constants.MAX_SIZE * 2, constants.MAX_SIZE, constants.MAX_SIZE * 2, constants.MAX_SIZE])
            corr *= np.array([1280,384,1280,384])
            for c in corr:
                draw.line(c, fill=col)
                draw.point(c, fill=col)
            out.append(np.array(img))
        out = np.array(out) / 255.0
        return utils.np_img_to_torch_img(out)

    def push_validation_data(self, data_pack, validation_data ,query , target ,img):
        val_loss = validation_data['val_loss']
        query_cpu = query.cpu().detach().numpy()
        target_cpu = target.cpu().detach().numpy()
        img_cpu = img.cpu()
        
#         print ('query_cpu shape' , query_cpu.shape)
#         print ('target_cpu shape' , target_cpu.shape)
#         print ('t_cpu shape' , target_cpu.shape)
#         print ('target_cpu shape' , target_cpu.shape)
        
        pred_corrs = np.concatenate((query_cpu, validation_data['pred'].cpu().numpy()), axis=-1)
        pred_corrs = self.draw_corrs(img_cpu, pred_corrs)
        gt_corrs = np.concatenate((query_cpu, target_cpu), axis=-1)
        gt_corrs = self.draw_corrs(img_cpu, gt_corrs, (0, 255, 0))

#         gt_img = vutils.make_grid(gt_corrs, normalize=True, scale_each=True)
#         pred_img = vutils.make_grid(pred_corrs, normalize=True, scale_each=True)
        gt_img = vutils.make_grid(gt_corrs, scale_each=True)
        pred_img = vutils.make_grid(pred_corrs, scale_each=True)
        
        tb_datapack = tensorboard_helper.TensorboardDatapack()
        tb_datapack.set_training(False)
        tb_datapack.set_iteration(self.iteration)
        tb_datapack.add_scalar({'loss/val': val_loss})
        tb_datapack.add_image({'image/gt_corrs': gt_img})
        tb_datapack.add_image({'image/pred_corrs': pred_img})
        self.tb_pusher.push_to_tensorboard(tb_datapack)

    def train_batch(self, data_pack):
        '''train for one batch of data
        '''
#         img = data_pack['image'].cuda()
#         query = data_pack['queries'].cuda()
#         target = data_pack['targets'].cuda()
        img = []
        corrs =[]
        
        for idx in range(len(data_pack['rgb'])):
            img_input = data_pack['rgb'][idx].cuda()
    #         img_reverse = data_pack['rgb_reverse'].cuda()
            corrs_input = data_pack['corrs'][idx].cuda()

            # batch stack 
            img.append(img_input)
            corrs.append(corrs_input)

        img = torch.stack(img)
        corrs = torch.stack(corrs)
#         print ('img shape' , img.shape)
        
        img = img.permute(0,2,3,1)
        query = corrs[:, :, :2]
        target = corrs[:, :, 2:]

        self.optim.zero_grad()
        pred = self.model(img, query)['pred_corrs']
        loss = torch.nn.functional.mse_loss(pred, target)
        
#         pred_corrs = np.concatenate((query.cpu().detach().numpy(), pred.cpu().detach().numpy()), axis=-1)
#         pred_corrs = self.draw_corrs(img.cpu(), pred_corrs)
#         gt_corrs   = np.concatenate((query.cpu().detach().numpy(), target.cpu().detach().numpy()), axis=-1)
#         gt_corrs   = self.draw_corrs(img.cpu(), gt_corrs, (0, 255, 0))
        
#         plt.figure(figsize=(20, 40))
#         plt.subplot(211)
#         plt.imshow(torchvision.utils.make_grid(pred_corrs).permute(1,2,0))
#         plt.title("pred_corrs", fontsize=22)
#         plt.axis('off')
#         plt.show()  
        
#         plt.figure(figsize=(40, 80))
# #         plt.subplot(212)
#         plt.imshow(torchvision.utils.make_grid(gt_corrs, normalize=True, scale_each=True).permute(1,2,0))
#         plt.title("gt_corrs", fontsize=22)
#         plt.axis('off')
#         plt.show()
        
        if self.opt.cycle_consis and self.opt.bidirectional:
            cycle = self.model(img, pred)['pred_corrs']
            mask = torch.norm(cycle - query, dim=-1) < 10 / constants.MAX_SIZE
            if mask.sum() > 0:
                cycle_loss = torch.nn.functional.mse_loss(cycle[mask], query[mask])
                loss += cycle_loss
        elif self.opt.cycle_consis and not self.opt.bidirectional:
                img_reverse = torch.cat([img[..., constants.MAX_SIZE:], img[..., :constants.MAX_SIZE]], axis=-1)
                query_reverse = pred.clone()
                query_reverse[..., 0] = query_reverse[..., 0] - 0.5
                cycle = self.model(img_reverse, query_reverse)['pred_corrs']
                cycle[..., 0] = cycle[..., 0] - 0.5
                mask = torch.norm(cycle - query, dim=-1) < 10 / constants.MAX_SIZE
                if mask.sum() > 0:
                    cycle_loss = torch.nn.functional.mse_loss(cycle[mask], query[mask])
                    loss += cycle_loss
        loss_data = loss.data.item()
        if np.isnan(loss_data):
            print('loss is nan during training')
            self.optim.zero_grad()
        else:
            loss.backward()
            self.push_training_data(data_pack, pred, target, loss)
        self.optim.step()

    def push_training_data(self, data_pack, pred, target, loss):
        tb_datapack = tensorboard_helper.TensorboardDatapack()
        tb_datapack.set_training(True)
        tb_datapack.set_iteration(self.iteration)
        tb_datapack.add_histogram({'distribution/pred': pred})
        tb_datapack.add_histogram({'distribution/target': target})
        tb_datapack.add_scalar({'loss/train': loss})
        self.tb_pusher.push_to_tensorboard(tb_datapack)

    def resume(self):
        '''resume training:
        resume from the recorded epoch, iteration, and saved weights.
        resume from the model with the same name.

        Arguments:
            opt {[type]} -- [description]
        '''
        if hasattr(self.opt, 'load_weights'):
            assert self.opt.load_weights is None or self.opt.load_weights == False
        # 1. load check point
        checkpoint_path = os.path.join(self.opt.out, 'checkpoint.pth.tar')
        if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
        else:
            raise FileNotFoundError(
                'model check point cannnot found: {0}'.format(checkpoint_path))
        # 2. load data
        self.epoch = checkpoint['epoch']
        self.iteration = checkpoint['iteration']
        self.load_pretrained_weights()
        self.optim.load_state_dict(checkpoint['optim_state_dict'])

    def load_pretrained_weights(self):
        '''
        load pretrained weights from another model
        '''
        # if hasattr(self.opt, 'resume'):
        #     assert self.opt.resume is False
        assert os.path.isfile(self.opt.load_weights_path), self.opt.load_weights_path

        saved_weights = torch.load(self.opt.load_weights_path)['model_state_dict']
        utils.safe_load_weights(self.model, saved_weights)
        content_list = []
        content_list += [f'Loaded pretrained weights from {self.opt.load_weights_path}']
        utils.print_notification(content_list)
