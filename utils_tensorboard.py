import torch
import numpy as np
import torchvision
import os
from COTR.trainers import tensorboard_helper
from utils_imgprocessing import draw_corrs

def push_training_data(opt, train_iter ,data_pack, pred, target, loss):
    tb_datapack = tensorboard_helper.TensorboardDatapack()
    tb_datapack.set_training(True)
    tb_datapack.set_iteration(train_iter)
    tb_datapack.add_histogram({'distribution/pred': pred})
    tb_datapack.add_histogram({'distribution/target': target})
    tb_datapack.add_scalar({'loss/train': loss})
    tensorboard_helper.TensorboardPusher(opt).push_to_tensorboard(tb_datapack)

def push_validation_data(opt, val_iter, data_pack, validation_data , query , target ,img):
    val_loss = validation_data['val_loss']
    query_cpu = query.cpu().detach().numpy()
    target_cpu = target.cpu().detach().numpy()
    img_cpu = img.cpu()
    
#         print ('query_cpu shape' , query_cpu.shape)
#         print ('target_cpu shape' , target_cpu.shape)
#         print ('t_cpu shape' , target_cpu.shape)
#         print ('target_cpu shape' , target_cpu.shape)
    
    pred_corrs = np.concatenate((query_cpu, validation_data['pred'].cpu().numpy()), axis=-1)
    pred_corrs = draw_corrs(img_cpu, pred_corrs)
    gt_corrs = np.concatenate((query_cpu, target_cpu), axis=-1)
    gt_corrs = draw_corrs(img_cpu, gt_corrs, (0, 255, 0))

#         gt_img = vutils.make_grid(gt_corrs, normalize=True, scale_each=True)
#         pred_img = vutils.make_grid(pred_corrs, normalize=True, scale_each=True)
    gt_img = torchvision.utils.make_grid(gt_corrs, scale_each=True)
    pred_img = torchvision.utils.make_grid(pred_corrs, scale_each=True)
    
    tb_datapack = tensorboard_helper.TensorboardDatapack()
    tb_datapack.set_training(False)
    tb_datapack.set_iteration(val_iter)
    tb_datapack.add_scalar({'loss/val': val_loss})
    tb_datapack.add_image({'image/gt_corrs': gt_img})
    tb_datapack.add_image({'image/pred_corrs': pred_img})
    tensorboard_helper.TensorboardPusher(opt).push_to_tensorboard(tb_datapack)

def save_model(epoch, batch_idx , optim, model ,out):
    
    
    eval = os.path.isdir(out)
    if os.path.isdir(out) == False :
        os.makedirs (out)
            
    torch.save({
        'epoch': epoch,
        'iteration': batch_idx,
        'optim_state_dict': optim.state_dict(),
        'model_state_dict': model.state_dict(),
    }, os.path.join(out, f'{epoch}_checkpoint.pth.tar'))
    
    # if self.iteration % (10 * self.valid_iter) == 0:
    #     torch.save({
    #         'epoch': self.epoch,
    #         'iteration': self.iteration,
    #         'optim_state_dict': self.optim.state_dict(),
    #         'model_state_dict': self.model.state_dict(),
    #     }, osp.join(self.out, f'{self.iteration}_checkpoint.pth.tar'))