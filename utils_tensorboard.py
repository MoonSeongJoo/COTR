import torch
import os.path as osp
from COTR.trainers import tensorboard_helper

def push_training_data(opt, train_iter ,data_pack, pred, target, loss):
    tb_datapack = tensorboard_helper.TensorboardDatapack()
    tb_datapack.set_training(True)
    tb_datapack.set_iteration(train_iter)
    tb_datapack.add_histogram({'distribution/pred': pred})
    tb_datapack.add_histogram({'distribution/target': target})
    tb_datapack.add_scalar({'loss/train': loss})
    tensorboard_helper.TensorboardPusher(opt).push_to_tensorboard(tb_datapack)

def save_model(epoch, batch_idx , optim, model ,out):
    torch.save({
        'epoch': epoch,
        'iteration': batch_idx,
        'optim_state_dict': optim.state_dict(),
        'model_state_dict': model.state_dict(),
    }, osp.join(out, 'checkpoint.pth.tar'))
    
    # if self.iteration % (10 * self.valid_iter) == 0:
    #     torch.save({
    #         'epoch': self.epoch,
    #         'iteration': self.iteration,
    #         'optim_state_dict': self.optim.state_dict(),
    #         'model_state_dict': self.model.state_dict(),
    #     }, osp.join(self.out, f'{self.iteration}_checkpoint.pth.tar'))