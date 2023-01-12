import torch
from torchvision.transforms import functional as tvtf
import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm 
from PIL import Image, ImageDraw

def colormap(disp):
    """"Color mapping for disp -- [H, W] -> [3, H, W]"""
    disp_np = disp.cpu().numpy()  # tensor -> numpy
    # disp_np = disp
    vmax = np.percentile(disp_np, 95)
    normalizer = mpl.colors.Normalize(vmin=disp_np.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')  #magma, plasma, etc.
    colormapped_im = (mapper.to_rgba(disp_np)[:, :, :3])
    return colormapped_im.transpose(2, 0, 1)
#         return colormapped_im

def two_images_side_by_side(img_a, img_b):
    assert img_a.shape == img_b.shape, f'{img_a.shape} vs {img_b.shape}'
    assert img_a.dtype == img_b.dtype
#         h, w, c = img_a.shape
    b, h, w, c = img_a.shape
#         canvas = np.zeros((h, 2 * w, c), dtype=img_a.dtype)
#         canvas = np.zeros((b, h, 2 * w, c), dtype=img_a.dtype)
#         canvas[:, 0 * w:1 * w, :] = img_a
#         canvas[:, 1 * w:2 * w, :] = img_b
    canvas = np.zeros((b, h, 2 * w, c), dtype=img_a.cpu().numpy().dtype)
    canvas[:, :, 0 * w:1 * w, :] = img_a.cpu().numpy()
    canvas[:, :, 1 * w:2 * w, :] = img_b.cpu().numpy()
    canvas_torch = torch.from_numpy(canvas).permute(0,3,1,2).cuda()
    canvas_torch = tvtf.normalize(canvas_torch, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
#         canvas[:, :, : , 0 * w:1 * w] = img_a.cpu().numpy()
#         canvas[:, :, : , 1 * w:2 * w] = img_b.cpu().numpy()
    return canvas_torch

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