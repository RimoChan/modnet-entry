import os

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from .MODNet.src.models.modnet import MODNet


_here = os.path.dirname(os.path.abspath(__file__))


def get_model(ckpt_name: str) -> MODNet:
    ckpt_path = os.path.join(_here, 'MODNet', 'pretrained', ckpt_name)
    modnet = MODNet(backbone_pretrained=False)
    modnet = nn.DataParallel(modnet)

    if torch.cuda.is_available():
        modnet = modnet.cuda()
        weights = torch.load(ckpt_path)
    else:
        weights = torch.load(ckpt_path, map_location=torch.device('cpu'))
    modnet.load_state_dict(weights)
    modnet.eval()
    return modnet


def infer(modnet: MODNet, im: np.ndarray[np.uint8], ref_size=1024) -> np.ndarray[np.float32]:
    # define image to tensor transform
    im_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    # unify image channels to 3
    if len(im.shape) == 2:
        im = im[:, :, None]
    if im.shape[2] == 1:
        im = np.repeat(im, 3, axis=2)
    elif im.shape[2] == 4:
        im = im[:, :, 0:3]

    # convert image to PyTorch tensor
    im = Image.fromarray(im)
    im = im_transform(im)

    # add mini-batch dim
    im = im[None, :, :, :]

    # resize image for input
    im_b, im_c, im_h, im_w = im.shape
    if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
        if im_w >= im_h:
            im_rh = ref_size
            im_rw = int(im_w / im_h * ref_size)
        elif im_w < im_h:
            im_rw = ref_size
            im_rh = int(im_h / im_w * ref_size)
    else:
        im_rh = im_h
        im_rw = im_w

    im_rw = im_rw - im_rw % 32
    im_rh = im_rh - im_rh % 32
    im = F.interpolate(im, size=(im_rh, im_rw), mode='area')

    # inference
    _, _, matte = modnet(im.cuda() if torch.cuda.is_available() else im, True)

    # resize and save matte
    matte = F.interpolate(matte, size=(im_h, im_w), mode='area')
    matte = matte[0][0].data.cpu().numpy()
    return matte


def infer2(modnet: MODNet, img_path: str, out_alpha_path: str = '', out_img_path: str = '') -> np.ndarray[np.float32]:
    assert out_alpha_path or out_img_path
    image = np.asarray(Image.open(img_path))
    alpha = infer(modnet, image)
    alpha_uint8 = (alpha * 255).astype('uint8')
    new_image = np.concatenate((image, alpha_uint8[:, :, None]), axis=2)
    if out_alpha_path:
        Image.fromarray(alpha_uint8, mode='L').save(out_alpha_path)
    if out_img_path:
        Image.fromarray(new_image).save(out_img_path)
