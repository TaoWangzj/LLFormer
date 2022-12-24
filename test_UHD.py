## Ultra-High-Definition Low-Light Image Enhancement: A Benchmark and Transformer-Based Method
## Tao Wang, Kaihao Zhang, Tianrun Shen, Wenhan Luo, Bjorn Stenger, Tong Lu
## https://arxiv.org/pdf/2212.11548.pdf

import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from PIL import Image
import os
from skimage import img_as_ubyte
from collections import OrderedDict
from natsort import natsorted
from glob import glob

import cv2
import argparse
from model.LLFormer import LLFormer
import torch
from torchvision.io import read_image, write_png
from torchvision.transforms.functional import crop

parser = argparse.ArgumentParser(description='Demo UHD Image Enhancement')
parser.add_argument('--input_dir', default='./datasets/UHD_4K/test/low', type=str, help='Input images')
parser.add_argument('--result_dir', default='./results/UHD-LOL4K/', type=str, help='Directory for results')
parser.add_argument('--weights',
                    default='./checkpoints/UHD-LOL4K/models/model_bestPSNR.pth', type=str,
                    help='Path to weights')

args = parser.parse_args()




def image_to_patches(image, psize, stride):
    psize_h, psize_w = psize if isinstance(psize, tuple) else (psize, psize)
    stride_h, stride_w = stride if isinstance(stride, tuple) else (stride, stride)

    h, w = image.shape[-2:]
    h_list = [i for i in range(0, h - psize_h + 1, stride_h)]
    w_list = [i for i in range(0, w - psize_w + 1, stride_w)]
    corners = [(hi, wi) for hi in h_list for wi in w_list]

    
    patches = torch.stack([
        crop(image, hi, wi, psize_h, psize_w)
        for (hi, wi) in corners
    ])
    return patches, corners


def patches_to_image(patches, corners, psize, shape):
    psize_h, psize_w = psize if isinstance(psize, tuple) else (psize, psize)
    images = torch.zeros(shape).cuda()
    counts = torch.zeros(shape).cuda()
    for (hi, wi), patch in zip(corners, patches):
        images[:, hi:hi + psize_h, wi:wi + psize_w] += patch
        counts[:, hi:hi + psize_h, wi:wi + psize_w] += 1
    images /= counts
    return images


def save_img(filepath, img):
    cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def load_checkpoint(model, weights):
    checkpoint = torch.load(weights)
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

inp_dir = args.input_dir
out_dir = args.result_dir

os.makedirs(out_dir, exist_ok=True)

files = natsorted(#glob(os.path.join(inp_dir, '*.jpg')) +
                  glob(os.path.join(inp_dir, '*.JPG'))
                  + glob(os.path.join(inp_dir, '*.png'))
                  + glob(os.path.join(inp_dir, '*.PNG')))

if len(files) == 0:
    raise Exception(f"No files found at {inp_dir}")

# Load corresponding models architecture and weights
model = LLFormer(inp_channels=3,out_channels=3,dim = 16,num_blocks = [2,4,8,16],num_refinement_blocks = 2,heads = [1,2,4,8],ffn_expansion_factor = 2.66,bias = False,LayerNorm_type = 'WithBias',attention=True,skip = False)
model.cuda()

load_checkpoint(model, args.weights)
model.eval()

print('restoring images......')

mul = 16
index = 0
psnr_val_rgb = []
for file_ in files:
    img = Image.open(file_).convert('RGB') # -------> 4K 3840x2160

    input_ = TF.to_tensor(img).unsqueeze(0).cuda()



    # Pad the input if not_multiple_of 16
    h, w = input_.shape[2], input_.shape[3]
    H, W = ((h + mul) // mul) * mul, ((w + mul) // mul) * mul
    padh = H - h if h % mul != 0 else 0
    padw = W - w if w % mul != 0 else 0
    input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')
    
    patch_size = 720, 1280
    stride = 720 // 2, 1280 // 2

    patches, corners = image_to_patches(input_[0], patch_size, stride)

    restored_patches = []
    with torch.no_grad():
        for batch_patch in patches.split(1):
            batch_patch = model(batch_patch)
            restored_patches.extend(batch_patch)
    shape = (3, H, W)
    restored = patches_to_image(restored_patches, corners, patch_size, shape)
    restored = restored.unsqueeze(0)

    restored = torch.clamp(restored, 0, 1)
    restored = restored[:, :, :h, :w]
    restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
    restored = img_as_ubyte(restored[0])

    f = os.path.splitext(os.path.split(file_)[-1])[0]
    save_img((os.path.join(out_dir, f + '.png')), restored)
    index += 1
    print('%d/%d' % (index, len(files)))

print(f"Files saved at {out_dir}")
print('finish !')
