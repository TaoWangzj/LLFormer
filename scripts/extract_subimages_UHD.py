import cv2
import numpy as np
import os
import sys
from multiprocessing import Pool
from os import path as osp
from tqdm import tqdm

# from basicsr.utils import scandir

def scandir(dir_path, suffix=None, recursive=False, full_path=False):
    """Scan a directory to find the interested files.

    Args:
        dir_path (str): Path of the directory.
        suffix (str | tuple(str), optional): File suffix that we are
            interested in. Default: None.
        recursive (bool, optional): If set to True, recursively scan the
            directory. Default: False.
        full_path (bool, optional): If set to True, include the dir_path.
            Default: False.

    Returns:
        A generator for all the interested files with relative paths.
    """

    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    root = dir_path

    def _scandir(dir_path, suffix, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith('.') and entry.is_file():
                if full_path:
                    return_path = entry.path
                else:
                    return_path = osp.relpath(entry.path, root)

                if suffix is None:
                    yield return_path
                elif return_path.endswith(suffix):
                    yield return_path
            else:
                if recursive:
                    yield from _scandir(entry.path, suffix=suffix, recursive=recursive)
                else:
                    continue

    return _scandir(dir_path, suffix=suffix, recursive=recursive)

def main():
    """A multi-thread tool to crop large images to sub-images for faster IO.

    It is used for DIV2K dataset.

    opt (dict): Configuration dict. It contains:
        n_thread (int): Thread number.
        compression_level (int):  CV_IMWRITE_PNG_COMPRESSION from 0 to 9.
            A higher value means a smaller size and longer compression time.
            Use 0 for faster CPU decompression. Default: 3, same in cv2.

        input_folder (str): Path to the input folder.
        save_folder (str): Path to save folder.
        crop_size (int): Crop size.
        step (int): Step for overlapped sliding window.
        thresh_size (int): Threshold size. Patches whose size is lower
            than thresh_size will be dropped.

    Usage:
        For each folder, run this script.
        Typically, there are four folders to be processed for DIV2K dataset.
            DIV2K_train_HR
            DIV2K_train_LR_bicubic/X2
            DIV2K_train_LR_bicubic/X3
            DIV2K_train_LR_bicubic/X4
        After process, each sub_folder should have the same number of
        subimages.
        Remember to modify opt configurations according to your settings.
    """

    opt = {}
    opt['n_thread'] = 20
    opt['compression_level'] = 1


    # High-light images
    opt['input_folder'] = 'datasets/UHD_4K/train/high'
    opt['save_folder'] = 'datasets/UHD_4K/train/high_sub'
    opt['crop_size'] = [720,1280] ### w,h
    opt['step'] = [720,1280]
    opt['thresh_size'] = 0
    extract_subimages(opt)

    # Low-light images
    opt['input_folder'] = 'datasets/UHD_4K/train/low'
    opt['save_folder'] = 'datasets/UHD_4K/train/low_sub'
    opt['crop_size'] =  [720,1280] ### w,h
    opt['step'] = [720,1280]
    opt['thresh_size'] = 0
    extract_subimages(opt)




    # # HR images
    # opt['input_folder'] = 'datasets/DIV2K/DIV2K_train_HR'
    # opt['save_folder'] = 'datasets/DIV2K/DIV2K_train_HR_sub'
    # opt['crop_size'] = 480
    # opt['step'] = 240
    # opt['thresh_size'] = 0
    # extract_subimages(opt)
    #
    # # LRx2 images
    # opt['input_folder'] = 'datasets/DIV2K/DIV2K_train_LR_bicubic/X2'
    # opt['save_folder'] = 'datasets/DIV2K/DIV2K_train_LR_bicubic/X2_sub'
    # opt['crop_size'] = 240
    # opt['step'] = 120
    # opt['thresh_size'] = 0
    # extract_subimages(opt)
    #
    # # LRx3 images
    # opt['input_folder'] = 'datasets/DIV2K/DIV2K_train_LR_bicubic/X3'
    # opt['save_folder'] = 'datasets/DIV2K/DIV2K_train_LR_bicubic/X3_sub'
    # opt['crop_size'] = 160
    # opt['step'] = 80
    # opt['thresh_size'] = 0
    # extract_subimages(opt)
    #
    # # LRx4 images
    # opt['input_folder'] = 'datasets/DIV2K/DIV2K_train_LR_bicubic/X4'
    # opt['save_folder'] = 'datasets/DIV2K/DIV2K_train_LR_bicubic/X4_sub'
    # opt['crop_size'] = 120
    # opt['step'] = 60
    # opt['thresh_size'] = 0
    extract_subimages(opt)


def extract_subimages(opt):
    """Crop images to subimages.

    Args:
        opt (dict): Configuration dict. It contains:
            input_folder (str): Path to the input folder.
            save_folder (str): Path to save folder.
            n_thread (int): Thread number.
    """
    input_folder = opt['input_folder']
    save_folder = opt['save_folder']
    if not osp.exists(save_folder):
        os.makedirs(save_folder)
        print(f'mkdir {save_folder} ...')
    else:
        print(f'Folder {save_folder} already exists. Exit.')
        sys.exit(1)

    img_list = list(scandir(input_folder, full_path=True))

    pbar = tqdm(total=len(img_list), unit='image', desc='Extract')
    pool = Pool(opt['n_thread'])
    for path in img_list:
        pool.apply_async(worker, args=(path, opt), callback=lambda arg: pbar.update(1))
    pool.close()
    pool.join()
    pbar.close()
    print('All processes done.')


def worker(path, opt):
    """Worker for each process.

    Args:
        path (str): Image path.
        opt (dict): Configuration dict. It contains:
            crop_size (int): Crop size.
            step (int): Step for overlapped sliding window.
            thresh_size (int): Threshold size. Patches whose size is lower
                than thresh_size will be dropped.
            save_folder (str): Path to save folder.
            compression_level (int): for cv2.IMWRITE_PNG_COMPRESSION.

    Returns:
        process_info (str): Process information displayed in progress bar.
    """
    crop_size = opt['crop_size']
    step = opt['step']
    thresh_size = opt['thresh_size']
    img_name, extension = osp.splitext(osp.basename(path))

    # remove the x2, x3, x4 and x8 in the filename for DIV2K
    # img_name = img_name.replace('x2', '').replace('x3', '').replace('x4', '').replace('x8', '')

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    h, w = img.shape[0:2]
    h_space = np.arange(0, h - crop_size[0] + 1, step[0])
    if h - (h_space[-1] + crop_size[0]) > thresh_size:
        h_space = np.append(h_space, h - crop_size[0])
    w_space = np.arange(0, w - crop_size[1] + 1, step[1])
    if w - (w_space[-1] + crop_size[1]) > thresh_size:
        w_space = np.append(w_space, w - crop_size[1])

    index = 0
    for x in h_space:
        for y in w_space:
            index += 1
            cropped_img = img[x:x + crop_size[0], y:y + crop_size[1], ...]
            cropped_img = np.ascontiguousarray(cropped_img)
            cv2.imwrite(
                osp.join(opt['save_folder'], f'{img_name}_s{index:03d}{extension}'), cropped_img,
                [cv2.IMWRITE_PNG_COMPRESSION, opt['compression_level']])
    process_info = f'Processing {img_name} ...'
    return process_info


if __name__ == '__main__':
    main()
