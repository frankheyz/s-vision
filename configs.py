# -*- coding: utf-8 -*-
import time
import copy
from torch import nn

"""
   Configuration file for the network
"""
configs = {
    # data loading configs
    "image_path": "/home/heyz/code/z-vision/images/mouse_brain_128.tif",
    "reference_img_path": "/home/heyz/code/z-vision/images/yj_1024.png",
    "original_lr_img_for_comparison": "/home/heyz/code/z-vision/images/mouse_brain_128.tif",
    "kernel_path": '/home/heyz/code/z-vision/images/psf_l_2d.mat',
    "data_format": 'jpg',
    "to_grayscale": True,
    "batch_size": 8,
    "num_workers": 0,

    # data preprocessing configs
    "manual_seed_num": 1,
    "scale_factor": [2.0, 2.0],  # list of list (vertical, horizontal) for gradual increments in resolution
    "provide_kernel": True,
    "kernel": 'cubic',
    "crop_size": (128, 128),
    "noise_std": 0.0,
    "rotation_angles": [90, 180, 270],
    "horizontal_flip_probability": 0.5,
    "vertical_flip_probability": 0.5,
    "output_flip": True,
    "back_projection_iters": [10],
    "upscale_method": 'cubic',
    "downscale_method": 'cubic',
    "normalization": False,

    # training hyper-parameters
    "model": 'old',
    "use_gpu": True,
    "serial_training": 2,
    "learning_rate": 0.00015,
    "adaptive_lr": False,
    "min_lr": 9e-6,
    "adaptive_lr_factor": 0.5,
    "loss_func": 'l2',
    "max_epochs": 1500,
    "min_epochs": 128,
    "show_loss": 50,
    "input_channel_num": 1,
    "output_channel_num": 1,
    "kernel_depth": 8,
    "kernel_size": 3,
    "kernel_channel_num": 64,
    "kernel_stride": (1, 1),
    "kernel_dilation": 1,
    "kernel_groups": 1,
    "padding": (1, 1),  # padding size should be dilation x (kernel_size - 1) / 2 to achieve same convolution
    "padding_mode": 'reflect',
    'background_threshold': 0.1,
    'background_percentage': 0.25,
    "time_lapsed": 100,
    "residual_learning": True,
    'interp_method': 'cubic',

    # ZVisionMini parameters
    'shrinking': 12,
    'out_channels': 56,
    'mid_layers': 4,
    'first_kernel_size': 5,
    'mid_kernel_size': 3,
    'last_kernel_size': 9,

    # save configs
    "configs_file_path": __file__,
    "checkpoint": 500,
    "save_path": '/home/heyz/code/z-vision/results/' + time.strftime("%Y%m%d_%H_%M_%S", time.localtime()) + "/",
    "model_dir": 'model/',
    "checkpoint_dir": 'checkpoint/',
    "save_output_img": True,
    "output_img_dir": 'output_image/',
    "model_name": "model.pt",
    "save_configs": True,
    "save_kernel": True,
    "copy_code": True,
    "output_configs_dir": 'output_configs/'
}

configs3D = copy.deepcopy(configs)
configs3D.update(
    {
        "image_path": "/home/heyz/code/z-vision/images/yj_256_256_12.tif",
        "reference_img_path": '/home/heyz/code/z-vision/images/yj_1024_1024_48.tif',
        "original_lr_img_for_comparison": "/home/heyz/code/z-vision/images/yj_256_256_12.tif",
        "kernel_path": '/home/heyz/code/z-vision/images/psf_l_3d.mat',
        "learning_rate": 0.001,
        "crop_size": (64, 64, 8),
        "scale_factor": [2, 2, 2],
        "kernel_stride": (1, 1, 1),
        "padding": (1, 1, 1),
        "kernel_channel_num": 64,
        'kernel_depth': 8,
        "padding_mode": 'zeros',

        # ZVisionMini parameters
        'shrinking': 12,
        'out_channels': 16,
        'mid_layers': 4,
        'first_kernel_size': 5,
        'mid_kernel_size': 3,
        'last_kernel_size': 9,
    }
)
