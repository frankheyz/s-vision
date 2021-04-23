# -*- coding: utf-8 -*-
import time
import copy

"""
   Configuration file for the network
"""
configs = {
    # data loading configs
    "image_path": "./images/yj_256.png",
    "reference_img_path": './images/yj_1024.png',
    "original_lr_img_for_comparison": "./images/yj_256.png",
    "kernel_path": './images/BSD100_100_lr_rand_ker_c_X2_0.mat',
    "data_format": 'jpg',
    "to_grayscale": True,
    "batch_size": 32,
    "num_workers": 0,

    # data preprocessing configs
    "manual_seed_num": 1,
    "scale_factor": [2.0, 2.0],  # list of list (vertical, horizontal) for gradual increments in resolution
    "provide_kernel": False,
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
    "use_gpu": True,
    "serial_training": 2,
    "learning_rate": 0.00015,
    "adaptive_lr": False,
    "min_lr": 9e-5,
    "adaptive_lr_factor": 0.8,
    "max_epochs": 1000,
    "min_epochs": 128,
    "show_loss": 50,
    "input_channel_num": 1,
    "output_channel_num": 1,
    "kernel_depth": 8,
    "kernel_size": 3,
    "kernel_channel_num": 64,
    "kernel_stride": (1, 1),
    "kernel_dilation": 1,
    "padding": (1, 1),  # padding size should be kernel_size//2 to achieve same convolution
    "padding_mode": 'reflect',
    "time_lapsed": 100,
    "residual_learning": True,
    'interp_method': 'cubic',

    # save configs
    "configs_file_path": __file__,
    "checkpoint": 500,
    "save_path": './results/' + time.strftime("%Y%m%d_%H_%M_%S", time.localtime()) + "/",
    "model_dir": 'model/',
    "checkpoint_dir": 'checkpoint/',
    "save_output_img": True,
    "output_img_dir": 'output_image/',
    "output_img_fmt": '.png',
    "model_name": "model.pt",
    "save_configs": True,
    "output_configs_dir": 'output_configs/'
}

configs3D = copy.deepcopy(configs)
configs3D.update(
    {
        "image_path": "./images/lr_default.tif",
        "reference_img_path": './images/hr_beads.tif',
        "crop_size": (32, 32, 4),
        "scale_factor": [2, 2, 2],
        "kernel_stride": (1, 1, 1),
        "padding": (1, 1, 1),
        "kernel_channel_num": 32,
        "padding_mode": 'zeros',
        "output_img_fmt": '.tif',
        "original_lr_img_for_comparison": "./images/lr_default.tif"
    }
)
