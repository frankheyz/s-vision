# -*- coding: utf-8 -*-
import time

"""
   Configuration file for the network
"""
configs = {
    # data loading configs
    "image_path": "/home/heyz/code/z-vision/images/",
    "images": 'yj_256.jpg',
    "data_format": 'jpg',
    "to_greyscale": True,
    "batch_size": 32,
    "num_workers": 0,

    # data preprocessing configs
    "manual_seed_num": 1,
    "scale_factor": [2.0, 2.0],  # list of list (vertical, horizontal) for gradual increments in resolution
    "kernel": 'cubic',
    "crop_size": (256, 256),
    "noise_std": 0.0,
    "rotation_angles": [90, 180, 270],
    "horizontal_flip_probability": 0.5,
    "vertical_flip_probability": 0.5,
    "output_flip": True,
    "back_projection_iters": [2],
    "upscale_method": 'cubic',
    "downscale_method": 'cubic',

    # training hyper-parameters
    "use_gpu": True,
    "learning_rate": 1e-4,
    "momentum": 0.9,
    "max_epochs": 1000,
    "min_epochs": 128,
    "show_loss": 25,
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

    # save configs
    "configs_file_path": __file__,
    "checkpoint": 500,
    "save_path": './results/' + time.strftime("%Y%m%d_%H_%M_%S", time.localtime()) + "/",
    "model_dir": 'model/',
    "checkpoint_dir": 'checkpoint/',
    "output_img_dir": 'output_image/',
    "output_img_name": 'output_img.png',
    "model_name": "model.pt"
}

