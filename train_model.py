# -*- coding: utf-8 -*-
import os
import sys
import argparse
import torch
from torch import nn
from utils import Logger
from utils import ZVisionDataset
from zvision import fit
from zvision import get_data
from zvision import get_model
from zvision import get_transform
from configs import configs as conf
from configs import configs3D as conf3D


def serial_training(serial_count):
    def decorator(func):
        def wrapper(*args, **kwargs):
            trained_model = None
            # train for different scales
            for i in range(serial_count):
                trained_model = func(*args, **kwargs)
                new_training_img = trained_model.output_img_path
                # update config
                if i < serial_count - 1:
                    kwargs['configs']['image_path'] = new_training_img

            return trained_model
        return wrapper

    return decorator


@serial_training(serial_count=conf['serial_training'])
def train_model(configs=conf, checkpoint_dir=None):
    # set random seed
    torch.manual_seed(configs['manual_seed_num'])
    # set gpu
    dev = torch.device(
        "cuda:0" if torch.cuda.is_available() and configs['use_gpu']
        else torch.device("cpu")
    )

    # todo 3d transform for crop etc.
    # compose transforms
    composed_transform = get_transform(configs=configs)
    # todo optimize loading of the input for data parallelism
    # define train data set and data loader
    train_ds = ZVisionDataset(
        configs=configs, transform=composed_transform
    )

    # define train data set and data loader
    valid_ds = ZVisionDataset(
        configs=configs, transform=composed_transform
    )

    train_dl, valid_dl = get_data(
        train_ds, valid_ds, configs=configs
    )

    # get model and optimizer
    model, opt = get_model(configs=configs)

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint")
        )
        model.load_state_dict(model_state)
        opt.load_state_dict(optimizer_state)

    # todo Data parallel
    if torch.cuda.device_count() > 2:
        print("Using", torch.cuda.device_count(), "GPUs.")
        model = nn.DataParallel(model)

    print('Input image: ', configs['image_path'])

    model.to(device=dev)

    # fit the model
    trained_model = fit(
        configs=configs,
        model=model,
        loss_func=nn.L1Loss() if configs['loss_func'] == 'l1' else nn.MSELoss(),
        opt=opt,
        train_dl=train_dl,
        valid_dl=valid_dl,
        device=dev,
    )

    trained_model.output()

    if 'tune' in configs:
        return
    else:
        return trained_model


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description="Train z-vision model.")
    parser.add_argument("-m", "--model", type=str, help="choose model", default='up')
    parser.add_argument("-c", "--configs", type=str, help="Input configs.", default="2d")
    parser.add_argument("-i", "--image_path", help="Input image path.", default=None)
    parser.add_argument("-k", "--provide_kernel", type=str,help="provide kernel.", default="False")
    parser.add_argument("-n", "--notes", type=str, help="Add notes.", default="-------------------")
    args = parser.parse_args()

    input_config = conf if args.configs == '2d' else conf3D
    input_config['model'] = 'up' if args.model.lower() == 'up' else 'Original model'
    input_config['image_path'] = args.image_path if args.image_path is not None else input_config['image_path']
    input_config['original_lr_img_for_comparison'] = args.image_path if args.image_path is not None \
        else input_config['original_lr_img_for_comparison']
    input_config['provide_kernel'] = True if args.provide_kernel.lower() == 'true' else False

    # logger
    path = input_config['save_path']
    sys.stdout = Logger(path)
    print(args.notes)
    print("model:", args.model.lower())

    m = train_model(configs=input_config)

    m.evaluate_error()

    # todo add ssim to objective function
    # todo background rejection
    # todo try direct 4x
    # todo percentile normalization
    # todo demostrate the advatange of image specific network
