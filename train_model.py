# -*- coding: utf-8 -*-
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


def serial_training(func):
    def decorator(*args, **kwargs):
        trained_model = None
        serial_count = kwargs['configs']['serial_training']
        # train for different scales
        for i in range(serial_count):
            trained_model = func(*args, **kwargs)
            trained_model.output()
            new_training_img = trained_model.output_img_path
            # update config
            if i < serial_count - 1:
                kwargs['configs']['image_path'] = new_training_img

        return trained_model
    return decorator


@serial_training
def train_model(configs=conf):
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

    # todo Data parallel
    if torch.cuda.device_count() > 2:
        print("Using", torch.cuda.device_count(), "GPUs.")
        model = nn.DataParallel(model)

    print('Input image: ', configs['image_path'])

    model.to(device=dev)
    if dev.type == 'cuda':
        model.dev = dev

    # fit the model
    trained_model = fit(
        configs=configs,
        model=model,
        loss_func=nn.MSELoss(),
        opt=opt,
        train_dl=train_dl,
        valid_dl=valid_dl,
        device=dev,
    )

    return trained_model


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description="Train z-vision model.")
    parser.add_argument("-c", "--print_string", help="Input configs.", default="2d")
    args = parser.parse_args()
    input_config = conf if args.print_string == '2d' else conf3D

    # logger
    path = input_config['save_path']
    sys.stdout = Logger(path)

    m = train_model(configs=input_config)
    result = m.output()
    m.evaluate_error()
    # import matplotlib.pyplot as plt
    # plt.imshow(result.cpu().detach().numpy(), cmap='gray')
    # plt.show()
    # pass

    # todo check 3d data augmentation
    # todo add adaptive gradient
