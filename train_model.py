# -*- coding: utf-8 -*-
import torch
from torch import nn
from silver import fit
from silver import get_data
from silver import get_model
from utils import SilverDataset
from utils import RotationTransform
from configs import configs as conf
from torchvision import transforms


def train_model(configs=conf):
    # set random seed
    torch.manual_seed(configs['manual_seed_num'])
    # set gpu
    dev = torch.device(
        "cuda:0" if torch.cuda.is_available() and configs['use_gpu']
        else torch.device("cpu")
    )

    # add rotations
    rotation = RotationTransform(angles=configs['rotation_angles'])
    # compose transforms

    composed_transform = transforms.Compose([
        transforms.RandomCrop(configs['crop_size']),
        transforms.RandomHorizontalFlip(p=configs['horizontal_flip_probability']),
        transforms.RandomVerticalFlip(p=configs['vertical_flip_probability']),
        rotation,
        transforms.ToTensor()
        # transforms.Normalize(mean=img_mean, std=img_std)
    ])

    # define train data set and data loader
    train_ds = SilverDataset(
        configs=configs, transform=composed_transform
    )

    # define train data set and data loader
    valid_ds = SilverDataset(
        configs=configs, transform=composed_transform
    )

    train_dl, valid_dl = get_data(
        train_ds, valid_ds, configs=configs
    )

    # get model and optimizer
    model, opt = get_model()

    # todo Data parallel
    if torch.cuda.device_count() > 2:
        print("Using", torch.cuda.device_count(), "GPUs.")
        model = nn.parallel.DistributedDataParallel(model)

    model.to(device=dev)
    if dev.type == 'cuda':
        model.dev = dev

    # fit the model
    fit(
        configs=configs,
        model=model,
        loss_func=nn.MSELoss(),
        opt=opt,
        train_dl=train_dl,
        valid_dl=valid_dl,
        device=dev,
    )


if __name__ == "__main__":
    train_model()

    # todo better tensor np conversion