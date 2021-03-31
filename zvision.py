# -*- coding: utf-8 -*-
import os
import time

import torch
import numpy as np

from PIL import Image
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader

from configs import configs as conf
from collections import OrderedDict

from utils import imresize
from utils import resize_tensor
from utils import is_greyscale
from utils import back_project_tensor

from matplotlib import pyplot as plt


class ZVision(nn.Module):
    base_sf = 1.0
    dev = None
    scale_factor = None
    scale_factor_idx = 0
    final_output = None

    def __init__(self, configs=conf):
        super(ZVision, self).__init__()
        self.configs = configs
        self.scale_factor = np.array(configs['scale_factor']) / np.array(self.base_sf)

        self.conv_first = nn.Conv2d(
            in_channels=configs['input_channel_num'],
            out_channels=configs['kernel_channel_num'],
            kernel_size=configs['kernel_size'],
            stride=configs['kernel_stride'],
            dilation=configs['kernel_dilation'],
            padding=configs['padding'],
            padding_mode=configs['padding_mode']
        )
        self.conv_mid = nn.Conv2d(
            in_channels=configs['kernel_channel_num'],
            out_channels=configs['kernel_channel_num'],
            kernel_size=configs['kernel_size'],
            stride=configs['kernel_stride'],
            dilation=configs['kernel_dilation'],
            padding=configs['padding'],
            padding_mode=configs['padding_mode']
        )
        self.conv_last = nn.Conv2d(
            in_channels=configs['kernel_channel_num'],
            out_channels=configs['output_channel_num'],
            kernel_size=configs['kernel_size'],
            stride=configs['kernel_stride'],
            dilation=configs['kernel_dilation'],
            padding=configs['padding'],
            padding_mode=configs['padding_mode']
        )

        # create layers
        layers = OrderedDict()

        # the 1st layer
        layers['0'] = self.conv_first

        # use a for loop to create hidden layers
        for i in range(1, self.configs['kernel_depth'] - 1):
            layers[str(i)] = self.conv_mid

        # the last layers
        layers[str(self.configs['kernel_depth'] - 1)] = self.conv_last
        self.layers = layers

    def forward(self, xb):
        # interpolate xb to high resolution
        xb_hi_res = resize_tensor(
            torch.squeeze(xb),  # keep x,y dimension only
            scale_factor=self.scale_factor,
            kernel=self.configs['upscale_method']
        )

        # TODO check which activation function is better
        xb_hi_res = xb_hi_res.unsqueeze(0).unsqueeze(0)
        xb_mid = F.relu(self.layers[str(0)](xb_hi_res))

        for layer in range(1, self.configs['kernel_depth'] - 1):
            xb_mid = F.relu(self.layers[str(layer)](xb_mid))

        # output the last layer
        # todo add residue
        xb_last = self.layers[str(self.configs['kernel_depth'] - 1)](xb_mid)

        return xb_last

    def output(self):
        # load image
        img_path = os.path.join(self.configs['image_path'], self.configs['images'])
        input_img = Image.open(img_path)
        # convert it to greyscale
        if self.configs['to_greyscale'] is True and is_greyscale(input_img) is False:
            input_img = input_img.convert("L")
        # convert it to tensor
        input_img_tensor = transforms.ToTensor()(input_img)
        if self.dev.type == 'cuda':
            input_img_tensor = input_img_tensor.to('cuda')

        # todo handle 3d case
        outputs = []
        # augmentation using rotation and flipping
        in_dims = input_img_tensor.shape.__len__()
        for i in range(0, 1 + 7 * self.configs['output_flip'], 1 + int(self.scale_factor[0] != self.scale_factor[1])):
            # Rotate 90*i degrees and flip if i>=4
            if i < 4:
                processed_input = torch.rot90(input_img_tensor, i, [in_dims - 2, in_dims - 1])
            else:
                in_dims = torch.squeeze(input_img_tensor).shape.__len__()
                processed_input = torch.fliplr(
                    torch.rot90(
                        torch.squeeze(input_img_tensor), i, [in_dims - 2, in_dims - 1]
                    ) # todo 3d rotation
                )
                # undo squeeze
                processed_input.unsqueeze_(0).unsqueeze_(0)

            # run forward propagation
            network_out = self.forward(processed_input)
            # undo processing
            out_dims = network_out.shape.__len__()
            if i < 4:
                network_out = torch.rot90(network_out, -i, [out_dims - 2, out_dims - 1])
            else:
                out_dims = torch.squeeze(network_out).shape.__len__()
                network_out = torch.rot90(
                    torch.fliplr(torch.squeeze(network_out)),  # todo 3d flip
                    -i,
                    [out_dims - 2, out_dims - 1]
                )

            # apply back projection
            for back_projection_iter in range(self.configs['back_projection_iters'][self.scale_factor_idx]):
                network_out = back_project_tensor(
                    y_sr=torch.squeeze(network_out),
                    y_lr=torch.squeeze(input_img_tensor),
                    down_kernel=self.configs['downscale_method'],
                    up_kernel=self.configs['upscale_method'],
                    sf=self.scale_factor
                )

            outputs.append(network_out)

        intermediate_network_out = torch.median(torch.stack(outputs), 0).values

        # apply back projection for the intermediate result
        for back_projection_iter in range(self.configs['back_projection_iters'][self.scale_factor_idx]):
            intermediate_network_out = back_project_tensor(
                    y_sr=intermediate_network_out,
                    y_lr=torch.squeeze(input_img_tensor),
                    down_kernel=self.configs['downscale_method'],
                    up_kernel=self.configs['upscale_method'],
                    sf=self.scale_factor
                )

        self.final_output = intermediate_network_out

        return self.final_output

    def base_change(self):
        pass

    def learn_rate_policy(self):
        pass


def get_model(configs=conf):
    model = ZVision()
    return model, optim.Adam(model.parameters(), lr=configs['learning_rate'])


def get_data(train_ds, valid_ds, configs=conf):
    bs = configs['batch_size']
    num_work = configs['num_workers']

    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=num_work),
        DataLoader(valid_ds, batch_size=bs * 2, num_workers=num_work),
    )


def fit(configs, model, loss_func, opt, train_dl, valid_dl, device=torch.device("cpu")):
    """
        Fit the network
        :param configs: training config
        :param model:
        :param loss_func:
        :param opt: optimizer
        :param train_dl: train data loader
        :param valid_dl: valid data loader
        :param device: cpu or gpu
        :param tuning: if it is called by ray tuning
        :return:
    """
    if device == torch.device("cpu"):
        print("Start training on CPU.")
    else:
        print("Start training on GPU.")

    start_time = time.time()
    loss_values = []
    for epoch in range(configs['max_epochs']):
        model.train()

        for _, sample in enumerate(train_dl):
            xb = sample['img_lr'].to(device)
            yb = sample['img'].to(device)
            loss_batch(model, loss_func, xb, yb, opt)

        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(
                    model, loss_func, sample['img_lr'].to(device), sample['img'].to(device)
                )
                    for _, sample in enumerate(valid_dl)]
            )
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        loss_values.append(val_loss)

        if epoch % configs['show_loss'] == 0:
            print("epoch: {epoch}/{epochs}  validation loss: {loss:.6f}".format(
                epoch=epoch, epochs=configs['max_epochs'], loss=val_loss)
            )

        if epoch != 0 and epoch % configs['time_lapsed'] == 0:
            time_lapsed = time.time() - start_time
            print(
                "{epoch} epoch passed after {time_lapsed:.2f}".format(epoch=epoch, time_lapsed=time_lapsed)
            )

        if epoch != 0 and epoch % configs['checkpoint'] == 0:
            save_path = configs['save_path'] + configs['checkpoint_dir']
            os.makedirs(save_path, exist_ok=True)
            # save the model state dict
            torch.save(model.state_dict(), save_path + configs['model_name'])

    plt.plot(loss_values)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Losses')
    plt.show()


def loss_batch(model, loss_func, xb, yb, opt=None):
    """
    Calculate the loss from a batch of samples
    :param model: input torch model
    :param loss_func:
    :param xb: input sample
    :param yb: target
    :param opt: optimizer
    :return: loss, sample size
    """
    # calculate loss
    loss = loss_func(model(xb), yb)  # model(xb) is the model output, yb is the target

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)