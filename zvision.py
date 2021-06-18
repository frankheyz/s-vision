# -*- coding: utf-8 -*-
import glob
from shutil import copy as copy_file
import json
import os
import time
import copy

from tifffile import imsave

import PIL.Image
import torch
import numpy as np
import torchio as tio
from scipy.io import loadmat
from ray import tune
from PIL import Image
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import transforms
from utils import RotationTransform
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torchvision.utils import save_image

from configs import configs as conf
from collections import OrderedDict

from utils import read_image
from utils import RandomCrop3D
from utils import resize_tensor
from utils import is_greyscale
from utils import locate_smallest_axis
from utils import back_project_tensor
from utils import valid_image_region
from utils import PixelShuffle3d

import math
from math import log10, sqrt
import warnings

from tabulate import tabulate
from matplotlib import pyplot as plt

from skimage.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim


class ZVision(nn.Module):
    base_sf = 1.0
    scale_factor = None
    scale_factor_idx = 0
    final_output = None

    def __init__(self, configs=conf):
        super(ZVision, self).__init__()
        self.configs = configs
        self.dev = torch.device(
            "cuda:0" if torch.cuda.is_available() and configs['use_gpu']
            else torch.device("cpu")
        )
        self.original_img_tensor = None
        self.output_img_path = None
        self.scale_factor = np.array(configs['scale_factor']) / np.array(self.base_sf)
        self.upscale_method = self.configs['upscale_method']

        # select 2D or 3D kernel
        self.kernel_selected = self.kernel_selector()
        # update padding
        pad_size = (self.configs['kernel_dilation'] * (self.configs['kernel_size'] - 1)) / 2
        self.configs['padding'] = (int(pad_size),) * 2 if self.configs['crop_size'].__len__() == 2 \
            else (int(pad_size), ) * 3
        self.conv_first = self.kernel_selected(
            in_channels=configs['input_channel_num'],
            out_channels=configs['kernel_channel_num'],
            kernel_size=configs['kernel_size'],
            stride=configs['kernel_stride'],
            dilation=configs['kernel_dilation'],
            padding=configs['padding'],
            padding_mode=configs['padding_mode']
        )
        self.conv_last = self.kernel_selected(
            in_channels=configs['kernel_channel_num'],
            out_channels=configs['output_channel_num'],
            kernel_size=configs['kernel_size'],
            stride=configs['kernel_stride'],
            dilation=configs['kernel_dilation'],
            padding=configs['padding'],
            padding_mode=configs['padding_mode']
        )

        # create layers
        layers = list()

        # the 1st layer
        layers.append(self.conv_first)

        # use a for loop to create hidden layers
        for i in range(1, self.configs['kernel_depth'] - 1):
            layers.append(
                self.kernel_selected(
                    in_channels=configs['kernel_channel_num'],
                    out_channels=configs['kernel_channel_num'],
                    kernel_size=configs['kernel_size'],
                    stride=configs['kernel_stride'],
                    dilation=configs['kernel_dilation'],
                    padding=configs['padding'],
                    padding_mode=configs['padding_mode'],
                )
            )

        # the last layers
        layers.append(self.conv_last)
        self.layers = nn.ModuleList(layers)

    def kernel_selector(self):
        # determine if it is 2D or 3D using crop size
        if len(self.configs['crop_size']) == 2:
            return nn.Conv2d
        elif len(self.configs['crop_size']) == 3:
            return nn.Conv3d
        else:
            raise ValueError('Incorrect crop size. Please input a list of 2 or 3 elements.')

    def forward(self, xb):
        # interpolate xb to high resolution
        xb_instance_high_res_list = []
        batch_size = xb.size()[0]
        for i in range(batch_size):
            if self.configs['crop_size'].__len__() == 2:
                xb_instance = torch.squeeze(xb, 1)[i, :]  # todo fix here
            else:  # 3d case
                xb_instance = torch.squeeze(xb, 1)[i, :, :, :]

            xb_instance_high_res = resize_tensor(
                xb_instance,  # keep x,y dimensions only
                scale_factor=self.scale_factor,
                kernel=self.upscale_method
            )
            xb_instance_high_res_list.append(xb_instance_high_res)

        # turn xb_stack to tensor
        xb_instance_high_res_tensor = torch.stack(xb_instance_high_res_list)

        # TODO check which activation function is better
        # add channel dimensions (not x,y,z, batch)
        xb_high_res = xb_instance_high_res_tensor.unsqueeze(1).float()
        xb_mid = F.relu(self.layers[0](xb_high_res))

        for layer in range(1, self.configs['kernel_depth'] - 1):
            xb_mid = F.relu(self.layers[layer](xb_mid))

        xb_last = self.layers[-1](xb_mid)
        # output the last layer with residue
        xb_output = xb_last + self.configs['residual_learning'] * xb_high_res

        return torch.clamp(xb_output, 0, 1)
        # return torch.clamp_min(xb_output, 0)

    def output(self):
        # load image
        input_img = read_image(self.configs['image_path'], self.configs['to_grayscale'])
        swap_z = False
        z_index = 0
        # convert PIL image to tensor
        if isinstance(input_img, PIL.Image.Image):
            input_img_tensor = transforms.ToTensor()(input_img)
            input_img_tensor = input_img_tensor.unsqueeze(0)  # add the 'batch size' dimension
        elif isinstance(input_img, torch.Tensor):
            input_img_tensor = input_img
            # swap z-axis to the first dimension so that flip and rotations are perform in the x-y plane
            z_index = locate_smallest_axis(input_img_tensor)
            input_img_tensor = input_img_tensor.moveaxis(z_index, 0)
            input_img_tensor = input_img_tensor.unsqueeze(0)  # add the 'batch size' dimension
            swap_z = True
        else:
            raise ValueError("Incorrect input image format. Only PIL or torch.Tensor is allowed.")

        if self.dev:
            if self.dev.type == 'cuda':
                input_img_tensor = input_img_tensor.to('cuda')

        # todo handle 3d case
        outputs = None
        # augmentation using rotation and flipping
        in_dims = input_img_tensor.shape.__len__()
        aug_number = 0
        for i in range(0, 1 + 7 * self.configs['output_flip'], 1 + int(self.scale_factor[0] != self.scale_factor[1])):
            aug_number += 1
            # Rotate 90*i degrees and flip if i>=4
            if i < 4:
                processed_input = torch.rot90(input_img_tensor, i, [in_dims - 2, in_dims - 1])
            else:  # todo check if this is dead code for 3d case
                in_dims = input_img_tensor.shape.__len__()
                processed_input = torch.fliplr(
                    torch.rot90(
                        input_img_tensor, i, [in_dims - 2, in_dims - 1]
                    )  # todo 3d rotation
                )

            # run forward propagation
            self.eval()
            with torch.no_grad():
                if swap_z:
                    # undo swapping, move z to the last axis for the model inference
                    processed_input = torch.moveaxis(processed_input, 1, -1)

                # output dimensions (1, 1, x, y) or (1, 1, x, y, z) [batch size, channel, l, w, h]
                if isinstance(self, ZVisionMini) and self.configs['crop_size'].__len__() == 3:
                    # todo: fix here
                    # this compensate line 144
                    processed_input = processed_input.unsqueeze(0)
                network_out = self.__call__(processed_input)
            # undo processing(rotation, flip, etc.)
            network_out = network_out.squeeze()
            # add a singleton dimension to make sure flipping is the same
            network_out = network_out.unsqueeze(0)

            # undo swapping
            if swap_z:
                # arrange network out as (1, z, x, y)
                network_out = torch.moveaxis(network_out, -1, 1)

            out_dims = network_out.shape.__len__()
            # todo up down flip
            if i < 4:
                network_out_undo_aug = torch.rot90(network_out, -i, [out_dims - 2, out_dims - 1])
                pass
            else:
                # special treatment for 3d output
                if network_out.shape.__len__() == 4:
                    network_out = network_out.squeeze()

                out_dims = network_out.shape.__len__()
                network_out_undo_aug = torch.rot90(
                    torch.fliplr(network_out),  # todo 3d flip
                    -i,
                    [out_dims - 2, out_dims - 1]
                )
                # show_tensor(input_img_tensor[0,:,:], title='input tensor')
                # show_tensor(processed_input[:,:,0], title='processed input')
                # show_tensor(network_out[0,:,:], title='net out')
                # show_tensor(network_out_undo_aug[0,:,:], 'net undo aug')

            # apply back projection
            network_out_bp = torch.squeeze(network_out_undo_aug)
            for back_projection_iter in range(self.configs['back_projection_iters'][self.scale_factor_idx]):
                network_out_bp = back_project_tensor(
                    # y_sr=torch.squeeze(network_out_undo_aug),
                    y_sr=network_out_bp,
                    y_lr=torch.squeeze(input_img_tensor),
                    down_kernel=self.configs['downscale_method'],
                    up_kernel=self.configs['upscale_method'],
                    sf=self.scale_factor
                )

            # normalize network_out_bp
            # todo check if normalization is necessary; check clipping of back projection
            # network_out_bp = network_out_bp / torch.max(network_out_bp)
            if outputs is None:
                # outputs = torch.cat((network_out_bp.unsqueeze(0), ), dim=0)
                outputs = network_out_bp.unsqueeze(0)
            else:
                # outputs = torch.cat((outputs, network_out_bp.unsqueeze(0)), dim=0)
                outputs += network_out_bp.unsqueeze(0)

        # use the mean to output is memory-friendly but may harm the performance
        intermediate_network_out = (outputs.squeeze()) / aug_number

        del outputs

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

        self.save_outputs()

        return self.final_output

    def save_outputs(self):
        # save output img
        if self.configs['save_output_img'] is True:
            out_path = os.path.join(
                self.configs['save_path'],
                self.configs['output_img_dir']
            )
            os.makedirs(out_path, exist_ok=True)
            img_name = self.configs["image_path"].split('/')[-1]
            out_name = img_name[:-4] \
                       + ''.join('X%.2f' % s for s in self.configs['scale_factor']) \
                       + '.' + self.configs['image_path'].split('.')[-1]
            self.output_img_path = os.path.join(out_path, out_name)
            if out_name.endswith('jpg') or out_name.endswith('png'):
                out_img = self.final_output #/ torch.max(self.final_output)
                save_image(out_img, self.output_img_path)
            elif out_name.endswith('tif'):
                # save as tif.
                out_img = self.final_output.cpu().numpy()
                if out_img.max() > 1:
                    out_img = out_img / out_img.max()
                out_img = out_img * 255
                out_img = out_img.astype('uint8')
                imsave(self.output_img_path, out_img)
            else:
                raise TypeError("Invalid output image format.")

        if self.configs['save_configs'] is True:
            out_path = os.path.join(
                self.configs['save_path'],
                self.configs['output_configs_dir']
            )
            os.makedirs(out_path, exist_ok=True)
            # save(copy) config for reproducibility
            with open(out_path + "configs.json", 'w') as f:
                json.dump(self.configs, f, indent=4)

        if self.configs['copy_code']:
            local_dir = os.path.dirname(__file__)
            for py_file in glob.glob(local_dir + '/*.py'):
                copy_file(py_file, self.configs['save_path'])

    def evaluate_error(self):
        # mse, ssim etc.
        # format output
        interp_factor = self.configs['serial_training'] * 2
        final_output_np = self.final_output.detach().cpu().numpy()
        # load reference image
        ref_path = self.configs['reference_img_path']
        ref_img = read_image(ref_path, self.configs['to_grayscale'])
        # ref_img = Image.open(ref_path).convert('L')
        ref_img = np.asarray(ref_img).astype(final_output_np.dtype)
        if locate_smallest_axis(ref_img) != locate_smallest_axis(final_output_np):
            # move the z axis to the last
            final_output_np = np.moveaxis(
                final_output_np, locate_smallest_axis(final_output_np), -1
            )

        ref_img_normalized = ref_img/np.max(ref_img)

        # interpolation
        original_lr_img = read_image(self.configs['original_lr_img_for_comparison'], self.configs['to_grayscale'])
        interp_img = resize_tensor(original_lr_img, interp_factor, kernel=self.configs['interp_method'])
        interp_img = interp_img.numpy()
        if locate_smallest_axis(ref_img) != locate_smallest_axis(interp_img):
            # move the z axis to the last
            interp_img = np.moveaxis(
                interp_img, locate_smallest_axis(interp_img), -1
            )
        interp_img = interp_img.astype('float32')
        interp_img_normalized = interp_img/np.max(interp_img)

        if ref_img_normalized.shape != final_output_np.shape:
            warnings.warn(
                message='The output image shape does not match the reference. No evaluation was performed.'
            )

            return

        sr_mse = mean_squared_error(
            valid_image_region(ref_img_normalized, self.configs),
            valid_image_region(final_output_np, self.configs)
        )
        sr_ssim = ssim(
            valid_image_region(ref_img_normalized, self.configs),
            valid_image_region(final_output_np, self.configs)
        )
        sr_psnr = 20 * log10(1/sqrt(sr_mse))

        interp_mse = mean_squared_error(
            valid_image_region(ref_img_normalized, self.configs),
            valid_image_region(interp_img_normalized, self.configs),
        )
        interp_ssim = ssim(
            valid_image_region(ref_img_normalized, self.configs),
            valid_image_region(interp_img_normalized, self.configs),
        )
        interp_psnr = 20 * log10(1/sqrt(interp_mse))

        print(
            tabulate(
                [
                    ["MSE", "{:.6f}".format(sr_mse), "{:.5f}".format(interp_mse)],
                    ["SSIM", "{:.6f}".format(sr_ssim), "{:.5f}".format(interp_ssim)],
                    ["PSNR", "{:.6f}".format(sr_psnr), "{:.5f}".format(interp_psnr)],
                ],
                headers=['Errors', 'SRx2', 'InterpX2'],
                tablefmt='grid'
            )
        )


class ZVisionUp(ZVision):
    def __init__(self, configs):
        ZVision.__init__(self, configs=configs)
        # upgrade network architecture

        # create layers
        layers = list()

        # the 1st layer
        layers.append(self.conv_first)

        # dilation setting
        dilation_list = [None, 1, 1, 1, 1, 1, 1]
        print(dilation_list)
        # paddings = [(d * (self.configs['kernel_size'] - 1)) / 2 for d in dilations]
        # use a for loop to create hidden layers
        for i in range(1, self.configs['kernel_depth'] - 1):
            dilation = dilation_list[i]
            padding = (dilation * (self.configs['kernel_size'] - 1)) / 2
            padding = (int(padding),) * 2 if self.configs['crop_size'].__len__() == 2 \
                else (int(padding),) * 3
            layers.append(
                self.kernel_selector()(
                    in_channels=configs['kernel_channel_num'],
                    out_channels=configs['kernel_channel_num'],
                    kernel_size=configs['kernel_size'],
                    stride=configs['kernel_stride'],
                    dilation=(dilation,),
                    padding=padding,
                    padding_mode=configs['padding_mode'],
                    groups=configs['kernel_groups'],
                )
            )
            nn.init.dirac_(layers[i].weight)
            nn.init.zeros_(layers[i].bias)

        layers.append(self.conv_last)
        self.layers = nn.ModuleList(layers)


class ZVisionMini(ZVision):
    def __init__(self, configs):
        ZVision.__init__(self, configs=configs)
        scale_factor = int(self.configs['scale_factor'][0])
        in_channels = self.configs['input_channel_num']
        out_channels = self.configs['out_channels']
        shrinking = self.configs['shrinking']
        mid_layers = self.configs['mid_layers']
        first_kernel_size = self.configs['first_kernel_size']
        mid_kernel_size = self.configs['mid_kernel_size']
        last_kernel_size = self.configs['last_kernel_size']

        self.kernel_selected = self.kernel_selector()

        self.conv_first = nn.Sequential(
            self.kernel_selected(
                in_channels,
                out_channels,
                kernel_size=first_kernel_size,
                padding=first_kernel_size//2
            ),
            nn.PReLU(out_channels)
        )
        del self.layers
        self.layers = [
            self.kernel_selected(
                in_channels=self.configs['out_channels'],
                out_channels=self.configs['shrinking'],
                kernel_size=1
            ),
            nn.PReLU(shrinking)
        ]
        for _ in range(mid_layers):
            self.layers.extend(
                [
                    self.kernel_selected(
                        in_channels=shrinking,
                        out_channels=shrinking,
                        kernel_size=mid_kernel_size,
                        padding=mid_kernel_size//2
                    ),
                    nn.PReLU(shrinking)
                ]
            )
        self.layers.extend(
            [
                self.kernel_selected(
                    in_channels=shrinking,
                    out_channels=out_channels,
                    kernel_size=1
                ),
                nn.PReLU(out_channels)
            ]
        )

        self.conv_second_last = self.kernel_selected(
            in_channels=out_channels,
            out_channels=scale_factor ** self.configs['scale_factor'].__len__(),
            kernel_size=3,
            padding=3 // 2
        )
        self.layers.extend([self.conv_second_last])

        self.layers = nn.Sequential(*self.layers)

        if self.configs['crop_size'].__len__() == 2:
            self.conv_last = nn.PixelShuffle(scale_factor)
        else:
            self.conv_last = PixelShuffle3d(scale_factor)
        # transpose_kernel = self.transpose_kernel_selector()
        # self.conv_last = transpose_kernel(
        #     in_channels=out_channels,
        #     out_channels=in_channels,
        #     kernel_size=last_kernel_size,
        #     stride=scale_factor,
        #     padding=last_kernel_size//2,
        #     output_padding=scale_factor-1
        # )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.conv_first:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=math.sqrt(2/(m.out_channels*m.weight[0][0].numel())))
                nn.init.zeros_(m.bias)
        for m in self.layers:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=math.sqrt(2/(m.out_channels*m.weight[0][0].numel())))
                nn.init.zeros_(m.bias)
        if hasattr(self.conv_last, 'weight'):
            nn.init.normal_(self.conv_last.weight, mean=0.0, std=0.001)
        if hasattr(self.conv_last, 'bias'):
            nn.init.zeros_(self.conv_last.bias)

    def forward(self, x):
        x = x.float()
        x = self.conv_first(x)
        x = self.layers(x)
        x = self.conv_last(x)
        # todo residual learning
        return torch.clamp_min(x, 0)

    def transpose_kernel_selector(self):
        # determine if it is 2D or 3D using crop size
        if len(self.configs['crop_size']) == 2:
            return nn.ConvTranspose2d
        elif len(self.configs['crop_size']) == 3:
            return nn.ConvTranspose3d
        else:
            raise ValueError('Incorrect crop size. Please input a list of 2 or 3 elements.')


def get_model(configs):
    if configs['model'] == 'up':
        # model = ZVisionUp(configs=configs)
        model = ZVisionMini(configs=configs)
        print('Upgraded model')
    else:
        model = ZVision(configs=configs)
        print('Original model')

    print(
        count_parameters(model), "trainable parameters."
    )

    return model, optim.Adam(model.parameters(), lr=configs['learning_rate'])


def get_data(train_ds, valid_ds, configs=conf):
    bs = configs['batch_size']
    num_work = configs['num_workers']

    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=num_work),
        DataLoader(valid_ds, batch_size=bs, num_workers=num_work),
    )


def get_transform(configs):
    # 2D case
    if len(configs['crop_size']) == 2:
        # calculate image mean and std.
        img = read_image(configs['image_path'])
        img_tensor = transforms.ToTensor()(img)
        img_tensor_mean = torch.mean(img_tensor).item()
        img_tensor_std = torch.std(img_tensor).item()
        # add rotations
        rotation = RotationTransform(angles=configs['rotation_angles'])
        # compose transforms
        # todo 3d transform for crop etc.
        composed_transform = transforms.Compose([
            transforms.RandomCrop(configs['crop_size']),
            transforms.RandomHorizontalFlip(p=configs['horizontal_flip_probability']),
            transforms.RandomVerticalFlip(p=configs['vertical_flip_probability']),
            rotation,
            transforms.ToTensor(),
        ])

        if configs['normalization']:
            composed_transform.transforms.append(transforms.Normalize(mean=img_tensor_mean, std=img_tensor_std))

    elif len(configs['crop_size']) == 3:  # 3D case
        normalization = tio.ZNormalization()
        (crop_x, crop_y, crop_z) = configs['crop_size']
        random_crop_3d = RandomCrop3D(crop_size=(crop_x, crop_y, crop_z))
        flips = tio.RandomFlip(axes=['LR', 'AP', 'IS'])
        # rotate about the z-axis
        rotations_dict = {
            tio.RandomAffine(scales=(1, 1, 1, 1, 1, 1), degrees=(0, 0, 0, 0, 90, 90)): 1 / 3,
            tio.RandomAffine(scales=(1, 1, 1, 1, 1, 1), degrees=(0, 0, 0, 0, 90 * 2, 90 * 2)): 1 / 3,
            tio.RandomAffine(scales=(1, 1, 1, 1, 1, 1), degrees=(0, 0, 0, 0, 90 * 3, 90 * 3)): 1 / 3,
        }
        rotation = tio.OneOf(rotations_dict)
        transforms_list = [random_crop_3d, flips, rotation]
        if configs['normalization']:
            transforms_list.insert(0, normalization)

        composed_transform = tio.Compose(transforms=transforms_list)

    else:
        raise Exception('Crop size invalid, please input 2D or 3D array.')

    return composed_transform


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
        print("**** Start training on CPU. ****")
    else:
        print("**** Start training on GPU. ****")

    if configs['provide_kernel']:
        print("**** Downscale kernel: ", configs['kernel_path'], '****')

    start_time = time.time()
    loss_values = []
    min_loss = 1
    best_model = None
    best_epoch = 0
    save_path = configs['save_path'] + configs['checkpoint_dir']
    os.makedirs(save_path, exist_ok=True)

    rate_scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer=opt,
        mode='min',
        factor=configs['adaptive_lr_factor'],
        min_lr=configs['min_lr'],
        verbose=True
    )

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

        if val_loss < min_loss:
            best_model = copy.deepcopy(model)
            best_epoch = epoch
            min_loss = val_loss

        loss_values.append(val_loss)

        if configs['adaptive_lr']:
            rate_scheduler.step(val_loss)

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
            # save the model state dict
            torch.save(best_model.state_dict(), save_path + configs['model_name'])

        # report to tune
        if 'tune' in configs:
            tune.report(loss=val_loss)

    torch.save(best_model.state_dict(), save_path + configs['model_name'])

    plt.plot(loss_values)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Losses')
    loss_path = os.path.join(configs['save_path'],configs['output_img_dir'])
    loss_file = os.path.join(loss_path, 'loss_fig')
    os.makedirs(loss_path, exist_ok=True)
    plt.savefig(loss_file)
    # plt.show()

    print("Best epoch: ", best_epoch)

    return best_model


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
        opt.zero_grad()
        loss.backward()
        opt.step()

    return loss.item(), len(xb)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

