# -*- coding: utf-8 -*-
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

from tabulate import tabulate
from matplotlib import pyplot as plt

from skimage.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim


class ZVision(nn.Module):
    base_sf = 1.0
    dev = None
    scale_factor = None
    scale_factor_idx = 0
    final_output = None

    def __init__(self, configs=conf):
        super(ZVision, self).__init__()
        self.configs = configs
        self.original_img_tensor = None
        self.output_img_path = None
        self.scale_factor = np.array(configs['scale_factor']) / np.array(self.base_sf)
        self.upscale_method = self.configs['upscale_method']

        # select 2D or 3D kernel
        self.kernel_selected = self.kernel_selector()
        self.conv_first = self.kernel_selected(
            in_channels=configs['input_channel_num'],
            out_channels=configs['kernel_channel_num'],
            kernel_size=configs['kernel_size'],
            stride=configs['kernel_stride'],
            dilation=configs['kernel_dilation'],
            padding=configs['padding'],
            padding_mode=configs['padding_mode']
        )
        self.conv_mid = self.kernel_selected(
            in_channels=configs['kernel_channel_num'],
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
        layers = OrderedDict()

        # the 1st layer
        layers['0'] = self.conv_first

        # use a for loop to create hidden layers
        for i in range(1, self.configs['kernel_depth'] - 1):
            layers[str(i)] = self.conv_mid

        # the last layers
        layers[str(self.configs['kernel_depth'] - 1)] = self.conv_last
        self.layers = layers

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
        xb_hi_res = resize_tensor(
            torch.squeeze(xb),  # keep x,y dimensions only
            scale_factor=self.scale_factor,
            kernel=self.upscale_method
        )

        # TODO check which activation function is better
        xb_hi_res = xb_hi_res.unsqueeze(0).unsqueeze(0)  # add non-x,y dimensions
        xb_mid = F.relu(self.layers[str(0)](xb_hi_res))

        for layer in range(1, self.configs['kernel_depth'] - 1):
            xb_mid = F.relu(self.layers[str(layer)](xb_mid))

        # output the last layer
        # todo add residue
        xb_last = self.layers[str(self.configs['kernel_depth'] - 1)](xb_mid)

        xb_output = xb_last + self.configs['residual_learning'] * xb_hi_res

        return xb_output

    def output(self):
        # load image
        input_img = read_image(self.configs['image_path'], self.configs['to_grayscale'])
        swap_z = False
        z_index = 0
        # convert PIL image to tensor
        if isinstance(input_img, PIL.Image.Image):
            input_img_tensor = transforms.ToTensor()(input_img)
        elif isinstance(input_img, torch.Tensor):
            input_img_tensor = input_img
            # swap z-axis to the first dimension so that flip and rotations are perform in the x-y plane
            z_index = locate_smallest_axis(input_img_tensor)
            input_img_tensor = input_img_tensor.transpose(0, z_index)
            swap_z = True
        else:
            raise ValueError("Incorrect input image format. Only PIL or torch.Tensor is allowed.")

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
            else:  # todo check if this is dead code for 3d case
                in_dims = input_img_tensor.shape.__len__()
                processed_input = torch.fliplr(
                    torch.rot90(
                        input_img_tensor, i, [in_dims - 2, in_dims - 1]
                    )  # todo 3d rotation
                )

            # run forward propagation
            self.eval()
            from utils import show_tensor
            with torch.no_grad():
                if swap_z:
                    # undo swapping, move z to the last axis for the model inference
                    processed_input = torch.moveaxis(processed_input, 0, -1)
                    z = 1

                # output dimensions (1, 1, x, y) or (1, 1, x, y, z)
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
                # todo fix z axis bug
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

                # z = 1
            # apply back projection
            network_out_bp = None
            for back_projection_iter in range(self.configs['back_projection_iters'][self.scale_factor_idx]):
                network_out_bp = back_project_tensor(
                    y_sr=torch.squeeze(network_out_undo_aug),
                    y_lr=torch.squeeze(input_img_tensor),
                    down_kernel=self.configs['downscale_method'],
                    up_kernel=self.configs['upscale_method'],
                    sf=self.scale_factor
                )

            outputs.append(network_out_bp)

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
                       + self.configs['output_img_fmt']
            self.output_img_path = os.path.join(out_path, out_name)
            if out_name.endswith('jpg') or out_name.endswith('png'):
                save_image(self.final_output, self.output_img_path)
            elif out_name.endswith('tif'):
                # save as tif.
                out_img = self.final_output.cpu().numpy()
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

    def evaluate_error(self):
        # mse, ssim etc.
        # format output
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
        interp_img = resize_tensor(original_lr_img, 4, kernel=self.configs['interp_method'])
        interp_img = interp_img.numpy()
        if locate_smallest_axis(ref_img) != locate_smallest_axis(interp_img):
            # move the z axis to the last
            interp_img = np.moveaxis(
                interp_img, locate_smallest_axis(interp_img), -1
            )
        interp_img = interp_img.astype('float32')
        interp_img_normalized = interp_img/np.max(interp_img)
        sr_mse = mean_squared_error(ref_img_normalized, final_output_np)
        sr_ssim = ssim(ref_img_normalized, final_output_np)

        interp_mse = mean_squared_error(ref_img_normalized, interp_img_normalized)
        interp_ssim = ssim(ref_img_normalized, interp_img_normalized)

        print(
            tabulate(
                [
                    ["MSE", "{:.6f}".format(sr_mse), "{:.6f}".format(interp_mse)],
                    ["SSIM", "{:.6f}".format(sr_ssim), "{:.6f}".format(interp_ssim)]
                ],
                headers=['Errors', 'SRx2', 'InterpX2'],
                tablefmt='grid'
            )
        )

    def base_change(self):
        pass

    def learn_rate_policy(self):
        pass


def get_model(configs=conf):
    model = ZVision(configs=configs)
    return model, optim.Adam(model.parameters(), lr=configs['learning_rate'])


def get_data(train_ds, valid_ds, configs=conf):
    bs = configs['batch_size']
    num_work = configs['num_workers']

    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=num_work),
        DataLoader(valid_ds, batch_size=bs * 10, num_workers=num_work),
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
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)
