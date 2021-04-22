# -*- coding: utf-8 -*-
import os
import sys
import json
from math import pi
from shutil import copyfile

import PIL.Image
import torch
import torchio
import torchio as tio
import random
import numpy as np
from skimage import io
from PIL import Image
from scipy.io import loadmat
from PIL.ImageStat import Stat
from os import listdir
from configs import configs
from matplotlib import pyplot as plt
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF

from skimage.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim

from tabulate import tabulate
from scipy.ndimage import filters, measurements, interpolation


class ZVisionDataset(Dataset):
    """
    Super-resolution dataset
    """
    base_sf = 1.0

    def __init__(self, configs=configs, transform=None):
        """
        init function
        :param configs: dict
        :param transform:
        """
        self.configs = configs
        self.transform = transform
        self.scale_factor = np.array(configs['scale_factor']) / np.array(self.base_sf)
        # For resize image, use kernel provided in .mat file or one of the default kernel
        self.kernel = loadmat(self.configs['kernel_path'])['Kernel'] if self.configs['provide_kernel'] is True \
            else self.configs['upscale_method']

        # load image
        img_path = configs['image_path']
        img = read_image(img_path, to_grayscale=self.configs['to_grayscale'])
        # img = Image.open(img_path)
        # img = img.convert('L')
        self.img = img

        # img = (img / img.max()).astype(np.float32)
        # self.img = torch.from_numpy(img)

    def __len__(self):
        # return 1 since it is zero-shot learning
        return 1

    # todo implement gradual SR where training data is gradually added as scale factor increases
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # use torio to perform transformations
        if isinstance(self.img, torch.Tensor) and len(self.img.shape) == 3:
            self.img = self.img.unsqueeze(0)
            self.img = tio.Subject(
                sample=tio.ScalarImage(tensor=self.img)
            )

        # transform input image for augmentation
        if self.transform:
            img = self.transform(self.img)  # note self.img has to be PIL or TIO subject
        else:
            img = self.img

        # extract data from subject
        if isinstance(self.img, torchio.data.subject.Subject):
            img = img['sample'][tio.DATA]

        # create low res image
        img_lr = self.high_res_2_low_res(img)
        # add the dimension for batch size
        img_lr = img_lr.unsqueeze(0)
        sample = {
            "img": img,
            "img_lr": img_lr
        }

        return sample

    def high_res_2_low_res(self, high_res):
        # Create son out of the father by downscaling and if indicated adding noise
        # resize tensor
        lr_son = resize_tensor(torch.squeeze(high_res), 1.0 / self.scale_factor, kernel=self.kernel)

        # add noise
        lr_son_with_noise = np.clip(lr_son + np.random.randn(*lr_son.shape) * self.configs['noise_std'], 0, 1)

        return lr_son_with_noise


class RandomCrop3D:
    def __init__(self, crop_size: tuple):
        self.crop_size = crop_size

    def __call__(self, input_tensor):
        sampler = tio.data.UniformSampler(self.crop_size)
        patch = sampler(input_tensor, 1)

        return list(patch)[0]


def read_image(img_path, to_grayscale=True):
    # load image
    if img_path.endswith('.tif'):
        img = io.imread(img_path)
        smallest_axis = locate_smallest_axis(img)
        if len(img.shape) == 3 and img.shape[0] >= 1:
            # assume the axis with the smallest size is the z axis
            img = np.moveaxis(img, smallest_axis, -1)  # move the z-axis to the last dimension
        # convert numpy img to tensor
        img = torch.from_numpy(img)
        # todo normalization
        img = img / torch.max(img)
    else:
        img = Image.open(img_path)

    if to_grayscale and 'convert' in dir(img):
        img = img.convert('L')

    # # convert to tensor
    # if not isinstance(img, torch.Tensor):
    #     img = transforms.ToTensor()(img).squeeze()

    return img


def locate_smallest_axis(img):
    """
    find the index of the smallest axis (usually the z-axis is the smallest)
    :param img: a torch tensor
    :return: an index
    """
    dims = img.shape
    dim_list = list(dims)
    smallest_axis = dim_list.index(min(dim_list))

    return smallest_axis


def is_greyscale(in_img):
    # todo verify this function
    # assume ndarray is gray scale
    if isinstance(in_img, np.ndarray) or isinstance(in_img, torch.Tensor):
        return True

    if isinstance(in_img, tio.data.subject.Subject):
        return True

    in_img_rgb = in_img.convert('RGB')
    in_img_rgb_np = np.array(in_img_rgb)
    in_img_np = np.array(in_img)

    if in_img_np.shape == in_img_rgb_np.shape:
        if np.sum(in_img_np - in_img_rgb_np) == 0:
            return False
        else:
            return True
    else:
        return False


def resize_tensor(
        tensor_in,
        scale_factor=None,
        output_shape=None,
        kernel=None,
        antialiasing=True,
        kernel_shift_flag=False
        ):
    # convert it to tensor if it is an PIL image
    if isinstance(tensor_in, PIL.Image.Image):
        tensor_in = transforms.ToTensor()(tensor_in)
        tensor_in = torch.squeeze(tensor_in)

    if len(tensor_in.shape) == 3:
        three_d_image = True
    else:
        three_d_image = False
    # First standardize values and fill missing arguments (if needed) by deriving scale from output shape or vice versa
    scale_factor, output_shape = fix_scale_and_size(tuple(tensor_in.shape), output_shape, scale_factor, three_d_image)

    # For a given numeric kernel case, just do convolution and sub-sampling (downscaling only)
    if type(kernel) == np.ndarray and scale_factor[0] <= 1:
        # todo update numerical_kernel for tensor input
        return numeric_kernel(tensor_in, kernel, scale_factor, output_shape, kernel_shift_flag)

    # Choose interpolation method, each method has the matching kernel size
    method, kernel_width = {
        "cubic": (cubic, 4.0),
        "lanczos2": (lanczos2, 4.0),
        "lanczos3": (lanczos3, 6.0),
        "box": (box, 1.0),
        "linear": (linear, 2.0),
        None: (cubic, 4.0)  # set default interpolation method as cubic
    }.get(kernel)

    # Antialiasing is only used when downscaling
    antialiasing *= (scale_factor[0] < 1)

    # Sort indices of dimensions according to scale of each dimension. since we are going dim by dim this is efficient
    sorted_dims = np.argsort(np.array(scale_factor)).tolist()

    # Iterate over dimensions to calculate local weights for resizing and resize each time in one direction
    tensor_out = tensor_in.clone()
    for dim in sorted_dims:
        # No point doing calculations for scale-factor 1. nothing will happen anyway
        if scale_factor[dim] == 1.0:
            continue

        # for each coordinate (along 1 dim), calculate which coordinates in the input image affect its result and the
        # weights that multiply the values there to get its result.
        weights, field_of_view = contributions(tensor_in.shape[dim], output_shape[dim], scale_factor[dim],
                                               method, kernel_width, antialiasing)

        # Use the affecting position values and the set of weights to calculate the result of resizing along this 1 dim
        tensor_out = resize_tensor_along_dim(tensor_out, dim, weights, field_of_view)

    return tensor_out


# Note that imresize only act upon the first two dimension (x, y) of the input image
def imresize(
        im,
        scale_factor=None,
        output_shape=None,
        kernel=None,
        antialiasing=True,
        kernel_shift_flag=False,
        three_d_image=False
        ):
    # First standardize values and fill missing arguments (if needed) by deriving scale from output shape or vice versa
    scale_factor, output_shape = fix_scale_and_size(im.shape, output_shape, scale_factor, three_d_image)

    # For a given numeric kernel case, just do convolution and sub-sampling (downscaling only)
    if type(kernel) == np.ndarray and scale_factor[0] <= 1:
        return numeric_kernel(im, kernel, scale_factor, output_shape, kernel_shift_flag)

    # Choose interpolation method, each method has the matching kernel size
    method, kernel_width = {
        "cubic": (cubic, 4.0),
        "lanczos2": (lanczos2, 4.0),
        "lanczos3": (lanczos3, 6.0),
        "box": (box, 1.0),
        "linear": (linear, 2.0),
        None: (cubic, 4.0)  # set default interpolation method as cubic
    }.get(kernel)

    # Antialiasing is only used when downscaling
    antialiasing *= (scale_factor[0] < 1)

    # Sort indices of dimensions according to scale of each dimension. since we are going dim by dim this is efficient
    sorted_dims = np.argsort(np.array(scale_factor)).tolist()

    # Iterate over dimensions to calculate local weights for resizing and resize each time in one direction
    if isinstance(im, torch.Tensor):
        im = im.detach().cpu()

    out_im = np.copy(im)
    for dim in sorted_dims:
        # No point doing calculations for scale-factor 1. nothing will happen anyway
        if scale_factor[dim] == 1.0:
            continue

        # for each coordinate (along 1 dim), calculate which coordinates in the input image affect its result and the
        # weights that multiply the values there to get its result.
        weights, field_of_view = contributions(im.shape[dim], output_shape[dim], scale_factor[dim],
                                               method, kernel_width, antialiasing)

        # Use the affecting position values and the set of weights to calculate the result of resizing along this 1 dim
        out_im = resize_along_dim(out_im, dim, weights, field_of_view)

    return out_im


class RotationTransform:
    """
    Rotate the input image by one of the given angles
    """
    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)


def imresize3(im, scale_factor=None, output_shape=None, kernel=None, antialiasing=True, kernel_shift_flag=False):
    # resize 3d image
    return imresize(im, scale_factor, output_shape, kernel, antialiasing, kernel_shift_flag, three_d_image=False)


def fix_scale_and_size(input_shape, output_shape, scale_factor, three_d_image=False):
    # First fixing the scale-factor (if given) to be standardized the function expects (a list of scale factors in the
    # same size as the number of input dimensions)
    if scale_factor is not None:
        # By default, if scale-factor is a scalar we assume 2d resizing and duplicate it.
        if np.isscalar(scale_factor) and three_d_image is False:
            scale_factor = [scale_factor, scale_factor]
        elif np.isscalar(scale_factor) and three_d_image is True:
            scale_factor = [scale_factor, scale_factor, scale_factor]

        # We extend the size of scale-factor list to the size of the input by assigning 1 to all the unspecified scales
        scale_factor = list(scale_factor)
        scale_factor.extend([1] * (len(input_shape) - len(scale_factor)))

    # Fixing output-shape (if given): extending it to the size of the input-shape, by assigning the original input-size
    # to all the unspecified dimensions
    if output_shape is not None:
        output_shape = list(np.uint(np.array(output_shape))) + list(input_shape[len(output_shape):])

    # Dealing with the case of non-give scale-factor, calculating according to output-shape. note that this is
    # sub-optimal, because there can be different scales to the same output-shape.
    if scale_factor is None:
        scale_factor = 1.0 * np.array(output_shape) / np.array(input_shape)

    # Dealing with missing output-shape. calculating according to scale-factor
    if output_shape is None:
        output_shape = np.uint(np.ceil(np.array(input_shape) * np.array(scale_factor)))

    return scale_factor, output_shape


def contributions(in_length, out_length, scale, kernel, kernel_width, antialiasing):
    # This function calculates a set of 'filters' and a set of field_of_view that will later on be applied
    # such that each position from the field_of_view will be multiplied with a matching filter from the
    # 'weights' based on the interpolation method and the distance of the sub-pixel location from the pixel centers
    # around it. This is only done for one dimension of the image.

    # When anti-aliasing is activated (default and only for downscaling) the receptive field is stretched to size of
    # 1/sf. this means filtering is more 'low-pass filter'.
    fixed_kernel = (lambda arg: scale * kernel(scale * arg)) if antialiasing else kernel
    kernel_width *= 1.0 / scale if antialiasing else 1.0

    # These are the coordinates of the output image
    out_coordinates = np.arange(1, out_length+1)

    # These are the matching positions of the output-coordinates on the input image coordinates.
    # Best explained by example: say we have 4 horizontal pixels for HR and we downscale by SF=2 and get 2 pixels:
    # [1,2,3,4] -> [1,2]. Remember each pixel number is the middle of the pixel.
    # The scaling is done between the distances and not pixel numbers (the right boundary of pixel 4 is transformed to
    # the right boundary of pixel 2. pixel 1 in the small image matches the boundary between pixels 1 and 2 in the big
    # one and not to pixel 2. This means the position is not just multiplication of the old pos by scale-factor).
    # So if we measure distance from the left border, middle of pixel 1 is at distance d=0.5, border between 1 and 2 is
    # at d=1, and so on (d = p - 0.5).  we calculate (d_new = d_old / sf) which means:
    # (p_new-0.5 = (p_old-0.5) / sf)     ->          p_new = p_old/sf + 0.5 * (1-1/sf)
    match_coordinates = 1.0 * out_coordinates / scale + 0.5 * (1 - 1.0 / scale)

    # This is the left boundary to start multiplying the filter from, it depends on the size of the filter
    left_boundary = np.floor(match_coordinates - kernel_width / 2)

    # Kernel width needs to be enlarged because when covering has sub-pixel borders, it must 'see' the pixel centers
    # of the pixels it only covered a part from. So we add one pixel at each side to consider (weights can zeroize them)
    expanded_kernel_width = np.ceil(kernel_width) + 2

    # Determine a set of field_of_view for each each output position, these are the pixels in the input image
    # that the pixel in the output image 'sees'. We get a matrix whos horizontal dim is the output pixels (big) and the
    # vertical dim is the pixels it 'sees' (kernel_size + 2)
    field_of_view = np.squeeze(np.uint(np.expand_dims(left_boundary, axis=1) + np.arange(expanded_kernel_width) - 1))

    # Assign weight to each pixel in the field of view. A matrix whos horizontal dim is the output pixels and the
    # vertical dim is a list of weights matching to the pixel in the field of view (that are specified in
    # 'field_of_view')
    weights = fixed_kernel(1.0 * np.expand_dims(match_coordinates, axis=1) - field_of_view - 1)

    # Normalize weights to sum up to 1. be careful from dividing by 0
    sum_weights = np.sum(weights, axis=1)
    sum_weights[sum_weights == 0] = 1.0
    weights = 1.0 * weights / np.expand_dims(sum_weights, axis=1)

    # We use this mirror structure as a trick for reflection padding at the boundaries
    mirror = np.uint(np.concatenate((np.arange(in_length), np.arange(in_length - 1, -1, step=-1))))
    field_of_view = mirror[np.mod(field_of_view, mirror.shape[0])]

    # Get rid of  weights and pixel positions that are of zero weight
    non_zero_out_pixels = np.nonzero(np.any(weights, axis=0))
    weights = np.squeeze(weights[:, non_zero_out_pixels])
    field_of_view = np.squeeze(field_of_view[:, non_zero_out_pixels])

    # Final products are the relative positions and the matching weights, both are output_size X fixed_kernel_size
    return weights, field_of_view


def resize_tensor_along_dim(tenser_in, dim, weights, field_of_view):
    # To be able to act on each dim, we swap so that dim 0 is the wanted dim to resize
    tmp_tensor_in = torch.transpose(tenser_in, dim, 0)
    unchanged_dimensions_shape = list(tmp_tensor_in.shape)[1:]

    # We add singleton dimensions to the weight matrix so we can multiply it with the big tensor we get for
    # tmp_im[field_of_view.T], (bsxfun style)
    weights = np.reshape(weights.T, list(weights.T.shape) + (np.ndim(tenser_in) - 1) * [1])

    # This is a bit of a complicated multiplication: tmp_im[field_of_view.T] is a tensor of order image_dims+1.
    # for each pixel in the output-image it matches the positions the influence it from the input image (along 1 dim
    # only, this is why it only adds 1 dim to the shape). We then multiply, for each pixel, its set of positions with
    # the matching set of weights. we do this by this big tensor element-wise multiplication (MATLAB bsxfun style:
    # matching dims are multiplied element-wise while singletons mean that the matching dim is all multiplied by the
    # same number
    field_of_view = field_of_view.astype('int32')
    new_dims = field_of_view.T.shape[0]
    leading_dimensions_shape = [new_dims, field_of_view.T.shape[1]]
    new_tensor_shape = tuple(leading_dimensions_shape + unchanged_dimensions_shape)
    # new_tensor_shape = (new_dims, field_of_view.T.shape[1], tmp_tensor_in.shape[dim])
    if tmp_tensor_in.device.type == 'cpu':
        new_tensor = torch.zeros(new_tensor_shape)
    else:
        weights = torch.from_numpy(weights).float().to(tmp_tensor_in.device)
        new_tensor = torch.cuda.FloatTensor(*new_tensor_shape).fill_(0)

    if len(new_tensor.shape) == 3:
        for i in range(new_dims):
            new_tensor[i, :, :] = tmp_tensor_in[field_of_view.T[i]]
    elif len(new_tensor.shape) == 4:
        for i in range(new_dims):
            new_tensor[i, :, :, :] = tmp_tensor_in[field_of_view.T[i]]
    else:
        raise ValueError('Incorrect tensor dimensions. Please check the dimensions of the new tensor.')

    tmp_out_im = torch.sum(new_tensor * weights, dim=0)

    # Finally we swap back the axes to the original order
    return torch.transpose(tmp_out_im, dim, 0)


def resize_along_dim(im, dim, weights, field_of_view):
    # To be able to act on each dim, we swap so that dim 0 is the wanted dim to resize
    tmp_im = np.swapaxes(im, dim, 0)

    # We add singleton dimensions to the weight matrix so we can multiply it with the big tensor we get for
    # tmp_im[field_of_view.T], (bsxfun style)
    weights = np.reshape(weights.T, list(weights.T.shape) + (np.ndim(im) - 1) * [1])

    # This is a bit of a complicated multiplication: tmp_im[field_of_view.T] is a tensor of order image_dims+1.
    # for each pixel in the output-image it matches the positions the influence it from the input image (along 1 dim
    # only, this is why it only adds 1 dim to the shape). We then multiply, for each pixel, its set of positions with
    # the matching set of weights. we do this by this big tensor element-wise multiplication (MATLAB bsxfun style:
    # matching dims are multiplied element-wise while singletons mean that the matching dim is all multiplied by the
    # same number
    tmp_out_im = np.sum(tmp_im[field_of_view.T] * weights, axis=0)

    # Finally we swap back the axes to the original order
    return np.swapaxes(tmp_out_im, dim, 0)


def back_project_tensor(y_sr, y_lr, down_kernel, up_kernel, sf=None):
    """
    Use back projection technique to reduce super resolution error
    :param y_sr:
    :param y_lr:
    :param down_kernel:
    :param up_kernel:
    :param sf:
    :return:
    """
    y_sr_low_res_projection = resize_tensor(y_sr,
                                            scale_factor=1.0/sf,
                                            output_shape=y_lr.shape,
                                            kernel=down_kernel)
    y_sr += resize_tensor(y_lr - y_sr_low_res_projection,
                          scale_factor=sf,
                          output_shape=y_sr.shape,
                          kernel=up_kernel)

    return torch.clamp(y_sr, 0, 1)


def numeric_kernel(im, kernel, scale_factor, output_shape, kernel_shift_flag):
    # See kernel_shift function to understand what this is
    if kernel_shift_flag:
        kernel = kernel_shift(kernel, scale_factor)

    # First run a correlation (convolution with flipped kernel)
    out_im = np.zeros_like(im)
    # if len(scale_factor) == 2:
    #     for channel in range(np.ndim(im)):
    #         out_im[:, :, channel] = filters.correlate(im[:, :, channel], kernel)
    # else:
    out_im = filters.correlate(im, kernel)

    # Then subsample and return
    if len(scale_factor) == 2:
        return out_im[np.round(np.linspace(0, im.shape[0] - 1 / scale_factor[0], output_shape[0])).astype(int)[:, None],
                      np.round(np.linspace(0, im.shape[1] - 1 / scale_factor[1], output_shape[1])).astype(int)]
    else:
        return out_im[
                # 1st dim
                np.round(np.linspace(0, im.shape[0] - 1 / scale_factor[0], output_shape[0])).astype(int)[:, None, None],
                # 2nd dim
                np.round(np.linspace(0, im.shape[1] - 1 / scale_factor[1], output_shape[1])).astype(int)[None, :, None],
                # 3rd dim
                np.round(np.linspace(0, im.shape[2] - 1 / scale_factor[2], output_shape[2])).astype(int)
               ]


# todo implement a 3d version of this function
def kernel_shift(kernel, sf):
    # There are two reasons for shifting the kernel:
    # 1. Center of mass is not in the center of the kernel which creates ambiguity. There is no possible way to know
    #    the degradation process included shifting so we always assume center of mass is center of the kernel.
    # 2. We further shift kernel center so that top left result pixel corresponds to the middle of the sfXsf first
    #    pixels. Default is for odd size to be in the middle of the first pixel and for even sized kernel to be at the
    #    top left corner of the first pixel. that is why different shift size needed between od and even size.
    # Given that these two conditions are fulfilled, we are happy and aligned, the way to test it is as follows:
    # The input image, when interpolated (regular bicubic) is exactly aligned with ground truth.

    # First calculate the current center of mass for the kernel
    current_center_of_mass = measurements.center_of_mass(kernel)

    # The second ("+ 0.5 * ....") is for applying condition 2 from the comments above
    wanted_center_of_mass = np.array(kernel.shape) / 2 + 0.5 * (sf - (kernel.shape[0] % 2))

    # Define the shift vector for the kernel shifting (x,y)
    shift_vec = wanted_center_of_mass - current_center_of_mass

    # Before applying the shift, we first pad the kernel so that nothing is lost due to the shift
    # (biggest shift among dims + 1 for safety)
    kernel = np.pad(kernel, np.int(np.ceil(np.max(shift_vec))) + 1, 'constant')

    # Finally shift the kernel and return
    return interpolation.shift(kernel, shift_vec)


# These next functions are all interpolation methods. x is the distance from the left pixel center


def cubic(x):
    absx = np.abs(x)
    absx2 = absx ** 2
    absx3 = absx ** 3
    return ((1.5*absx3 - 2.5*absx2 + 1) * (absx <= 1) +
            (-0.5*absx3 + 2.5*absx2 - 4*absx + 2) * ((1 < absx) & (absx <= 2)))


def lanczos2(x):
    return (((np.sin(pi*x) * np.sin(pi*x/2) + np.finfo(np.float32).eps) /
             ((pi**2 * x**2 / 2) + np.finfo(np.float32).eps))
            * (abs(x) < 2))


def box(x):
    return ((-0.5 <= x) & (x < 0.5)) * 1.0


def lanczos3(x):
    return (((np.sin(pi*x) * np.sin(pi*x/3) + np.finfo(np.float32).eps) /
            ((pi**2 * x**2 / 3) + np.finfo(np.float32).eps))
            * (abs(x) < 3))


def linear(x):
    return (x + 1) * ((-1 <= x) & (x < 0)) + (1 - x) * ((0 <= x) & (x <= 1))


def show_tensor(tensor_in, title=None):
    import torchvision.transforms

    to_pil = torchvision.transforms.ToPILImage()
    img = to_pil(tensor_in*10)
    img.show(title=title)


def evaluate_error_of_imgs(ref_img, sr_img, lr_img):
    # mse, ssim etc.
    # format output
    final_output_np = sr_img

    # ref_img = np.asarray(ref_img).astype(final_output_np.dtype)
    if locate_smallest_axis(ref_img) != locate_smallest_axis(final_output_np):
        # move the z axis to the last
        final_output_np = np.moveaxis(
            final_output_np, locate_smallest_axis(final_output_np), -1
        )

    ref_img_normalized = ref_img / np.max(ref_img)

    # interpolation
    original_lr_img = lr_img
    interp_img = resize_tensor(original_lr_img, 4, kernel='cubic')
    interp_img = interp_img.numpy()
    if locate_smallest_axis(ref_img) != locate_smallest_axis(interp_img):
        # move the z axis to the last
        interp_img = np.moveaxis(
            interp_img, locate_smallest_axis(interp_img), -1
        )
    interp_img = interp_img.astype('float32')
    interp_img_normalized = interp_img / np.max(interp_img)
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


class Logger:
    def __init__(self, path, file_name='log.txt'):
        self.console = sys.stdout
        os.makedirs(path, exist_ok=True)
        self.file = open(
            os.path.join(path,file_name), 'w'
        )

    def write(self, message):
        self.console.write(message)
        self.file.write(message)

    def flush(self):
        self.console.flush()
        self.file.flush()


if __name__ == "__main__":
    from time import sleep

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

    data_set = ZVisionDataset(
        configs=configs, transform=composed_transform
    )

    data_loader = DataLoader(data_set, batch_size=configs['batch_size'], num_workers=configs['num_workers'])

    for i in range(10):
        data_iter = iter(data_loader)
        sample_instance = data_iter.__next__()
        image = torch.squeeze(sample_instance["img"])
        lr_image = torch.squeeze(sample_instance["img_lr"])
        # convert img for display, and pick the first img
        image_numpy = image.numpy()
        convolved_image_numpy = lr_image.numpy()

        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(image_numpy, cmap='gray')
        ax1.set_title('Original image')
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.imshow(convolved_image_numpy, cmap='gray')
        ax2.set_title('LR image')
        fig.show()
        sleep(2)