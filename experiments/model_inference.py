# -*- coding: utf-8 -*-
import os
import time
import torch
from zvision import ZVision
from zvision import ZVisionUp
from zvision import ZVisionMini
from zvision import count_parameters
from configs import configs
from configs import configs3D


def infer_image(conf, image_path, model_path, model='old', times=2):
    print('model', model)
    # set gpu
    dev = torch.device(
        "cuda:0" if torch.cuda.is_available() and configs['use_gpu']
        else torch.device("cpu")
    )
    conf['save_path'] = '/home/heyz/code/z-vision/results/fov_tests/' + time.strftime("%Y%m%d_%H_%M_%S",
                                                                                      time.localtime()) + "/"
    out = None
    for i in range(times):
        # load the image and the model
        conf['image_path'] = image_path

        if model == 'old':
            model = ZVision(configs=conf)
        else:
            model = ZVisionMini(configs=conf)

        print("model parameters: ", count_parameters(model))
        model.load_state_dict(torch.load(model_path))

        # model to device
        model.to(dev)
        if dev.type == 'cuda':
            model.dev = dev

        model.eval()
        # inference
        out = model.output()
        image_path = model.output_img_path
        pass

    return out


def test_different_models(model_list, test_image, config):
    for model in model_list:
        infer_image(
            conf=config,
            image_path=test_image,
            model_path=model,
            model='new'
        )


def test_different_images(model, image_list, config):
    for img in image_list:
        infer_image(
            conf=config,
            image_path=img,
            model_path=model,
            model='new'
        )

if __name__ == "__main__":
    # old_modelodel_p_path = '../results/20210629_20_49_43/checkpoint/model.pt'
    # upgraded_math = '../results/20210629_20_49_43/checkpoint/model.pt'

    # models = [
    #     '../results/20210629_20_49_43/checkpoint/model.pt',
    #     '../results/20210629_21_00_11/checkpoint/model.pt',
    #     '../results/20210629_21_55_37/checkpoint/model.pt',
    # ]
    # for m in models:
    #     infer_image(
    #         conf=configs3D,
    #         image_path='/home/heyz/code/z-vision/images/test_lr'
    #                    '/thy1_zone3_new_center_cropped_64_64_8.tif',
    #         model_path=m,
    #         model='new'
    #     )

    image_dir = '/home/heyz/code/z-vision/images/yj_cx3'
    images = os.listdir(image_dir)

    for img in images:
        img_path = os.path.join(image_dir, img)
        infer_image(
            conf=configs3D,
            image_path=img_path,
            model_path='/home/heyz/code/z-vision/results/20210705_19_17_09/checkpoint/model.pt',
            model='new'
        )
