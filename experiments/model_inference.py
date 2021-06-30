# -*- coding: utf-8 -*-
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


if __name__ == "__main__":
    old_model_path = '../results/20210629_20_49_43/checkpoint/model.pt'
    upgraded_model_path = '../results/20210629_20_49_43/checkpoint/model.pt'

    models = [
        '../results/20210629_20_49_43/checkpoint/model.pt',
        '../results/20210629_21_00_11/checkpoint/model.pt',
        '../results/20210629_21_55_37/checkpoint/model.pt',
    ]
    for m in models:
        infer_image(
            conf=configs3D,
            image_path='/home/heyz/code/z-vision/images/test_lr'
                       '/thy1_zone3_new_center_cropped_96_96_13.tif',
            model_path=m,
            model='new'
        )