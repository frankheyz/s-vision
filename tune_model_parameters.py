# -*- coding: utf-8 -*-
import os
import json
import argparse
import torch
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

from functools import partial
from configs import configs
from configs import configs3D
from train_model import train_model


def main(num_of_samples=50, max_num_epochs=100, gpu_per_trail=2, conf=configs):
    # determine if input is 2d or 3d
    if conf['crop_size'].__len__() == 2:
        conf.update(
            {
                "tune": True,
                "learning_rate": tune.loguniform(1e-5, 1e-2),
                "batch_size": tune.choice([8, 16, 32]),
                "crop_size": tune.choice([(32, 32), (64, 64), (128, 128)]),
                "kernel_depth": tune.choice([4, 8, 16]),
                "kernel_channel_num": tune.choice([16, 32, 64])
            }
        )
    elif conf['crop_size'].__len__() == 3:
        conf.update(
            {
                "tune": True,
                "learning_rate": tune.loguniform(1e-5, 1e-2),
                "batch_size": tune.choice([8, 16, 32]),
                "crop_size": tune.choice(
                    [(32, 32, 4), (64, 64, 4), (128, 128, 4)]
                ),
                "kernel_depth": tune.choice([4, 8, 16]),
                "kernel_channel_num": tune.choice([16, 32, 64])
            }
        )
    else:
        raise ValueError("Incorrect input configs.")

    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=10,
        reduction_factor=2
    )

    reporter = CLIReporter(
        metric_columns=['loss', 'training_iterations']
    )

    result = tune.run(
        partial(train_model,),
        resources_per_trial={'cpu': 2, 'gpu': gpu_per_trail},
        config=conf,
        num_samples=num_of_samples,
        scheduler=scheduler,
        progress_reporter=reporter
    )

    best_trail = result.get_best_trial('loss', 'min', 'last')
    # todo add psf as metric?
    print("Best trail config: {}".format(best_trail.config))
    print("Best trail final validation loss: {}".format(best_trail.last_result['loss']))

    # save the best configs
    out_path = os.path.join(
        configs['save_path'],
        configs['output_configs_dir']
    )
    os.makedirs(out_path, exist_ok=True)

    # save(copy) config for reproducibility
    with open(out_path + "best_configs.json", 'w') as f:
        json.dump(best_trail.config, f, indent=4)


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description="Train z-vision model.")
    parser.add_argument("-c", "--print_string", help="Input configs.", default="2d")
    args = parser.parse_args()
    input_config = configs if args.print_string == '2d' else configs3D

    torch.manual_seed(0)
    main(
        num_of_samples=50,
        max_num_epochs=100,
        gpu_per_trail=2,
        conf=configs3D
    )
