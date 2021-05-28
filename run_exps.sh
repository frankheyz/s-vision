#!/bin/bash
python3 train_model.py -c 3d -m 'up' -i "/home/heyz/code/z-vision/images/low_density_128_128_15.tif"
python3 train_model.py -c 3d -m 'up' -i "/home/heyz/code/z-vision/images/low_density_128_128_20.tif"
python3 train_model.py -c 3d -m 'up' -i "/home/heyz/code/z-vision/images/low_density_128_128_25.tif"
python3 train_model.py -c 3d -m 'up' -i "/home/heyz/code/z-vision/images/low_density_128_128_30.tif"

echo all experiments finished