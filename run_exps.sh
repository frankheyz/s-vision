#!/bin/bash
for i in 10 15 20 25 30
do
	img='/home/heyz/code/z-vision/images/low_density_128_128_'
	suffix="${i}.tif"
	file_name="${img}${suffix}"

	python3 train_model.py -c 3d -m 'up' -i $file_name

	echo "all experiments finished"
done