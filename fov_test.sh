#!/bin/bash
lateral_sizes="64 96 128 160 192 224 256"
axial_sizes="8 13 18 23 28 33 38"
array1=($lateral_sizes)
array2=($axial_sizes)

count=${#array1[@]}

for i in $(seq 1 $count); do
  lateral_lr=${array1[$i - 1]}
  axial_lr=${array2[$i - 1]}

  lateral_hr=$((lateral_lr * 4))
  axial_hr=$((axial_lr * 4))

  lr_prefix='/home/heyz/code/z-vision/images/test_lr/center_cropped_'
  hr_prefix='/home/heyz/code/z-vision/images/test_hr/center_cropped_'

  lr_suffix="${lateral_lr}_${lateral_lr}_${axial_lr}.tif"
  hr_suffix="${lateral_hr}_${lateral_hr}_${axial_hr}.tif"

  lr_file="${lr_prefix}${lr_suffix}"
  hr_file="${hr_prefix}${hr_suffix}"

  python3 train_model.py -c 3d -m 'up' -i $lr_file -r $hr_file
done

echo "All FOV tests finished."