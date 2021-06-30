#!/bin/bash
zones="3 4 5 7"
lateral_sizes="64 96 128 160 192 224 256"
axial_sizes="8 13 18 23 28 33 38"
array0=($zones)
array1=($lateral_sizes)
array2=($axial_sizes)

zone_count=${#array0[@]}
size_count=${#array1[@]}

for z in $(seq 1 $zone_count); do
  zone=${array0[$z - 1]}

  for i in $(seq 1 $size_count); do
    lateral_lr=${array1[$i - 1]}
    axial_lr=${array2[$i - 1]}

    lateral_hr=$((lateral_lr * 4))
    axial_hr=$((axial_lr * 4))

    lr_prefix='/home/heyz/code/z-vision/images/test_lr/thy1_zone'
    hr_prefix='/home/heyz/code/z-vision/images/test_hr/thy1_zone'

    lr_suffix="${zone}_new_center_cropped_${lateral_lr}_${lateral_lr}_${axial_lr}.tif"
    hr_suffix="${zone}_new_center_cropped_${lateral_hr}_${lateral_hr}_${axial_hr}.tif"

    lr_file="${lr_prefix}${lr_suffix}"
    hr_file="${hr_prefix}${hr_suffix}"
    python3 train_model.py -c 3d -m 'up' -i $lr_file -r $hr_file
    echo "${lr_file} and ${hr_file} finished"
  done
done

echo "All FOV tests finished."
