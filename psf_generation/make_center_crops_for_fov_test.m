%% crop center
clc; clear
% zone3,4,5,7
test_file_name = 'thy1_zone4_new';
test_stacks = read_tiff(['D:\data\20210623\tif_data\' test_file_name '.tif']);
test_lr_dir = 'D:\data\20210623\tif_data\test\test_lr\';
test_lr_3d_dir = 'D:\data\20210623\tif_data\test\test_lr_3d\';
test_hr_dir = 'D:\data\20210623\tif_data\test\test_hr\';

lr_lateral_sizes = 64:32:256;
lr_axial_sizes = 8:5:38;

sample_factor = 4;
test_stacks_lr = imresize3(test_stacks, 1/sample_factor);

for i = 1:numel(lr_lateral_sizes)
    
    lr_lateral_size = lr_lateral_sizes(i);
    lr_axial_size = lr_axial_sizes(i);
    
    hr_lateral_size = lr_lateral_size * 4;
    hr_axial_size = lr_axial_size * 4;
    
    hr_crop = center_crop_3d(test_stacks, hr_lateral_size, hr_axial_size);
    write_tiff(hr_crop, [test_hr_dir test_file_name '_center_cropped_' num2str(hr_lateral_size) '_' ...
        num2str(hr_lateral_size) '_' num2str(hr_axial_size) '.tif'])
    
    lr_crop = center_crop_3d(test_stacks_lr, lr_lateral_size, lr_axial_size);
    write_tiff(lr_crop, [test_lr_3d_dir test_file_name '_center_cropped_' num2str(lr_lateral_size) '_' ...
        num2str(lr_lateral_size) '_' num2str(lr_axial_size) '.tif'])
    
end
    
