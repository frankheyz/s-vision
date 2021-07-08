%%
clear;clc
input = read_tiff('D:\data\20210623\tif_data\test\test_lr_3d\thy1_zone3_new_center_cropped_64_64_8.tif');
ref = read_tiff('D:\data\20210623\tif_data\test\test_hr\thy1_zone3_new_center_cropped_256_256_32.tif');
small_fov = read_tiff('D:\codes\z-vision\results\fov_tests\20210701_10_23_38\output_image\thy1_zone3_new_center_cropped_64_64_8X2.00X2.00X2.00X2.00X2.00X2.00.tif');
large_fov = read_tiff('D:\codes\z-vision\results\fov_tests\20210701_10_23_43\output_image\thy1_zone3_new_center_cropped_64_64_8X2.00X2.00X2.00X2.00X2.00X2.00.tif');

%%
z_slice_list = [30 3]; % 3 and 30
z_no = numel(z_slice_list);
f1=figure;
for i =1:z_no
    z_slice = z_slice_list(i);
    small_xy = double(small_fov(:,:,z_slice));
    small_xy_max = max(small_xy(:));

    large_xy = double(large_fov(:,:,z_slice));
    large_xy_max = max(large_xy(:));

    input_xy = double(input(:,:,round(z_slice/4)));
    input_xy_max = max(input_xy(:));

    ref_xy = double(ref(:,:,round(z_slice)));
    ref_xy_max = max(ref_xy(:));

    norm_max = max([input_xy_max ref_xy_max small_xy_max large_xy_max]);

    small_xy_norm = small_xy./norm_max;
    large_xy_norm = large_xy./norm_max;

    input_xy_norm = input_xy./norm_max;
    ref_xy_norm = ref_xy./norm_max;
    
    subaxis(z_no,4,1+4*(i-1),'Spacing',0.01);
    reduction = 15;
    imshow(input_xy_norm(reduction:end-reduction,reduction:end-reduction), [0 1])

    subaxis(z_no,4,2+4*(i-1))
    imshow(small_xy_norm(reduction*4:end-reduction*4,reduction*4:end-reduction*4), [0 1])

    subaxis(z_no,4,3+4*(i-1))
    imshow(large_xy_norm(reduction*4:end-reduction*4,reduction*4:end-reduction*4), [0 1])

    subaxis(z_no,4,4+4*(i-1))
    imshow(ref_xy_norm(reduction*4:end-reduction*4,reduction*4:end-reduction*4), [0 1])
    colormap(hot)
end

darkBackground(f1)
set(gcf, 'inverthardcopy', 'off') 
%%
y_slice = 121;

small_yz = squeeze(double(small_fov(y_slice,:,:)));
[y_size, z_size] = size(small_yz);
% small_yz = imresize(small_yz, [y_size, z_size*2]);
small_yz_max = max(small_yz(:));

input_yz = squeeze(double(input(round(y_slice/4), :, :)));
% input_yz = imresize(input_yz, [y_size, z_size*2]);
input_yz_max = max(input_yz(:));

ref_yz = squeeze(double(ref(round(y_slice), :, :)));
% ref_yz = imresize(ref_yz, [y_size, z_size*2]);
ref_yz_max = max(ref_yz(:));

large_yz = squeeze(double(large_fov(y_slice,:,:)));
% large_yz = imresize(large_yz, [y_size, z_size*2]);
large_yz_max = max(large_yz(:));

norm_max = max([input_yz_max small_yz_max large_yz_max ref_yz_max]);

small_yz_norm = small_yz./norm_max;
large_yz_norm = large_yz./norm_max;

input_yz_norm = input_yz./norm_max;
ref_yz_norm = ref_yz./norm_max;

f2=figure;
subaxis(4,1,1,'Spacing',0.05);
imshow(input_yz_norm', [0 1])

subaxis(4,1,2)
imshow(small_yz_norm', [0 1])

subaxis(4,1,3)
imshow(large_yz_norm', [0 1])

subaxis(4,1,4)
imshow(ref_yz_norm', [0 1])
colormap(hot);

darkBackground(f2)
set(gcf, 'inverthardcopy', 'off') 