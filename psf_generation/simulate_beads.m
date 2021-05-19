%% down sample to create LR images
clear
image_x = 512;
image_y = 512;
image_z = 1;

% fix random seed
rng(1);

beads_img = zeros(image_x, image_y, image_z);
number_of_beads = 2000;
rand_idx = randperm(image_x * image_y * image_z);
rand_idx = rand_idx(1:number_of_beads);
beads_img(rand_idx) = 1;
beads_img = beads_img*255;  % convert to 8 bits
% figure, imshow3D(beads_img, []);

%% low density
ratio = 0.5;
rand_idx_mid_density = rand_idx(1:round(ratio*numel(rand_idx)));
rand_idx_low_density = rand_idx(1:round(ratio*ratio*ratio*ratio*numel(rand_idx)));

beads_img_mid_density = zeros(image_x, image_y, image_z);
beads_img_low_density = zeros(image_x, image_y, image_z);

beads_img_mid_density(rand_idx_mid_density) = 1;
beads_img_low_density(rand_idx_low_density) = 1;

beads_img_mid_density = beads_img_mid_density * 255;
beads_img_low_density = beads_img_low_density * 255;
%% convolve with psf
psf = fspecial3('gaussian', [13 13 13], [1 1 2.5]);
% convolution
gt_img = convn(beads_img, psf, 'same') ;
mid_density_img = convn(beads_img_mid_density, psf, 'same') ;
low_density_img = convn(beads_img_low_density, psf, 'same') ;

figure,
subplot(1,2,1)
imshow(psf(:,:,7), []);
colorbar
title('Simulated PSF (axial)', 'FontSize', 14,'FontWeight','bold')
subplot(1,2,2)
imshow(squeeze(psf(:,7,:))', []);
colorbar
title('Simulated PSF (lateral)', 'FontSize', 14,'FontWeight','bold')
% figure, imshow3D(gt_img);
% figure, imshow3D(mid_density_img);
% figure, imshow3D(low_density_img);
%% create LR image
lr_size = [image_x/4, image_y/4, image_z/4];

lr_default = imresize3(gt_img, lr_size);
lr_mid_density = imresize3(mid_density_img, lr_size);
lr_low_density = imresize3(low_density_img, lr_size, 'cubic');

% Normalize to 0-1
% normalize original image
gt_img_max = max(gt_img(:));
gt_img_norm = gt_img./gt_img_max;
mid_density_img_norm = mid_density_img./gt_img_max;
low_density_img_norm = low_density_img./gt_img_max;
% normalization of downsampled images with respect to the ground truth
ground_truth_max =  max(lr_default(:));
lr_default_norm = lr_default ./ground_truth_max;
lr_mid_density_norm = lr_mid_density ./ground_truth_max;
lr_low_density_norm = lr_low_density ./ground_truth_max;
% figure, imshow3D(lr_default, []);

%% write 2d image first
[h,w,l] = size(lr_low_density_norm);
[h_original,w_original,l_original] = size(low_density_img_norm);
imwrite(lr_low_density_norm(:,:,round(l/2)), 'low_density_z8_lr.png')
imwrite(low_density_img_norm(:,:,round(l_original/2)), 'low_density_z32.png')
% 
% imwrite(lr_default(:,:,round(l/2)), 'high_density_lr.png')
% imwrite(lr_default_norm(:,:,round(l_original/2)), 'high_density.png')
%% write 3d image
write_tiff(lr_low_density_norm, 'low_density_z8_lr_3d.tif')
write_tiff(low_density_img_norm, 'low_density_z32_3d.tif')