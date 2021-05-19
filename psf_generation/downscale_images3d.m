%% read img tiff
clear
clc
reference_stacks = double(read_tiff('cardiosphere_ref.tif'));
[x_size, y_size, z_size] = size(reference_stacks);

%% create 3d LR image

% lr_size_256_10 = [256, 256, 10];
% lr_size_512_20 = [512, 512, 20];
% lr_size_1024_40 = [1024, 1024, 40];

% create PSF
psf_size = 13;
psf_l = fspecial3('gaussian', [psf_size psf_size psf_size], [2.5 2.5 5]);
psf_h = fspecial3('gaussian', [psf_size psf_size psf_size], [0.5 0.5 1.25]);

figure, imshow3D([psf_h, psf_l]); title('Compare PSF_h and PSF_l')
% lr_quarter = imresize3(reference_stacks, 0.25, 'Antialiasing', true);

% normalization
max_val = double(max(reference_stacks(:)));
reference_stacks_norm = reference_stacks / max_val;

% convolution
ref_stack_psf_l = convn(reference_stacks_norm, psf_l, 'same') ;
ref_stack_psf_h = convn(reference_stacks_norm, psf_h, 'same') ;

figure, imshow3D([ref_stack_psf_h ref_stack_psf_l])
%% save 3d tiff image
% down sampling the image filtered by psf_l
lr_psf_l_128_8 = imresize3(ref_stack_psf_l, 0.5, 'antialiasing', false);

% save reference
write_tiff(reference_stacks_norm, 'cardiosphere_ref.tif');
 
% save lr image
write_tiff(lr_psf_l_128_8, 'cardiosphere_psf_l_lr.tif')

% save psf
Kernel = psf_l;
save('psf_l_3d.mat', 'Kernel');
% lr_512_20 = imresize3(hr_stacks, lr_size_512_20);
% lr_1024_40 = imresize3(hr_stacks, lr_size_1024_40);

%% 2d projection
hr_2d = sum(reference_stacks, 3);
img_2046 = imresize(hr_2d, [2046 2046]);
img_2046_n = img_2046./max(img_2046(:));

img_1024 = imresize(hr_2d, [1024 1024]);
img_1024_n = img_1024./max(img_1024(:));

img_512 = imresize(hr_2d, [512 512]);
img_512_n = img_512./max(img_1024(:));

img_256 = imresize(hr_2d, [256 256]);
img_256_n = img_256./max(img_2046(:));


imwrite(img_2046_n, 'yj_2046.png');
imwrite(img_1024_n, 'yj_1024.png');
imwrite(img_512_n, 'yj_512.png');
imwrite(img_256_n, 'yj_256.png');
% figure, imshow(hr_2d, [])

