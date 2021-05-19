%% create PSF
clc
clear
img = double(read_tiff('cardiosphere_ref.tif'));
% padding 
img = cat(3,zeros(512,512, 1),img);
% img = img(:,:,1);
[x_size, y_size, z_size] = size(img);

psf_size = 3;
psf_l = fspecial3('gaussian', [psf_size psf_size psf_size], 2);

% Fourier transform
img_f = fftn(img);
psf_l_f = fftn(psf_l, [x_size, y_size, z_size]);

% multiplication
img_psf_f = img_f .* psf_l_f;
if_img_psf_f = ifftn(img_psf_f );

figure, imshow3D([img real(if_img_psf_f)], [])

% error
error = abs(if_img_psf_f) - img;
sum(error(:))


%% 2d
psf_size_2d = 13;
img_2d =double(img(:,:,17));
img_2d_norm = img_2d ./ max(img_2d(:));
psf_l_2d = fspecial('gaussian', psf_size_2d, 2);
img_2d_convolved = conv2(img_2d_norm, psf_l_2d, 'same');
figure, imshow([img_2d_norm img_2d_convolved])

img_2d_convolved_lr = imresize(img_2d_convolved, 0.5, 'antialiasing', false);

% save
Kernel = psf_l_2d;
save('psf_l_2d.mat', 'Kernel');
imwrite(img_2d_norm, 'cardiosphere.png')
imwrite(img_2d_convolved_lr, 'cardiosphere_lr.png')

%% check if direct down sampling 4x is the same as 2x + 2x
img_2d_convolved_down_2 = imresize(img_2d_convolved, 0.5, 'antialiasing', false);
img_2d_convolved_down_4 = imresize(img_2d_convolved_down_2, 0.5, 'antialiasing', false);
diff_img = img_2d_convolved_lr - img_2d_convolved_down_4;
imshow(diff_img, []);