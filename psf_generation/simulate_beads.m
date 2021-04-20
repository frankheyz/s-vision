%% down sample to create LR images
image_x = 1024;
image_y = 1024;
image_z = 16;

beads_img = zeros(image_x, image_y, image_z);
number_of_beads = 2500;
rand_idx = randperm(image_x * image_y * image_z);
rand_idx = rand_idx(1:number_of_beads);
beads_img(rand_idx) = 1;
beads_img = beads_img*256;  % convert to 8 bits
figure, imshow3D(beads_img, []);

%% convolve with psf
psf = fspecial3('gaussian', [13 13 13], [1 1 2.5]);
gt_img = convn(beads_img, psf, 'same') ;


figure, imshow3D(gt_img);

%% create LR image
lr_size = [256, 256, 4];
lr_default = imresize3(gt_img, lr_size);

figure, imshow3D(lr_default, []);

%% write
write_tiff(gt_img, 'hr_beads.tif')
write_tiff(lr_default, 'lr_default.tif')