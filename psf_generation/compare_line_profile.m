%% load data
clc; clear;
original = double(read_tiff('1024_1024_60_gray.tif'));
sr = double(read_tiff('256_256_15_grayX2.00X2.00X2.00X2.00X2.00X2.00.tif'));
lr = double(read_tiff('256_256_15_gray.tif'));

max_original = max(original(:));

lr_interp_nn = imresize3(lr, 4, 'nearest');
lr_interp = imresize3(lr, 4);

% Normalization
original = original ./ max_original;
sr = sr ./ max_original;
lr_interp = lr_interp ./ max_original;
lr_interp_nn = lr_interp_nn./ max_original;
%% x-y plane
start = [260, 744];
finish = [271, 759];

% line profile
z_pos = 30;
original_xy_slice = original(:,:,z_pos);
sr_xy_slice = sr(:,:,z_pos);
lr_interp_xy_slice = lr_interp(:,:,z_pos);

% error
sr_error_img = original_xy_slice - sr_xy_slice;
lr_interp_error_img = original_xy_slice - lr_interp_xy_slice;
original_error_img = original_xy_slice - original_xy_slice ;

% figure,
% imshow(original_error_img, [0 0.5]);colormap(jet);colorbar
% figure,
% imshow(sr_error_img, [0 0.5]);colormap(jet);colorbar
% figure,
% imshow(lr_interp_error_img , [0 0.5]);colormap(jet);colorbar

original_xy_profile = line_profile(original_xy_slice, start, finish);
sr_xy_profile = line_profile(sr_xy_slice, start, finish);
lr_interp_xy_profile = line_profile(lr_interp_xy_slice, start, finish);


% fitting lateral psf
f_gt_psf_lateral  = fit_psf(original_xy_profile);
f_lr_interp_psf_lateral = fit_psf(lr_interp_xy_profile);
f_sr_psf_lateral = fit_psf(sr_xy_profile);

fwhm_gt_psf_lateral = 2.355*f_gt_psf_lateral.c1;
fwhm_lr_interp_psf_lateral  = 2.355*f_lr_interp_psf_lateral .c1;
fwhm_sr_psf_lateral = 2.355*f_sr_psf_lateral.c1;

figure;
plot(original_xy_profile, 'ro', 'LineWidth', 2);
hold on; grid on;
plot(lr_interp_xy_profile, 'b+', 'LineWidth', 2);
plot(sr_xy_profile, 'k*', 'LineWidth', 2);

plot(f_gt_psf_lateral, 'r');
hold on; grid on;
plot(f_lr_interp_psf_lateral, 'b');
plot(f_sr_psf_lateral, 'k');

legend('Original','Interpolation','Super-resolution', ...
    ['Original (Fitted) \sigma = ' num2str(fwhm_gt_psf_lateral, 3)], ...
    ['Interp. (Fitted) \sigma = ' num2str(fwhm_lr_interp_psf_lateral, 3)],...
    ['SR (Fitted) \sigma = ' num2str(fwhm_sr_psf_lateral,  3)])
xlabel('Pixels')
ylabel('Amplitude (a.u.)')
title('Lateral line profile')
set(gca,'FontSize',18,'FontWeight', 'bold')
hline = findobj(gcf, 'type', 'line');
set(hline,'LineWidth',2)
%% y-z plane
start = [21, 197];
finish = [45, 197];

% line profile
x_pos = 475;
original_yz_slice = squeeze(original(x_pos, :, :))';
sr_yz_slice = squeeze(sr(x_pos, :, :))';
lr_interp_yz_slice = squeeze(lr_interp(x_pos, :, :))';
lr_interp_nn_yz_slice = squeeze(lr_interp_nn(x_pos, :, :))';

% error
sr_error_img_yz = original_yz_slice - sr_yz_slice;
lr_interp_error_img_yz = original_yz_slice - lr_interp_yz_slice;
original_error_img_yz = original_yz_slice - original_yz_slice ;

% figure,
% imshow(original_error_img_yz, [0 0.5]);colormap(jet);colorbar
% figure,
% imshow(sr_error_img_yz, [0 0.5]);colormap(jet);colorbar
% figure,
% imshow(lr_interp_error_img_yz , [0 0.5]);colormap(jet);colorbar

original_yz_profile = line_profile(original_yz_slice, start, finish);
sr_yz_profile = line_profile(sr_yz_slice, start, finish);
lr_interp_yz_profile = line_profile(lr_interp_yz_slice, start, finish);
% fitting axial psf
f_gt_psf_axial  = fit_psf(original_yz_profile);
f_lr_interp_psf_axial = fit_psf(lr_interp_yz_profile);
f_sr_psf_axial = fit_psf(sr_yz_profile);

fwhm_gt_psf_axial = 2.355*f_gt_psf_axial.c1;
fwhm_lr_interp_psf_axial  = 2.355*f_lr_interp_psf_axial .c1;
fwhm_sr_psf_axial = 2.355*f_sr_psf_axial.c1;
% plot
figure;
plot(original_yz_profile, 'ro', 'LineWidth', 2);
hold on; grid on;
plot(lr_interp_yz_profile, 'b+', 'LineWidth', 2);
plot(sr_yz_profile, 'k*', 'LineWidth', 2);

plot(f_gt_psf_axial, 'r');
hold on; grid on;
plot(f_lr_interp_psf_axial, 'b');
plot(f_sr_psf_axial, 'k');

legend('Original','Interpolation','Super-resolution', ...
    ['Original (Fitted) \sigma = ' num2str(fwhm_gt_psf_axial, 3)], ...
    ['Interp. (Fitted) \sigma = ' num2str(fwhm_lr_interp_psf_axial, 3)],...
    ['SR (Fitted) \sigma = ' num2str(fwhm_sr_psf_axial,  3)])
xlabel('Pixels')
ylabel('Amplitude (a.u.)')
title('Axial line profile')
set(gca,'FontSize',18,'FontWeight', 'bold')
hline = findobj(gcf, 'type', 'line');
set(hline,'LineWidth',2)
