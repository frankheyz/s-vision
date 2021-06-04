%% load data
clc; clear;
original = read_tiff('yj_1024_1024_48.tif');
sr = read_tiff('yj_256_256_12X2.00X2.00X2.00X2.00X2.00X2.00.tif');
lr = read_tiff('yj_256_256_12.tif');

lr_interp = imresize3(lr, 4);

%% x-y plane
start = [213, 680];
finish = [222, 695];

% line profile
z_pos = 1;
original_xy_slice = original(:,:,z_pos);
sr_xy_slice = sr(:,:,z_pos);
lr_interp_xy_slice = lr_interp(:,:,z_pos);

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

figure(1);
plot(original_xy_profile,'LineWidth', 2);
hold on; grid on;
plot(lr_interp_xy_profile,'LineWidth', 2);
plot(sr_xy_profile,'LineWidth', 2);

legend('Original', 'SR', 'Interp.')
xlabel('Pixels')
ylabel('Amplitude (a.u.)')
set(gca,'FontSize',20,'FontWeight', 'bold')
legend( ...
    ['Original (Fitted) \sigma = ' num2str(fwhm_gt_psf_lateral, 3)], ...
    ['Interpolation (Fitted) \sigma = ' num2str(fwhm_lr_interp_psf_lateral, 3)],...
    ['Super-resolution (Fitted) \sigma = ' num2str(fwhm_sr_psf_lateral, 3)])
%% y-z plane
start = [21, 197];
finish = [45, 197];

% line profile
x_pos = 475;
original_yz_slice = squeeze(original(x_pos, :, :))';
sr_yz_slice = squeeze(sr(x_pos, :, :))';
lr_interp_yz_slice = squeeze(lr_interp(x_pos, :, :))';

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
figure(2);
plot(original_yz_profile,'LineWidth', 2);
hold on; grid on;
plot(lr_interp_yz_profile,'LineWidth', 2);
plot(sr_yz_profile,'LineWidth', 2);

xlabel('Pixels')
ylabel('Amplitude (a.u.)')
set(gca,'FontSize',20,'FontWeight', 'bold')
legend( ...
    ['Original (Fitted) \sigma = ' num2str(fwhm_gt_psf_axial, 3)], ...
    ['Interpolation (Fitted) \sigma = ' num2str(fwhm_lr_interp_psf_axial, 3)],...
    ['Super-resolution (Fitted) \sigma = ' num2str(fwhm_sr_psf_axial, 3)])

