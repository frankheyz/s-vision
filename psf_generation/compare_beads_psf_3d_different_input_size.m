%% compare 3d images
clear
clc
input_depth = 30;
lr = read_tiff(['low_density_128_128_' num2str(input_depth) '.tif']);
gt = read_tiff('low_density_512_512_60.tif');
sr = read_tiff([
    'low_density_128_128_' num2str(input_depth) ['X2.00X2.00X2.00X2.00X2.00X2.00.tif']
    ]);

lr_interp = imresize3(lr, size(gt));

% get a slice 
gt_slice = gt(:,:,30);
lr_interp_slice = lr_interp(:,:,30);
sr_slice = sr(:,:,30);

%% show
figure,
subplot(2,3,1)
imshow(gt_slice)
title('Original (512 \times 512)', 'FontSize', 14,'FontWeight','bold')

subplot(2,3,2)
imshow(lr_interp_slice)
title('Interpolation x 4 (512 \times 512)', 'FontSize', 14,'FontWeight','bold')

subplot(2,3,3)
imshow(sr_slice)
title('Super-resolution x 4 output (512 \times 512)', 'FontSize', 14,'FontWeight','bold')

selected_x = 256;
selected_y = 235;

sub_x = selected_x - 70: selected_x + 70;
sub_y = selected_y - 70: selected_y +70;

subplot(2,3,4)
imshow(gt_slice(sub_x, sub_y))
title('Original (512 \times 512)', 'FontSize', 14,'FontWeight','bold')

subplot(2,3,5)
imshow(lr_interp_slice(sub_x, sub_y))
title('Interpolation x 4 (512 \times 512)', 'FontSize', 14,'FontWeight','bold')

subplot(2,3,6)
imshow(sr_slice(sub_x, sub_y))
title('Super-resolution x 4 output (512 \times 512)', 'FontSize', 14,'FontWeight','bold')

%% extract lateral psf
row = selected_x;
col = selected_y;
delta_x = 7;
gt_psf = gt_slice(row-delta_x : row+delta_x+1, col);
lr_interp_psf = lr_interp_slice(row-delta_x : row+delta_x+1, col);
sr_psf = sr_slice(row-delta_x : row+delta_x+1, col);

%% fitting lateral psf
f_gt_psf_lateral  = fit_psf(gt_psf);
f_lr_interp_psf_lateral = fit_psf(lr_interp_psf);
f_sr_psf_lateral = fit_psf(sr_psf);

fwhm_gt_psf_lateral = 2.355*f_gt_psf_lateral.c1;
fwhm_lr_interp_psf_lateral  = 2.355*f_lr_interp_psf_lateral .c1;
fwhm_sr_psf_lateral = 2.355*f_sr_psf_lateral.c1;

%% axial
% extract axial data
row_3d = 134;
col_3d = 246;
gt_axial_slice = gt(row_3d,col_3d,:);
gt_axial_slice = squeeze(gt_axial_slice);
lr_interp_axial_slice = lr_interp(row_3d,col_3d,:);
lr_interp_axial_slice = squeeze(lr_interp_axial_slice);
sr_axial_slice = sr(row_3d,col_3d,:);
sr_axial_slice = squeeze(sr_axial_slice);

% determine the horizontal axis
gt_z_size = numel(gt_axial_slice);
sr_z_size = numel(sr_axial_slice);

% interpolate sr data for psf comparison
sr_x_coords = linspace(1, gt_z_size, sr_z_size);
sr_axial_slice_resized = interp1(sr_x_coords,squeeze(double(sr_axial_slice)), 1:gt_z_size);
sr_axial_slice_resized = sr_axial_slice_resized';
% % todo check resampling methods
% if sr_z_size > gt_z_size
%     sr_axial_slice_resized = downsample(sr_axial_slice, sr_z_size/gt_z_size);
% elseif sr_z_size == gt_z_size
%     sr_axial_slice_resized = sr_axial_slice;
% else
%     sr_axial_slice_resized = upsample(sr_axial_slice, gt_z_size/sr_z_size);
% end

%% fitting axial psf
f_gt_psf_axial = fit_psf(gt_axial_slice);
f_lr_interp_psf_axial =fit_psf(lr_interp_axial_slice);
f_sr_psf_axial = fit_psf(sr_axial_slice_resized);

fwhm_gt_psf_axial = 2.355*f_gt_psf_axial.c1;
fwhm_lr_interp_psf_axial  = 2.355*f_lr_interp_psf_axial .c1;
fwhm_sr_psf_axial = 2.355*f_sr_psf_axial.c1;

%%
% show lateral PSF
figure,
% subplot(1,2,1)
% plot(gt_psf, 'ro', 'LineWidth', 2);
% hold on; grid on;
% plot(lr_interp_psf, 'b+', 'LineWidth', 2);
% plot(sr_psf, 'k*', 'LineWidth', 2);
% xlabel('Pixels')
% ylabel('Amplitude (a.u.)')
% set(gca,'FontSize',20,'FontWeight', 'bold')
% 
% plot(f_gt_psf_lateral, 'r');
% hold on; grid on;
% plot(f_lr_interp_psf_lateral, 'b');
% plot(f_sr_psf_lateral, 'k');
% legend('Original','Interpolation','Super-resolution', ...
%     ['Original (Fitted) \sigma = ' num2str(fwhm_gt_psf_lateral, 3)], ...
%     ['Interpolation (Fitted) \sigma = ' num2str(fwhm_lr_interp_psf_lateral, 3)],...
%     ['Super-resolution (Fitted) \sigma = ' num2str(fwhm_sr_psf_lateral,  3)])
% xlabel('Pixels')
% ylabel('Amplitude (a.u.)')
% title('Laterial PSF')
% xlim([0 20])
% ylim([0 300])
% set(gca,'FontSize',20,'FontWeight', 'bold')
% hline = findobj(gcf, 'type', 'line');
% set(hline,'LineWidth',2)
% 
% % show axial psf
% subplot(1,2,2)
plot(gt_axial_slice, 'ro', 'LineWidth', 2);
hold on; grid on;
plot(lr_interp_axial_slice, 'b+', 'LineWidth', 2);
plot(sr_axial_slice_resized, 'k*', 'LineWidth', 2);
xlabel('Pixels')
ylabel('Amplitude (a.u.)')
set(gca,'FontSize',20,'FontWeight', 'bold')

plot(f_gt_psf_axial, 'r');
hold on; grid on;
plot(f_lr_interp_psf_axial, 'b');
plot(f_sr_psf_axial, 'k');
legend('Original','Interpolation','Super-resolution', ...
    ['Original (Fitted) \sigma = ' num2str(fwhm_gt_psf_axial, 3)], ...
    ['Interpolation (Fitted) \sigma = ' num2str(fwhm_lr_interp_psf_axial, 3)],...
    ['Super-resolution (Fitted) \sigma = ' num2str(fwhm_sr_psf_axial, 3)])
xlabel('Pixels')
ylabel('Amplitude (a.u.)')
title('Axial PSF')
% xlim([0 20])
ylim([0 400])
set(gca,'FontSize',20,'FontWeight', 'bold')
hline = findobj(gcf, 'type', 'line');
set(hline,'LineWidth',2)
title(['Input z: ' num2str(input_depth)])