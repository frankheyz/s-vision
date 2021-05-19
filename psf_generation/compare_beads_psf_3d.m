%% compare 3d images
clear
lr = read_tiff('low_density_z8_lr_3d.tif');
gt = read_tiff('low_density_z32_3d.tif');
sr = read_tiff('low_density_z8_lr_3dX2.00X2.00X2.00X2.00X2.00X2.00.tif');

lr_interp = imresize3(lr, 4);

% get a slice 
gt_slice = gt(:,:,16);
lr_interp_slice = lr_interp(:,:,16);
sr_slice = sr(:,:,16);

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

sub_x = 222-70:222+70;
sub_y = 158-70:158+70;

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
row = 222;
col = 158;
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

%% show axial
figure,
subplot(3,1,1)
imshow(squeeze(gt(:,230,:))')
title('Original (512 \times 20)', 'FontSize', 14,'FontWeight','bold')
subplot(3,1,2)
imshow(squeeze(lr_interp(:,230,:))')
title('Interpolation x 4 (512 \times 20)', 'FontSize', 14,'FontWeight','bold')
subplot(3,1,3)
imshow(squeeze(sr(:,230,:))')
title('Super-resolution x 4 (512 \times 20)', 'FontSize', 14,'FontWeight','bold')

%% extract axial psf (3d)
row_3d=230;
col_3d=230;
gt_axial_slice = gt(row_3d,col_3d,:);
gt_axial_slice = squeeze(gt_axial_slice);
lr_interp_axial_slice = lr_interp(row_3d,col_3d,:);
lr_interp_axial_slice = squeeze(lr_interp_axial_slice);
sr_axial_slice = sr(row_3d,col_3d,:);
sr_axial_slice = squeeze(sr_axial_slice);

%% fitting axial psf
f_gt_psf_axial = fit_psf(gt_axial_slice);
f_lr_interp_psf_axial =fit_psf(lr_interp_axial_slice);
f_sr_psf_axial = fit_psf(sr_axial_slice);

fwhm_gt_psf_axial = 2.355*f_gt_psf_axial.c1;
fwhm_lr_interp_psf_axial  = 2.355*f_lr_interp_psf_axial .c1;
fwhm_sr_psf_axial = 2.355*f_sr_psf_axial.c1;

%%
% show lateral PSF
figure,
subplot(1,2,1)
plot(gt_psf, 'ro', 'LineWidth', 2);
hold on; grid on;
plot(lr_interp_psf, 'b+', 'LineWidth', 2);
plot(sr_psf, 'k*', 'LineWidth', 2);
xlabel('Pixels')
ylabel('Amplitude (a.u.)')
set(gca,'FontSize',20,'FontWeight', 'bold')

plot(f_gt_psf_lateral, 'r');
hold on; grid on;
plot(f_lr_interp_psf_lateral, 'b');
plot(f_sr_psf_lateral, 'k');
legend('Original','Interpolation','Super-resolution', ...
    ['Original (Fitted) \sigma = ' num2str(fwhm_gt_psf_lateral, 3)], ...
    ['Interpolation (Fitted) \sigma = ' num2str(fwhm_lr_interp_psf_lateral, 3)],...
    ['Super-resolution (Fitted) \sigma = ' num2str(fwhm_sr_psf_lateral,  3)])
xlabel('Pixels')
ylabel('Amplitude (a.u.)')
title('Laterial PSF')
xlim([0 20])
ylim([0 300])
set(gca,'FontSize',20,'FontWeight', 'bold')
hline = findobj(gcf, 'type', 'line');
set(hline,'LineWidth',2)

% show axial psf
subplot(1,2,2)
plot(gt_axial_slice, 'ro', 'LineWidth', 2);
hold on; grid on;
plot(lr_interp_axial_slice, 'b+', 'LineWidth', 2);
plot(sr_axial_slice, 'k*', 'LineWidth', 2);
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
xlim([0 20])
ylim([0 350])
set(gca,'FontSize',20,'FontWeight', 'bold')
hline = findobj(gcf, 'type', 'line');
set(hline,'LineWidth',2)
