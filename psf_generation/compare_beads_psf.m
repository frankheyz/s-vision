%% compare 2d images
clear
lr = imread('low_density_lr.png');
gt = imread('low_density.png');
% sr = imread('low_density_lrX2.00X2.00X2.00X2.00_dilated.png');
sr = imread('low_density_lrX2.00X2.00X2.00X2.00.png');

lr_interp = imresize(lr, 4);

figure,
subplot(2,3,1)
imshow(gt)
title('Original (512 \times 512)', 'FontSize', 14,'FontWeight','bold')

subplot(2,3,2)
imshow(lr_interp)
title('Interpolation x 4 (512 \times 512)', 'FontSize', 14,'FontWeight','bold')

subplot(2,3,3)
imshow(sr)
title('Super-resolution x 4 output (512 \times 512)', 'FontSize', 14,'FontWeight','bold')


sub_x = 210:350;
sub_y = 210:350;

subplot(2,3,4)
imshow(gt(sub_x, sub_y))
title('Original (512 \times 512)', 'FontSize', 14,'FontWeight','bold')

subplot(2,3,5)
imshow(lr_interp(sub_x, sub_y))
title('Interpolation x 4 (512 \times 512)', 'FontSize', 14,'FontWeight','bold')

subplot(2,3,6)
imshow(sr(sub_x, sub_y))
title('Super-resolution x 4 output (512 \times 512)', 'FontSize', 14,'FontWeight','bold')

%%
% sub_x_small= round(210/4):round(350/4);
% sub_y_small = round(210/4):round(350/4);
% imshow(lr(sub_x_small, sub_y_small))
% title('Input (128 \times 128)', 'FontSize', 14,'FontWeight','bold')
%% psf
rows = [230 358 202 329 245];
cols = [230 390 502 123 462];
% rows = [230];
% cols = [230];
delta_x = 7;
gt_psf = zeros(delta_x * 2 + 2, numel(rows));
lr_interp_psf = zeros(delta_x * 2 + 2, numel(rows));
sr_psf = zeros(delta_x * 2 + 2, numel(rows));
for i = 1:numel(rows)
    row = rows(i);
    col = cols(i);
    gt_psf(:, i) = gt(row-delta_x : row+delta_x+1, col);
    lr_interp_psf(:, i) = lr_interp(row-delta_x : row+delta_x+1, col);
    sr_psf(:, i) = sr(row-delta_x : row+delta_x+1, col);
end
gt_psf = mean(gt_psf, 2);
lr_interp_psf = mean(lr_interp_psf, 2);
sr_psf = mean(sr_psf, 2);
%% fitting
x = 1:16;
x = x';
f_gt_psf = fit(x, gt_psf,'gauss1');
f_lr_interp_psf = fit(x, lr_interp_psf,'gauss1');
f_sr_psf = fit(x, sr_psf,'gauss1');

fwhm_gt_psf = 2.355*f_gt_psf.c1;
fwhm_lr_interp_psf  = 2.355*f_lr_interp_psf .c1;
fwhm_sr_psf = 2.355*f_sr_psf.c1;
%%
figure,
plot(gt_psf, 'ro', 'LineWidth', 2);
hold on; grid on;
plot(lr_interp_psf, 'b+', 'LineWidth', 2);
plot(sr_psf, 'k*', 'LineWidth', 2);
xlabel('Pixels')
ylabel('Amplitude (a.u.)')
set(gca,'FontSize',20,'FontWeight', 'bold')

plot(f_gt_psf, 'r');
hold on; grid on;
plot(f_lr_interp_psf, 'b');
plot(f_sr_psf, 'k');
legend('Original','Interpolation','Super-resolution', ...
    ['Original (Fitted) \sigma = ' num2str(fwhm_gt_psf)], ...
    ['Interpolation (Fitted) \sigma = ' num2str(fwhm_lr_interp_psf)],...
    ['Super-resolution (Fitted) \sigma = ' num2str(fwhm_sr_psf)])
xlabel('Pixels')
ylabel('Amplitude (a.u.)')
title('Laterial PSF')
xlim([0 20])
set(gca,'FontSize',20,'FontWeight', 'bold')
hline = findobj(gcf, 'type', 'line');
set(hline,'LineWidth',2)

fwhm_gt_psf = 2.355*f_gt_psf.c1;
fwhm_lr_interp_psf  = 2.355*f_lr_interp_psf .c1;
fwhm_sr_psf = 2.355*f_sr_psf.c1;

%% mse
sr_mse = immse(sr(:,:,1), gt);
sr_ssim = ssim(sr(:,:,1), gt);
fprintf('\n MSE of SR %0.4f\n', sr_mse);
fprintf('\n SSIM of SR %0.4f\n', sr_ssim);

fprintf('---------------------')

lr_mse = immse(lr_interp, gt);
lr_ssim = ssim(lr_interp(:,:,1), gt);
fprintf('\n MSE of Interp %0.4f\n', lr_mse);
fprintf('\n SSIM of Interp %0.4f\n', lr_ssim);

%%
% %% compare images
% lr = read_tiff('lr_default.tif');
% gt = read_tiff('hr_beads.tif');
% sr =  read_tiff('lr_defaultX2.00X2.00X2.00X2.00X2.00X2.00.tif');
% 
% lr_interp = imresize3(lr, 4);
% %%
% figure, 
% gt_psf = gt(467:487,497,8);
% hr_psf = sr(467:487,497,8);
% lr_psf = lr_interp(467:487,497,8);
% plot(gt_psf);
% hold on;
% plot(hr_psf);
% plot(lr_psf);
% legend('GT', 'SR', 'LR interp')
% %%
% figure, imshow3D(gt);
% figure, imshow3D(sr);
% figure, imshow3D(lr_interp);
