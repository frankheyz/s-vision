%% load multiple images for ssim/psnr comparison
clear;clc;
gt_dir = ['D:\data\20210623\tif_data\test\test_hr\' 'thy1_zone7_new'];
dfcan_dir = 'D:\codes\z-vision\results\20210623\output_DFCAN-SISR';
pssr_dir = 'D:\codes\z-vision\results\20210623\s_1_s_1_20210617_gfp_thy1_new_crap_AG_SP_e50_e50_256';
my_sr = double(read_tiff('D:\codes\z-vision\results\20210623\thy1_zone7_new_direct_downsampleX2.00X2.00X2.00X2.00X2.00X2.00.tif'));

my_psnr_array = [];
my_ssim_array = [];

dfcan_psnr_array = [];
dfcan_ssim_array = [];

pssr_psnr_array = [];
pssr_ssim_array = [];
% select images
for i = 1:4:152
    dfcan_sr = double(read_tiff([dfcan_dir '\' 'thy1_zone7_new_' num2str(i) '.tif']));
    pssr_sr = double(read_tiff([pssr_dir '\' 'thy1_zone7_new_' num2str(i) '_s_1_s_1_20210617_gfp_thy1_new_crap_AG_SP_e50_e50_256' '.tif']));
    original = double(read_tiff([gt_dir '_' num2str(i) '.tif']));
    my_sr_xy = my_sr(:,:,i);
    
    % 16bits to double for dfcan
    dfcan_sr = double(uint8(dfcan_sr / 256));
    
    % 8bit to double for pssr
    pssr_sr = double(pssr_sr);
    
    % normalization
    original = original ./ 255;
    my_sr_xy = my_sr_xy./ 255;
    dfcan_sr = dfcan_sr ./ 255;
    pssr_sr = pssr_sr ./ 255;
    
    % psnr and ssim
    my_psnr = psnr(original, my_sr_xy);
    my_ssim = ssim(original, my_sr_xy);
    my_psnr_array(end+1) = my_psnr;
    my_ssim_array(end+1) = my_ssim;
    
    dfcan_psnr = psnr(original, dfcan_sr);
    dfcan_ssim = ssim(original, dfcan_sr);
    dfcan_psnr_array(end+1) = dfcan_psnr;
    dfcan_ssim_array(end+1) = dfcan_ssim;
    
    pssr_psnr = psnr(original, pssr_sr);
    pssr_ssim = ssim(original, pssr_sr);
    pssr_psnr_array(end+1) = pssr_psnr;
    pssr_ssim_array(end+1) = pssr_ssim;
end

% plot
figure, 
subplot(1,2,1)
boxplot([my_psnr_array',dfcan_psnr_array', pssr_psnr_array'],'Notch','off','Labels',{'Ours','DFCAN', 'PSSR'}); 
grid on
ylabel('PSNR')
ylim([10 45])
title('PSNR comparison')
set(gca,'FontSize',18,'FontWeight', 'bold')

subplot(1,2,2)
boxplot([my_ssim_array',dfcan_ssim_array', pssr_ssim_array'],'Notch','off','Labels',{'Ours','DFCAN', 'PSSR'}); grid on
ylabel('SSIM')
ylim([0.25 1])
title('SSIM comparison')
set(gca,'FontSize',18,'FontWeight', 'bold')