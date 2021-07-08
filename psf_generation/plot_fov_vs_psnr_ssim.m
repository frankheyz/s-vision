%%
pixel_no_list = [32768 32768 32768 32768 119808 119808 119808 119808 294912 294912 ...
    294912 294912 588800 588800 588800 588800 1032192 1032192 1032192 1032192 ...
    1655808 1655808 1655808 1655808 2490368 2490368 2490368 2490368];

psnr_list = [19.8882 23.8628 27.8943 27.1071 27.7553 34.606 31.275 37.0804 36.3168 36.7641 ...
    38.4056 33.4529 35.6727 34.6871 33.7165 33.605 36.0253 37.2251 35.042 32.0728 ...
    36.2465 33.2901 33.3975 33.1244 33.4761 37.0486 35.5625 36.3285];

ssim_list = [0.519081 0.696451 0.725626 0.722232 0.828091 0.933966 0.887052 0.959491 ...
    0.950868 0.954431 0.971197 0.936801 0.951989 0.952232 0.933584 0.936066 0.949672 ...
    0.965374 0.942875 0.932008 0.95956 0.933262 0.939084 0.928115 0.941733 0.964554 ... 
    0.948329 0.95233 ];



%%
pixel_no_list_reshaped = reshape(pixel_no_list, [4,7]);
ssim_list_reshaped = reshape(ssim_list, [4,7]);
psnr_list_reshaped = reshape(psnr_list, [4,7]);

pixel_no_mean = mean(pixel_no_list_reshaped,1);
ssim_mean = mean(ssim_list_reshaped,1);
psnr_mean = mean(psnr_list_reshaped,1);

f1 = figure;
yyaxis left
p1=semilogx(pixel_no_list,ssim_list, 'o', 'linewidth', 2);
hold on;
p2=semilogx(pixel_no_mean,ssim_mean, 'linewidth', 2);
ylim([0.4 1])
ylabel('SSIM')

yyaxis right
p3=semilogx(pixel_no_list,psnr_list, 'x', 'linewidth', 2);
hold on;
p4=semilogx(pixel_no_mean,psnr_mean, 'linewidth', 2);
ylim([18 45])
ylabel('PSNR (dB)')

xlabel('Voxel number')
grid on;
set(gca,'FontSize',16,'FontWeight', 'bold')

title('Performance v.s. FOV')
legend([p1,p3], 'SSIM', 'PSNR')
darkBackground(f1)
set(gcf, 'inverthardcopy', 'off') 