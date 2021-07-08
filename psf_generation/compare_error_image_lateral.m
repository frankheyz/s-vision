%% load data
clc; clear;
file_name = 'thy1_zone7_new';
input = double(read_tiff(['D:\data\20210623\tif_data\test\test_lr\' file_name '_73.tif']));
original = double(read_tiff(['D:\data\20210623\tif_data\' file_name '.tif']));
my_sr = double(read_tiff('D:\codes\z-vision\results\20210623\thy1_zone7_new_direct_downsampleX2.00X2.00X2.00X2.00X2.00X2.00.tif'));
dfcan_sr = double(imread('D:\data\20210623\dfcan_test\thy1_zone7_new_73.tif'));
pssr_sr = double(imread('D:\data\20210623\pssr_test\thy1_zone7_new_73_s_1_s_1_20210617_gfp_thy1_new_crap_AG_SP_e50_e50_256.tif'));
dspnet_sr = double(read_tiff(['D:\share_dir\data\dsp-net\validate\tubulin3d-simu_2stage-dbpn+rdn_factor-4_norm-fixed_loss-mse\' ...
    'SR-thy1_zone7_new_center_cropped_256_256_38_block_38_64_64.tif';
    ]));

% resize input
input = imresize(input, 4, 'nearest');

% 16bits to 8
dfcan_sr = double(uint8(dfcan_sr / 256));

% get a slize from z-v
original = original(:,:,73);
my_sr_xy = my_sr(:,:,73);
dspnet_sr_xy = dspnet_sr(:,:,73);

% to float
input = input ./ 255;
original = original ./ 255;
my_sr_xy = my_sr_xy ./ 255;
dfcan_sr = dfcan_sr ./ 255;
pssr_sr = pssr_sr ./ 255;
dspnet_sr_xy = dspnet_sr_xy ./ 255;

% normalize original and my_sr_xy for fair comparison
my_sr_xy = my_sr_xy./max(my_sr_xy(:));
original = original./max(original(:));
dspnet_sr_xy = dspnet_sr_xy./max(dspnet_sr_xy(:));

% error
my_sr_error_img = original - my_sr_xy;
dfcan_sr_error_img = original - dfcan_sr;
pssr_sr_error_img = original - pssr_sr;
dspnet_sr_error_img = original - dspnet_sr_xy;
original_error_img = original - original ;

write_tiff(original, 'D:\data\20210623\original_test\thy1_zone7_new_73_norm_original.tif')

%% show mag pics
input_mag = imread(['D:\data\20210623\with_mag_inset\thy1_zone7_new_73_input.tif']);
original_mag = imread(['D:\data\20210623\with_mag_inset\thy1_zone7_new_73_norm_original.tif']);
my_sr_mag = imread(['D:\data\20210623\with_mag_inset\thy1_zone7_new_73_norm_ours.tif']);
dfcan_sr_mag = imread(['D:\data\20210623\with_mag_inset\thy1_zone7_new_73_dfcan.tif']);
pssr_sr_mag = imread(['D:\data\20210623\with_mag_inset\thy1_zone7_new_73_s_1_s_1_20210617_gfp_thy1_new_crap_AG_SP_e50_e50_256.tif']);


position =  [20 100];

input_mag_text = insertText(input_mag,position, 'Input', 'AnchorPoint','LeftBottom', 'Fontsize', 60, ...
    'BoxColor', 'black', 'TextColor', 'white', 'BoxOpacity', 0.5);
my_sr_mag_text = insertText(my_sr_mag,position, 'Ours', 'AnchorPoint','LeftBottom', 'Fontsize', 60, ...
    'BoxColor', 'black', 'TextColor', 'white', 'BoxOpacity', 0.5);
dfcan_sr_mag_text = insertText(dfcan_sr_mag,position, 'DFCAN', 'AnchorPoint','LeftBottom', 'Fontsize', 60, ...
    'BoxColor', 'black', 'TextColor', 'white', 'BoxOpacity', 0.5);
pssr_sr_mag_text = insertText(pssr_sr_mag,position, 'PSSR', 'AnchorPoint','LeftBottom', 'Fontsize', 60, ...
    'BoxColor', 'black', 'TextColor', 'white', 'BoxOpacity', 0.5);

dspnet_sr_mag_text = insertText(dspnet_sr_xy,position, 'DSPNet', 'AnchorPoint','LeftBottom', 'Fontsize', 60, ...
    'BoxColor', 'black', 'TextColor', 'white', 'BoxOpacity', 0.5);

original_sr_mag_text = insertText(original_mag,position, 'Original', 'AnchorPoint','LeftBottom', 'Fontsize', 60, ...
    'BoxColor', 'black', 'TextColor', 'white', 'BoxOpacity', 0.5);


%% show image with magnified inset
figure,
subaxis(2,5,1,'Spacing',0.01);
imshow(input_mag_text, [0 1])
hold on;
quiver([60 60], [970 970], [8 0], [0 -8], 10, 'LineWidth', 2.5, 'color', 'white', 'MaxHeadSize', 5)
text(150,965, 'X', 'color', 'white', 'FontSize', 12, 'FontWeight', 'bold')
text(45,852, 'Y', 'color', 'white', 'FontSize', 12, 'FontWeight', 'bold')

ax(7) = subaxis(2,5,2);
imshow(my_sr_mag_text, [0 1]);

subaxis(2,5,3);
imshow(dfcan_sr_mag_text, [0 1])

subaxis(2,5,4);
imshow(pssr_sr_mag_text, [0 1])

subaxis(2,5,5);
imshow(dspnet_sr_mag_text, [0 1])

ax(1) = subaxis(2,5,6);
imshow(original_sr_mag_text, [0 1]);


ax(2) = subaxis(2,5,7);
imshow(abs(my_sr_error_img), [0 1]);
text(40, 40, '|Ours - Original|', 'color', 'white', 'FontSize', 14, 'FontWeight', 'bold')

ax(3) = subaxis(2,5,8);
imshow(abs(dfcan_sr_error_img), [0 1]);
text(40, 40, '|DFCAN - Original|', 'color', 'white', 'FontSize', 14, 'FontWeight', 'bold')

ax(4) = subaxis(2,5,9);
imshow(abs(pssr_sr_error_img), [0 1]);
text(40, 40, '|PSSR - Original|', 'color', 'white', 'FontSize', 14, 'FontWeight', 'bold')

ax(5) = subaxis(2,5,10);
imshow(abs(dspnet_sr_error_img), [0 1]);
text(40, 40, '|DSPNET - Original|', 'color', 'white', 'FontSize', 14, 'FontWeight', 'bold')

colormap(ax(2), 'jet')
colormap(ax(3), 'jet')
colormap(ax(4), 'jet')
colormap(ax(5), 'jet')
colormap(ax(7), 'jet')
%% show original image
figure,
subaxis(2,4,1,'Spacing',0.01);
imshow(input, [0 1])
% imshow(input./max(input(:)), [0 1])

subaxis(2,4,2);
write_tiff(my_sr_xy, 'D:\data\20210623\ours_test\thy1_zone7_new_73_norm_ours.tif')
imshow(my_sr_xy , [0 1])

subaxis(2,4,3);
imshow(dfcan_sr, [0 1])

subaxis(2,4,4);
imshow(pssr_sr, [0 1])

ax(1) = subaxis(2,4,5);
imshow(original, [0 1]);

ax(2) = subaxis(2,4,6);
imshow(abs(my_sr_error_img), [0 1]);

ax(3) = subaxis(2,4,7);
dfcan_sr_norm_error_img = abs(original-dfcan_sr);
imshow(abs(dfcan_sr_error_img), [0 1]);

ax(4) = subaxis(2,4,8);
pssr_sr_norm_error_img = abs(original-pssr_sr);
imshow(abs(pssr_sr_error_img), [0 1]);


colormap(ax(2), 'jet')
colormap(ax(3), 'jet')
colormap(ax(4), 'jet')