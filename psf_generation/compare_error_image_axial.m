%% load data
clc; clear;
file_name = 'thy1_zone7_new';
input = double(read_tiff(['D:\data\20210623\tif_data\test\test_lr_3d\' file_name '_center_cropped_256_256_38.tif']));
original = double(read_tiff(['D:\data\20210623\tif_data\' file_name '.tif']));
my_sr = double(read_tiff('D:\codes\z-vision\results\20210623\thy1_zone7_new_center_cropped_256_256_38X2.00X2.00X2.00X2.00X2.00X2.00.tif'));
dfcan_sr = double(imgs2stack('D:\data\20210623\dfcan_test\'));
pssr_sr = double(imgs2stack('D:\data\20210623\pssr_test\'));
dspnet_sr = double(read_tiff(['D:\share_dir\data\dsp-net\validate\tubulin3d-simu_2stage-dbpn+rdn_factor-4_norm-fixed_loss-mse\' ...
    'SR-thy1_zone7_new_center_cropped_256_256_38_block_38_64_64.tif';
    ]));

% to float
input = input ./ 255;
original = original ./ 255;
my_sr = my_sr ./ 255;
dfcan_sr = dfcan_sr ./ 256 ./ 256;  % the ouput of dfcan is 16 bits
pssr_sr = pssr_sr ./ 255;
dspnet_sr = dspnet_sr./ 255;

[size_x, size_y, size_z] = size(original);
%% extract axial slice from non 3d algos
dfcan_sr = dfcan_sr(:, :, 1:4:end);
pssr_sr = pssr_sr(:, :, 1:4:end);
% resize axial dimension for comparison
dfcan_sr = imresize3(dfcan_sr, [size_x, size_y, size_z]);
pssr_sr = imresize3(pssr_sr, [size_x, size_y, size_z]);
%%
y_pos = 404;
original_axial_slice = original(:,y_pos,:);
original_axial_slice = squeeze(original_axial_slice)';

input_axial_slice = input(:,round(y_pos/4),:);
input_axial_slice = squeeze(input_axial_slice)';

my_sr_axial_slice = my_sr(:,y_pos,:);
my_sr_axial_slice = squeeze(my_sr_axial_slice)';

dfcan_sr_axial_slice = dfcan_sr(:,y_pos,:);
dfcan_sr_axial_slice = squeeze(dfcan_sr_axial_slice)';

pssr_sr_axial_slice = pssr_sr(:,y_pos,:);
pssr_sr_axial_slice = squeeze(pssr_sr_axial_slice)';

my_sr_error_abs = abs(original_axial_slice - my_sr_axial_slice);
dfcan_sr_error_abs = abs(original_axial_slice - dfcan_sr_axial_slice);
pssr_sr_error_abs = abs(original_axial_slice - pssr_sr_axial_slice);
%% plot original image
figure,
hAxis(1) = subplot(2,4,1);
imshow(input_axial_slice , [0 1])
hold on;
quiver([6 6], [36 36], [6 0], [0 -3], 2.5, 'LineWidth', 2.0, 'color', 'white', 'MaxHeadSize', 0.5)
text(2,24, 'Z', 'color', 'white', 'FontSize', 12, 'FontWeight', 'bold')
text(23,35, 'X', 'color', 'white', 'FontSize', 12, 'FontWeight', 'bold')

set(gca,'DataAspectRatio',[2 1 1])
hAxis(2) = subplot(2,4,2);
imshow(my_sr_axial_slice, [0 1])
set(gca,'DataAspectRatio',[2 1 1])
hAxis(3) = subplot(2,4,3);
imshow(dfcan_sr_axial_slice, [0 1])
set(gca,'DataAspectRatio',[2 1 1])
hAxis(4) = subplot(2,4,4);
imshow(pssr_sr_axial_slice, [0 1])
set(gca,'DataAspectRatio',[2 1 1])

hAxis(5) = subplot(2,4,5);
imshow(original_axial_slice, [0 1])
set(gca,'DataAspectRatio',[2 1 1])
hAxis(6) =subplot(2,4,6);
imshow(my_sr_error_abs, [0 1])
set(gca,'DataAspectRatio',[2 1 1])
hAxis(7) =subplot(2,4,7);
imshow(dfcan_sr_error_abs, [0 1])
set(gca,'DataAspectRatio',[2 1 1])
hAxis(8) =subplot(2,4,8);
imshow(pssr_sr_error_abs, [0 1])
set(gca,'DataAspectRatio',[2 1 1])


colormap(hAxis(6), 'jet')
colormap(hAxis(7), 'jet')
colormap(hAxis(8), 'jet')
% adjust position
pos  = get(hAxis(2), 'Position');
x_shift = 0.042;
pos(1) = pos(1)-x_shift;
set(hAxis(2), 'Position', pos)

pos  = get(hAxis(3), 'Position');
pos(1) = pos(1)-x_shift*2;
set(hAxis(3), 'Position', pos)

pos  = get(hAxis(4), 'Position');
pos(1) = pos(1)-x_shift*3;
set(hAxis(4), 'Position', pos)

pos  = get(hAxis(6), 'Position');
pos(1) = pos(1)-x_shift;
set(hAxis(6), 'Position', pos)

pos  = get(hAxis(7), 'Position');
pos(1) = pos(1)-x_shift*2;
set(hAxis(7), 'Position', pos)

pos  = get(hAxis(8), 'Position');
pos(1) = pos(1)-x_shift*3;
set(hAxis(8), 'Position', pos)
% vertical
y_shift=0.371;
pos  = get(hAxis(5), 'Position');
pos(2) = pos(2)+y_shift;
set(hAxis(5), 'Position', pos)

pos  = get(hAxis(6), 'Position');
pos(2) = pos(2)+y_shift;
set(hAxis(6), 'Position', pos)

pos  = get(hAxis(7), 'Position');
pos(2) = pos(2)+y_shift;
set(hAxis(7), 'Position', pos)

pos  = get(hAxis(8), 'Position');
pos(2) = pos(2)+y_shift;
set(hAxis(8), 'Position', pos)
%% plot magnified images
input_axial_slice_interp = imresize(input_axial_slice, 4, 'nearest');
% center on hr images
x_center=38;
y_center=411;
radius=22;
input_axial_slice_interp_mag = input_axial_slice_interp(x_center-radius:x_center+radius, y_center-radius*2:y_center+radius*2);
original_axial_slice_mag = original_axial_slice(x_center-radius:x_center+radius, y_center-radius*2:y_center+radius*2);
my_sr_axial_slice_mag =my_sr_axial_slice(x_center-radius:x_center+radius, y_center-radius*2:y_center+radius*2);

figure,
subaxis(1,3,1,'Spacing',0.01);
imshow(input_axial_slice_interp_mag , [0 1]); colormap(hot)
hold on;
quiver([5 5], [43 43], [5 0], [0 -2.5], 2.5, 'LineWidth', 2.0, 'color', 'white', 'MaxHeadSize', 0.5)
text(3,3, 'Input', 'color', 'white', 'FontSize', 10, 'FontWeight', 'bold')
text(3,34, 'Z', 'color', 'white', 'FontSize', 8, 'FontWeight', 'bold')
text(21,43, 'X', 'color', 'white', 'FontSize', 8, 'FontWeight', 'bold')
set(gca,'DataAspectRatio',[2 1 1])

subaxis(1,3,2);
imshow(original_axial_slice_mag, [0 1]); colormap(hot)
text(3,3, 'Original', 'color', 'white', 'FontSize', 10, 'FontWeight', 'bold')
set(gca,'DataAspectRatio',[2 1 1])

subaxis(1,3,3);
imshow(my_sr_axial_slice_mag, [0 1]); colormap(hot)
text(3,3, 'Output', 'color', 'white', 'FontSize', 10, 'FontWeight', 'bold')
set(gca,'DataAspectRatio',[2 1 1])
%% plot a line profile from images
rows= 5:13;
cols = 93;

rows=5:20;
cols = 150;

rows=7:19;
cols = 138;
figure
input_axial_lp = input_axial_slice(rows,cols);
plot(rows, input_axial_lp, 'LineWidth', 3)
hold on;grid on;
original_axial_slice_lp = original_axial_slice(rows(1)*4:rows(end)*4, cols*4);
original_axial_slice_lp_x = rows(1)*4:rows(end)*4;
original_axial_slice_lp_x = original_axial_slice_lp_x  /4;
plot(original_axial_slice_lp_x, original_axial_slice_lp, 'LineWidth', 3)

my_sr_axial_slice_lp = my_sr_axial_slice(rows(1)*4:rows(end)*4, cols*4);
plot(original_axial_slice_lp_x, my_sr_axial_slice_lp, 'LineWidth', 3)

ylabel('Intensity (a.u.)')
xlabel('Z position (\mum)')
legend('Input', 'Original', 'Ours')
set(gca,'FontSize',20,'FontWeight', 'bold')