%% load
block_1_1 = read_tiff('D:\data\yj\20191008\gfpvivo\fov_test\20210706_10_22_47\output_image\block_1_1X2.00X2.00X2.00X2.00X2.00X2.00.tif');
block_1_2 = read_tiff('D:\data\yj\20191008\gfpvivo\fov_test\20210706_10_21_54\output_image\block_1_2X2.00X2.00X2.00X2.00X2.00X2.00.tif');
block_1_3 = read_tiff('D:\data\yj\20191008\gfpvivo\fov_test\20210706_10_22_20\output_image\block_1_3X2.00X2.00X2.00X2.00X2.00X2.00.tif');

block_2_1 = read_tiff('D:\data\yj\20191008\gfpvivo\fov_test\20210706_10_23_13\output_image\block_2_1X2.00X2.00X2.00X2.00X2.00X2.00.tif');
block_2_2 = read_tiff('D:\data\yj\20191008\gfpvivo\fov_test\20210706_10_21_28\output_image\block_2_2X2.00X2.00X2.00X2.00X2.00X2.00.tif');
block_2_3 = read_tiff('D:\data\yj\20191008\gfpvivo\fov_test\20210706_10_23_40\output_image\block_2_3X2.00X2.00X2.00X2.00X2.00X2.00.tif');

block_3_1 = read_tiff('D:\data\yj\20191008\gfpvivo\fov_test\block_3_1X2.00X2.00X2.00X2.00X2.00X2.00.tif');
block_3_2 = read_tiff('D:\data\yj\20191008\gfpvivo\fov_test\20210706_10_20_35\output_image\block_3_2X2.00X2.00X2.00X2.00X2.00X2.00.tif');
block_3_3 = read_tiff('D:\data\yj\20191008\gfpvivo\fov_test\20210706_10_21_03\output_image\block_3_3X2.00X2.00X2.00X2.00X2.00X2.00.tif');
%%

full_fov = zeros(2400, 2400, 100);
full_fov(1:800,1:800,:) = block_1_1;
full_fov(1:800,801:1600,:) = block_1_2;
full_fov(1:800,1601:end,:) = block_1_3;
full_fov(801:1600,1:800,:) = block_2_1;
full_fov(801:1600,801:1600,:) = block_2_2;
full_fov(801:1600,1601:end,:) = block_2_3;
full_fov(1601:end,1:800,:) = block_3_1;
full_fov(1601:end,801:1600,:) = block_3_2;
full_fov(1601:end,1601:end,:) = block_3_3;
full_fov_norm = full_fov / 255;
write_tiff(full_fov_norm, 'D:\data\yj\20191008\gfpvivo\full_view_stack_sr.tif')
clear block_1_1 block_1_2 block_1_3 block_2_1 block_2_2 block_2_3 block_3_1 block_3_2 block_3_1
pack