%% generate train and validate samples
clc; clear
file_name = 'zone7';
stacks = read_tiff(['D:\data\20210617\tif_data\raw\' file_name '.tif']);
train_lr_dir = 'D:\data\20210617\tif_data\train\train_lr\';
train_hr_dir = 'D:\data\20210617\tif_data\train\train_hr\';

validate_lr_dir = 'D:\data\20210617\tif_data\validate\validate_lr\';
validate_hr_dir = 'D:\data\20210617\tif_data\validate\validate_hr\';

[x,y,z] = size(stacks);
validate_every_n_frame = 10;

resize_factor = 0.25;
for i = 1:z
    img = stacks(:,:,i);
%     img = img(128:383, 128:383);
    img_lr = imresize(img, resize_factor);
    
    if mod(i, validate_every_n_frame) == 0
         write_tiff(img, [validate_hr_dir file_name '_' num2str(i) '.tif'])
         write_tiff(img_lr, [validate_lr_dir file_name '_' num2str(i) '.tif'])
    else
        write_tiff(img, [train_hr_dir file_name '_' num2str(i) '.tif'])
        write_tiff(img_lr, [train_lr_dir file_name '_' num2str(i) '.tif'])
    end
    
end

%% test set
% todo check lr generation method use imresize only?
test_file_name = 'thy1_zone7_new';
test_stacks = read_tiff(['D:\data\20210623\tif_data\' test_file_name '.tif']);
test_lr_dir = 'D:\data\20210623\tif_data\test\test_lr\';
test_lr_3d_dir = 'D:\data\20210623\tif_data\test\test_lr_3d\';
test_hr_dir = 'D:\data\20210623\tif_data\test\test_hr\';

[x,y,z] = size(test_stacks);
for i = 1:z
    img = test_stacks(:,:,i);
    write_tiff(img, [test_hr_dir test_file_name '_' num2str(i) '.tif'])
    
    img_lr = imresize(img, resize_factor);
    write_tiff(img_lr, [test_lr_dir test_file_name '_' num2str(i) '.tif'])
end

test_stack_lr = imresize3(test_stacks, resize_factor);
write_tiff(test_stack_lr , [test_lr_3d_dir test_file_name '_' 'direct_downsample' '.tif'])