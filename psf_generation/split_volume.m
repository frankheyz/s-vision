%% split a large volume to smaller volume for SR
clc; clear
stacks = read_tiff('D:\data\gyf\CUBIC 1\400 um\5x\file_aoon_00003_uint8.tif');
split_size = [256, 256, 25];

tiles = mat2tiles(stacks, split_size);

for i = 1: numel(tiles)
    file_path = 'D:\data\gyf\CUBIC 1\400 um\5x\';
    file_name = ['file_aoon_00003_uint8' '_' num2str(i) '.tif'];
    current_tile = tiles(i);
    write_tiff(current_tile{1,1}, [file_path file_name]);
end




