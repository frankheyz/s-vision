%% read all 2d tif in a dir and combine to a 3d tiff stack
function image_stack=imgs2stack(dir_path)
files = dir(dir_path);
image = read_tiff([dir_path '\' files(3).name]);
[size_x, size_y] = size(image);
size_z = numel(files) - 2;

files_sorted = cell(1, size_z);

for i = 3:size_z+2
    files_sorted{i-2} = files(i).name;
end

files_sorted = natsortfiles(files_sorted);

image_stack = zeros(size_x, size_y, size_z);

for i = 1:size_z
    image = read_tiff([dir_path '\' files_sorted{i}]);
    image_stack(:,:,i) = image;
end