function tiff_stack = read_tiff(tiff_img)

% return tiff structure, one element per image
tiff_info = imfinfo(tiff_img); 
% read in first image
tiff_stack = imread(tiff_img, 1) ; 

%concatenate each successive tiff to tiff_stack
for ii = 2 : size(tiff_info, 1)
    temp_tiff = imread(tiff_img, ii);
    tiff_stack = cat(3 , tiff_stack, temp_tiff);
end
