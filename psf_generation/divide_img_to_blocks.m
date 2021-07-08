function cell_blocks=divide_img_to_blocks(input_img, block_size_r, block_size_c)
[size_x, size_y] = size(input_img);

whole_block_rows = floor(size_x / block_size_r);
block_vec_r = [block_size_r * ones(1, whole_block_rows), rem(size_x, block_size_r)];

whole_block_cols = floor(size_y / block_size_c);
block_vec_c = [block_size_c * ones(1, whole_block_cols), rem(size_y, block_size_c)];

cell_blocks = mat2cell(input_img(:,:,1), block_vec_r, block_vec_c);