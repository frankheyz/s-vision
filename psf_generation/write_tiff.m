function write_tiff(tiff_stack, save_name)

for ii = 1 : size(tiff_stack, 3)
    imwrite(tiff_stack(:,:,ii) , save_name , 'WriteMode' , 'append') ;
end