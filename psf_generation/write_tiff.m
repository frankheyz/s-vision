function write_tiff(tiff_stack, save_name)

% delete file if exists
if isfile(save_name)
    delete(save_name)
end
    
for ii = 1 : size(tiff_stack, 3)
    imwrite(tiff_stack(:,:,ii) , save_name , 'WriteMode' , 'append') ;
end