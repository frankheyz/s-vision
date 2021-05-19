function write_gif(input_img_stack, delay, out_name)

% normalization
input_img_stack = input_img_stack ./ max(input_img_stack(:));
% convert img to 8 bits
input_img_stack = input_img_stack .* 255;

for i = 1:20
    % draw stuff
    img =  input_img_stack(:,:,i);
    if i == 1
        imwrite(img,[out_name '.gif'],'gif','LoopCount',Inf,'DelayTime', delay);
    else
        imwrite(img,[out_name '.gif'],'gif','WriteMode','append','DelayTime',delay);
    end
end