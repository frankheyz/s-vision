function line_data = line_profile(img, start_point, end_point)

[x, y]=bresenham(start_point(1),start_point(2),end_point(1),end_point(2));

x_size = numel(x);
line_data = zeros(x_size, 1);

for i=1:x_size
    val = img(x(i), y(i));
    line_data(i, 1) = val;
end