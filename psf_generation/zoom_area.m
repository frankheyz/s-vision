function zoomed=zoom_area(input_img, center, x_radius, y_radius)

zoomed = input_img(center(1)-x_radius:center(1)+x_radius, center(2)-y_radius:center(2)+y_radius);