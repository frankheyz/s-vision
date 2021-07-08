function cropped = center_crop_3d(input_stack, lateral_size, axial_size)

[x,y,z] = size(input_stack);

x_mid = round(x / 2);
y_mid = round(y / 2);
z_mid = round(z / 2);

lateral_range = x_mid - round(lateral_size/2) + 1 : x_mid - round(lateral_size/2) + lateral_size;
axial_range = z_mid - round(axial_size/2) + 1 : z_mid - round(axial_size/2) + axial_size;

cropped = input_stack(lateral_range, lateral_range, axial_range);