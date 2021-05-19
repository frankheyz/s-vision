%% show kernel
crisp = hr_stack_psf_h;
blurred =hr_stack_psf_l;

crisp = (crisp) ./ 255;
blurred = (blurred) ./ 255;

crispfft = fftn(crisp);
blurredfft = fftn(blurred);

PSFfft = blurredfft ./ crispfft;
PSF = ifftshift(ifftn(PSFfft));

figure; imshow3D(PSF,[]);
img = deconvwnr(blurred,PSF);
figure; imshow3D(img,[]);title('recovered')
figure; imshow3D(blurred,[]);title('blurred')