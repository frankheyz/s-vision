function fitted_function = fit_psf(psf_curve)

x = 1:numel(psf_curve);
x = x(:);
fitted_function = fit(x, double(psf_curve),'gauss1');

end
