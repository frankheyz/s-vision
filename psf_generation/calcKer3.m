%inputs: B,G - gray level blurred and sharp images respectively (double)
%        szKer - 2 element vector specifying the size of the required kernel
%outputs: mKer - the recovered kernel, 
%         imBsynth - the sharp image convolved with the recovered kernel
%
%example usage:  mKer = calcKer(B, G, [11 11]);

function [mKer, imBsynth] = calcKer3(B, G, szKer)

  %get the "valid" pixels from B (i.e. those that do not depend 
  %on zero-padding or a circular assumption
  dim = numel(szKer);
  
  if dim == 3
      target_size = [100 100 32];
      B = B(51:51+target_size(1), 51:51+target_size(2), 1:target_size(3));
      G = G(51:51+target_size(1), 51:51+target_size(2), 1:target_size(3));
  end
  
  if dim == 2
      imBvalid = B(ceil(szKer(1)/2):end-floor(szKer(1)/2), ...
      ceil(szKer(2)/2):end-floor(szKer(2)/2));
  elseif dim == 3
      imBvalid = B(ceil(szKer(1)/2):end-floor(szKer(1)/2), ...
        ceil(szKer(2)/2):end-floor(szKer(2)/2), ceil(szKer(3)/2):end-floor(szKer(3)/2));
  end

  %get a matrix where each row corresponds to a block from G 
  %the size of the kernel
  if dim == 3
    mGconv = im2col3(G, szKer, 'sliding')';
  elseif dim == 2
    mGconv = im2col(G, szKer, 'sliding')';
  end
    
  %solve the over-constrained system using MATLAB's backslash
  %to get a vector version of the cross-correlation kernel
  vXcorrKer = mGconv \ imBvalid(:);

  %reshape and rotate 180 degrees to get the convolution kernel
  mKer = rot90(reshape(vXcorrKer, szKer), 2);

  if (nargout > 1)
      %if there is indeed a convolution relationship between B and G
      %the following will result in an image similar to B
      if dim == 2
          imBsynth = conv2(G, mKer, 'valid');
      elseif dim == 3
          imBsynth = conv3(G, mKer, 'valid');
      end
      
  end

end