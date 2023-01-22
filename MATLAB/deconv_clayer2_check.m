validInputSize = [28 28];
deconv = deconv_clayer2([2 4], [28 28], [28 28]);
layout = networkDataLayout(validInputSize);
layer = initialize(deconv,layout);
checkLayer(layer, validInputSize)

% xm = randn(28, 28);
% xmf = fft2(xm);
% [Z, memory] = deconv.forward(xm);
% deconv.backward(xm, xm, xm, memory);