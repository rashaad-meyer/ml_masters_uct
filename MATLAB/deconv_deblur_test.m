img = imread('C:/Users/Rashaad/Documents/Postgrad/data/deconv_tests/deblur-single-image/kid.png');

% normalize the image
img = cast(img, 'double')/255.0;

% Current factor weights
hrf0 = zeros(2,4);
hrf0(1) = 1;  hrf0(2) = 0.2;
%hrf0 = 0.1*randn(size(hrf0));  hrf0(1) = 1;

% Apply four-factor blur
hr0 = conv2(conv2(conv2(hrf0,fliplr(hrf0)),flipud(hrf0)),rot90(hrf0,2));
img_blur = conv2(img,hr0,'same');

plot = false;

if plot
    subplot(1,2,1);  imagesc(img);  axis equal;  axis tight;  colormap(gray);  title('Unblurred');
    subplot(1,2,2);  imagesc(img_blur);  axis equal;  axis tight;  colormap(gray);  title('Blurred');
end

y = img;
X = img_blur;

inputSize = [96 96];

layers = [
    imageInputLayer(inputSize)
    deconv_clayer2([2, 4], [96 96], [96 96])
    regressionLayer
    ];

options = trainingOptions("adam", ...
    MaxEpochs=100, ...
    InitialLearnRate=1e-3, ...
    Verbose=true, ...
    Plots='training-progress');

net = trainNetwork(X, y, layers, options);

YPred = predict(net, X);

% print and store deconv kernel
deconv_kernel = net.Layers(2).hrfp

plot = true;
if plot
    subplot(1,3,1);  imagesc(img);  axis equal;  axis tight;  colormap(gray);  title('Unblurred');
    subplot(1,3,3);  imagesc(img_blur);  axis equal;  axis tight;  colormap(gray);  title('Blurred');
    subplot(1,3,2);  imagesc(YPred);  axis equal;  axis tight;  colormap(gray);  title('Predicted');
end
