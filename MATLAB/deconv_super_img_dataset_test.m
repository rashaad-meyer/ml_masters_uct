[X, y] = read_super_image_res_dataset(2);

plot = true;
if plot
    subplot(1,2,1);  imagesc(squeeze(y(1, :, :)));  axis equal;
    axis tight;  colormap(gray);  title('Unblurred');

    subplot(1,2,2);  imagesc(squeeze(X(1, :, :)));  axis equal;
    axis tight;  colormap(gray);  title('Blurred');
end

inputSize = [256 256];

deconv = deconv_clayer2([2, 4], [96 96], [96 96]);
y_test = deconv.forward(X);

% layers = [
%     imageInputLayer(inputSize)
%     deconv_clayer2([2, 4], [96 96], [96 96])
%     regressionLayer
%     ];
% 
% options = trainingOptions("adam", ...
%     MaxEpochs=100, ...
%     InitialLearnRate=1e-3, ...
%     Verbose=true, ...
%     Plots='training-progress', ...
%     MiniBatchSize=1);
% 
% net = trainNetwork(X, y, layers, options);
% 
% YPred = predict(net, X);
% 
% % print and store deconv kernel
% deconv_kernel = net.Layers(2).hrfp
% 
% plot = true;
% if plot
%     subplot(1,3,1);  imagesc(img);  axis equal;  axis tight;  colormap(gray);  title('Unblurred');
%     subplot(1,3,3);  imagesc(img_blur);  axis equal;  axis tight;  colormap(gray);  title('Blurred');
%     subplot(1,3,2);  imagesc(YPred);  axis equal;  axis tight;  colormap(gray);  title('Predicted');
% end
