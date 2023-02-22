function [X, y] = read_super_image_res_dataset(data_length)
    x_path = 'C:/Users/Rashaad/Documents/Postgrad/data/deconv_tests/super-res-image/low_res/';
    y_path = 'C:/Users/Rashaad/Documents/Postgrad/data/deconv_tests/super-res-image/high_res/';
    X = [];
    y = [];

    for k = 1:data_length
        png_filename = string(k) + '.png';

        x_img_path = x_path + png_filename;
        y_img_path = y_path + png_filename;
        
        x_img = imread(x_img_path);
        y_img = imread(y_img_path);
        
        
        x_img = im2gray(x_img);
        y_img = im2gray(y_img);
        
        X(k, :, :) = x_img;
        y(k, :, :) = y_img;
    end

    X = cast(X, 'double')/255.0;
    y = cast(y, 'double')/255.0;
end