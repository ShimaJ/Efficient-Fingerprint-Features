%
%
% Matlab Code for R2019b version
%code for "Efficient Fingerprint Features for Gender Recognition " journal
% Code written by shima jalali
%
% load images from datasets and do all preproccess steps in "for" loop for
% all of images in dataset.
%image preproccess

%A:segmentation
image_segmentation(input_image);

%B:normalization
normalizedImage = uint8(255*mat2gray(input_image));

%C:	medianFilter
image_normalization= medfilt2(input_image);

%D:	binary
binary_image = binary_image(input_image);
%figure;imshow(binary_image);title('Input image');

%E: thinning
thin_image=~bwmorph(binary_image,'thin',Inf);










