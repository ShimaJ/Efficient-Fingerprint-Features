%
%
% Matlab Code for R2019b version
%code for "Efficient Fingerprint Features for Gender Recognition " journal
% Code written by shima jalali
%
% load images from datasets and do all preproccess steps in "for" loop for
% all of images in dataset.
%image feature extraction

[outMinutiaExractionImgage,bifurcation_x, bifurcation_y,ridge_x, ridge_y,vally_x, vally_y]=minutiaExtraction(input_image);
len=length(bifurcation_x);
%For Display
for i=1:len
    outMinutiaExractionImgage((bifurcation_x(i)-3):(bifurcation_x(i)+3),(bifurcation_y(i)-3),1:2)=0;
    outMinutiaExractionImgage((bifurcation_x(i)-3):(bifurcation_x(i)+3),(bifurcation_y(i)+3),1:2)=0;
    outMinutiaExractionImgage((bifurcation_x(i)-3),(bifurcation_y(i)-3):(bifurcation_y(i)+3),1:2)=0;
    outMinutiaExractionImgage((bifurcation_x(i)+3),(bifurcation_y(i)-3):(bifurcation_y(i)+3),1:2)=0;
    outMinutiaExractionImgage((bifurcation_x(i)-3):(bifurcation_x(i)+3),(bifurcation_y(i)-3),3)=255;
    outMinutiaExractionImgage((bifurcation_x(i)-3):(bifurcation_x(i)+3),(bifurcation_y(i)+3),3)=255;
    outMinutiaExractionImgage((bifurcation_x(i)-3),(bifurcation_y(i)-3):(bifurcation_y(i)+3),3)=255;
    outMinutiaExractionImgage((bifurcation_x(i)+3),(bifurcation_y(i)-3):(bifurcation_y(i)+3),3)=255;
end

%For Display
len=length(ridge_x);
for i=1:len
    outMinutiaExractionImgage((ridge_x(i)-3):(ridge_x(i)+3),(ridge_y(i)-3),2:3)=0;
    outMinutiaExractionImgage((ridge_x(i)-3):(ridge_x(i)+3),(ridge_y(i)+3),2:3)=0;
    outMinutiaExractionImgage((ridge_x(i)-3),(ridge_y(i)-3):(ridge_y(i)+3),2:3)=0;
    outMinutiaExractionImgage((ridge_x(i)+3),(ridge_y(i)-3):(ridge_y(i)+3),2:3)=0;
    
    outMinutiaExractionImgage((ridge_x(i)-3):(ridge_x(i)+3),(ridge_y(i)-3),1)=255;
    outMinutiaExractionImgage((ridge_x(i)-3):(ridge_x(i)+3),(ridge_y(i)+3),1)=255;
    outMinutiaExractionImgage((ridge_x(i)-3),(ridge_y(i)-3):(ridge_y(i)+3),1)=255;
    outMinutiaExractionImgage((ridge_x(i)+3),(ridge_y(i)-3):(ridge_y(i)+3),1)=255;
end

%lbp
xtractLBPFeatures(input_image);
%dct2
tempDct= dct2(input_image);
%entropy
 entropy(input_image);
 %RTVTR
 rtvtr=length(ridge_x)/length(vally_x);
