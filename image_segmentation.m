function [varianceImage] = image_segmentation(inputImage)
inputImage=inputImage(:,1:150);% for round number 152=>150
%size of image
[row_Image,col_Image] = size(inputImage);
size_of_block=5;
rowLoop=row_Image/size_of_block;
colLoop=col_Image/size_of_block;


%calculate variance of each blocks
for i = 1:row_Image%200
    for j = 1:col_Image%150
        if(i>201||j>151)%for true array bounds
             block_image=inputImage(i:i+4,j:j+4);
             block_image = double(block_image);
             variance_block_image = var(block_image);
        end 
         j=j+5;  
    end
    i=i+5;
    
end
blockSize = [5 5];
varFilterFunction = @(theBlockStructure) var(double(theBlockStructure.data(:)));
blockyImagevar = blockproc(inputImage,blockSize, varFilterFunction);
varianceImage=blockyImagevar(:);
end

