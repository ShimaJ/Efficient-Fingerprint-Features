function [varargout] = lineFiltLBP(inImg, nFiltDims, hLBP, isScale, targetClass)
%% lineLBP
% The function implements LLBP (Line Local Binary Pattern) analysis.
%
%% Syntax
%  lineLBP(inImg, nFiltDims, hLBP, isScale, targetClass);
%  LLBP = lineLBP(inImg, nFiltDims, hLBP, isScale, targetClass);
%  [LLBP, verticalLBP, horizontalLBP] = lineLBP(inImg, nFiltDims, hLBP, isScale, targetClass);
%  [LLBP, leftLBP, rightLBP, topLBP, bottomLBP] = lineLBP(inImg, nFiltDims, hLBP, isScale, targetClass);
%
%% Description
% The LLBP tests the relation between pixel and its neighbors, encoding this relation into
%   a binary word. This allows detection of patterns/features. The current version uses horizontal
%   and vertical shaped filter, resulting in "line" shape, for which the name is given. 
%   An illustration of Line LBP can be seen here: 
%       http://www.ncbi.nlm.nih.gov/pmc/articles/PMC3251987/figure/f6-sensors-11-11357/
%   While this shape implies options for specific efficient 1D filter based implementation, here I
%   implemented it via wrapping of my "efficientLBP" function- which achieves not optimal, but
%   reasonable run-time. And I like code reuse :)
%
%% Input arguments (defaults exist):
% inImg- input image, a 2D matrix/3D color images.
% nFiltDims- a 2 elements integer vector, defining vertical and horizontal filter dimensions.
% hLBP- the LBP function used for intermediate LBP data extraction, currently @pixelwiseLBP and
%   @efficientLBP are supported. It is strongly advised to use efficientLBP.
% isScale- scales returned data to [0, 1]- making it easy to present (via imshow), analyze and
%   store (via imwrite).
% targetClass- a string specifying the class of the returned data. Any numerical Matlab class
%   uint8, single, double, uint16
%
%% Output arguments
%   missing outputs will result in presenting figures.
%   LLBP-    lineLBP image UINT8/UINT16/UINT32/UINT64/DOUBLE of same dimensions as inImg.
%   leftLBP-   partial (left) lineLBP image UINT8/UINT16/UINT32/UINT64/DOUBLE of dimensions as inImg.
%   rightLBP-  partial (right) lineLBP image UINT8/UINT16/UINT32/UINT64/DOUBLE of dimentions as inImg.
%   topLBP-    partial (top) lineLBP image UINT8/UINT16/UINT32/UINT64/DOUBLE of dimensions as inImg.
%   bottomLBP- partial (down) lineLBP image UINT8/UINT16/UINT32/UINT64/DOUBLE of dimentions as inImg.
%
%% Issues & Comments
% - The implementation is not optimal, we could achieve same result with less filtering operations,
%   and utilize the fact the shift here are integer.
%
%% Example
% img=imread('peppers.png');
% lineFiltLBP(img, [17, 17]);
%
%% See also
% A filtering based implementation of LBP supporting arbitrary shaped neighborhood.
%   efficientLBP:  http://www.mathworks.com/matlabcentral/fileexchange/36484-local-binary-patterns     
% A shift based implementation of LBP supporting arbitrary shaped neighborhood.
%   shiftBasedLBP: http://www.mathworks.com/matlabcentral/fileexchange/49787-shift-based-lbp     
%
%% Revision history
% First version: Nikolay S. 2015-03-16.
% Last update:   Nikolay S. 2015-08-22.
%
% *List of Changes:*
% 2015-08-22- minor bug fixes- change from vargout to varargout (typo) and replace nested if-else
%   dealing with varargout population to swich-case.

%% Default parameters
if nargin < 2
    nFiltDims = [17, 17];
end
if nargin < 3
    hLBP = @efficientLBP;	% @pixelwiseLBP, @efficientLBP
end
if nargin < 4
    isScale = false;
end
if nargin < 5
    targetClass = [];
end

%% Verify parameters are legal
nFiltDims = nFiltDims + mod(nFiltDims+1, 2); % convert any even value to odd
if numel(nFiltDims) == 1
    nFiltDims = repmat(nFiltDims, [2, 1]);
end
if isScale
    targetClass = 'single';
end

%% Construct filters
upFiltLLBP = generateFilterLLBP( nFiltDims(1) );
if numel(nFiltDims)==1 || nFiltDims(1) == nFiltDims(2);
    leftFiltLLBP = permute(upFiltLLBP, [2, 1, 3]);
else
    leftFiltLLBP = generateFilterLLBP( nFiltDims(2) );
end
downFiltLLBP  = upFiltLLBP(end:-1:1, :, :);
rigthFiltLLBP = leftFiltLLBP(:, end:-1:1, :);

isChanWiseRot = false;
paramsFiltLBP = {'isRotInv', false, 'isChanWiseRot', isChanWiseRot};
% Calculate 2 Vertical Line components
upLLBP    = hLBP(inImg, 'filtR', upFiltLLBP,    paramsFiltLBP{:});
downLLBP  = hLBP(inImg, 'filtR', downFiltLLBP,  paramsFiltLBP{:});
% Calculate 2 Horizontal Line components
leftLLBP  = hLBP(inImg, 'filtR', leftFiltLLBP,  paramsFiltLBP{:});
rightLLBP = hLBP(inImg, 'filtR', rigthFiltLLBP, paramsFiltLBP{:});

vertLLBP = single(leftLLBP) + single(rightLLBP);
horzLLBP = single(upLLBP)   + single(downLLBP);

% Combine Vrtical and Horizontal lines combinations
LLBP = sqrt(horzLLBP.^2 + vertLLBP.^2);

if isScale
    % Scale data to be [0, 1]
    upLLBP = scale(upLLBP);
    downLLBP = scale(downLLBP);
    leftLLBP = scale(leftLLBP);
    rightLLBP = scale(rightLLBP);
    LLBP = scale(LLBP);    
end

if isempty(targetClass)
    LLBP = cast( LLBP, 'like', upLLBP);
elseif ~strcmpi( targetClass, class(LLBP) )
    LLBP = cast( LLBP, targetClass);
end

%% Prepare Out arguments
% return out arguments accourding to number of outarguments sepcifyied by calling function
switch(nargout)
    case(0)
        figure;
        if size(inImg, 3) == 1
            subplot(2, 2, 1); imshow(upLLBP, []); title('Up LLBP'); colorbar;
            subplot(2, 2, 2); imshow(downLLBP, []); title('Down LLBP'); colorbar;
            subplot(2, 2, 3); imshow(leftLLBP, []); title('Left LLBP'); colorbar;
            subplot(2, 2, 4); imshow(rightLLBP, []); title('Right LLBP'); colorbar;
            
            figure;
            subplot(1, 2, 1); imshow(inImg, []);  title('Input image'); colorbar;
            subplot(1, 2, 2); imshow(LLBP, []);  title('Whole LLBP'); colorbar;
        else
            if ~isScale
                upLLBP = scale(upLLBP);
                downLLBP = scale(downLLBP);
                leftLLBP = scale(leftLLBP);
                rightLLBP = scale(rightLLBP);
                
                inImg = scale(inImg);
                LLBP = scale(LLBP);
            end
            subplot(2, 2, 1); imshow(upLLBP); title('Up LLBP');
            subplot(2, 2, 2); imshow(downLLBP); title('Down LLBP');
            subplot(2, 2, 3); imshow(leftLLBP); title('Left LLBP');
            subplot(2, 2, 4); imshow(rightLLBP); title('Right LLBP');
            
            figure;
            subplot(1, 2, 1); imshow(inImg);  title('Input image');
            subplot(1, 2, 2); imshow(LLBP);  title('Whole LLBP');
        end
    case(1)
        varargout{1} = LLBP;
    case(2)
        varargout{1} = LLBP;
        varargout{2} = cat(3, leftLLBP, rightLLBP, upLLBP, downLLBP);
    case(3)
        varargout{1} = LLBP;
        varargout{2} = cat(3, leftLLBP, rightLLBP);
        varargout{3} = cat(3, upLLBP,   downLLBP);
    case(5)
        varargout{1} = LLBP;
        varargout{2} = leftLLBP;
        varargout{3} = rightLLBP;
        varargout{4} = upLLBP;
        varargout{5} = downLLBP;
end

end

%% Servise function
% Scale input matrix each 2D slice to be [0, 1] 
function inMat = scale(inMat)
inMat = single(inMat);
nClrs=size(inMat,3);
for iClr = 1:nClrs
    currClr = inMat(:, :, iClr);
    currClr = currClr-min( currClr(:) );
    currClr = currClr/max( currClr(:) );
    inMat(:, :, iClr) = currClr;
end

end