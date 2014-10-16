% *************************************************************************
% Title: Function-Compute Correlation between two images using the 
% similarity measure of Sum of Hamming Distances (SHD) with Right Image 
% as reference.
% Author: Siddhant Ahuja
% Modified by: Akshay Hodigere Arunkumar & Rutvij Girish Kharkanis
% Created: May 2008, Modified: April 2014
% Copyright Siddhant Ahuja, 2008
% *************************************************************************

leftImage = imread('ra.jpg');
rightImage = imread('la.jpg');
windowSize=9;
dispMin=0;
dispMax=15;
leftImage=rgb2gray(leftImage);   
rightImage=rgb2gray(rightImage);
leftImage = imresize(leftImage,[288 384]);
rightImage = imresize(rightImage,[288 384]);
[nrLeft,ncLeft] = size(leftImage);
dispMap=zeros(nrLeft, ncLeft);
win=(windowSize-1)/2;
numberOfBits=8;
tic;
for(i=1+win:1:nrLeft-win)
    for(j=1+win:1:ncLeft-win-dispMax)
        min=0;
        position=0;
        rightWindow=rightImage(i, j-win:j+win);
        for(dispRange=dispMin:1:dispMax)
            if (j+win+dispRange <= ncLeft)
                leftWindow=leftImage(i, j-win+dispRange:j+win+dispRange);
                bloc3=bitxor(rightWindow,leftWindow);
                distance=uint8(zeros(1,windowSize));
                for (k=1:1:numberOfBits)
                    distance=distance+bitget(bloc3,k);
                end
                dif=sum(sum(distance));
                if (dispRange==0)
                    min=dif;
                elseif (min>dif)
                    min=dif;
                    position=dispRange;
                end
            end 
        end
        dispMap(i,j) = position*position;
    end
end

H = fspecial('average', [3 3]);
dispMap = imfilter(dispMap, H);

timeTaken=toc
imwrite(uint8(dispMap),'MATLAB_Outa.jpg');
imshow((uint8(dispMap)));