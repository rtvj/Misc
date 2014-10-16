% *************************************************************************
% Title: Disparity Map of Stereoscopic Images using Adaptive Threshold
% Author: Akshay Hodigere Arunkumar
%		  Rutvij Girish Kharkanis
%
% Created: April 2014
% *************************************************************************

clear all
ax=imread('ra.jpg');
bx=imread('la.jpg');
ax = imresize(ax,[288 384]);
bx = imresize(bx,[288 384]);
[z,y,l]=size(ax);
a=int16(ax);
b=int16(bx); 
d(z,y)=uint8(0);
c(1,1:3,l)=int16(0);
i=1;
tic;
for n=1:(z/2)
m1=1;
m2=3;
 for m=1:y-3
     x=0;
   while d(2*n,m)~=i && x<100
       i=3;
    while i<15 && m+i<y-3
    i=i+1;
    c=b(2*n,m1:m2,:)-a(2*n,m1+i:m2+i,:); 
    c=abs(c);
    cs1=sum(c);
    cs=sum(cs1);
     if(cs<x)
         d(2*n,m)=i;
         d(2*n-1,m)=i;
         break;
     end
    end
     if d(2*n,m)~=0
         break;
     end
    x=x+10;
   end
    m1=m1+1;
    m2=m2+1;
 end 
end
dispMap = d.*d;
H = fspecial('average', [3 3]);
dispMap = imfilter(dispMap, H);

timeTaken=toc
imwrite(uint8(dispMap),'MATLAB_Outa.jpg');
imshow((uint8(dispMap)));
