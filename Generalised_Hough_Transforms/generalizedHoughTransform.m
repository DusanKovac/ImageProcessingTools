%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File Name:   simpleEdgeDetection.m
% Description: Uses matlab functions and edge operators we learnt in class
% Input:       .png files which are unsharpened 
% Output:      .bmp files that have been sharpened
% Environment: Matlab R2018b
% Usage:       In Matlab Command Window, type 'simpleEdgeDetection'.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Original image, we will need to gray value it later and use smoothing
%filter
im = imread('animals.jpg');

%Template we searching for, will need to be gray scaled
im_template = imread('template_bear.png');

%Convert to Gray levels
im_gray = rgb2gray(im);
im_tem_gray = rgb2gray(im_template);

%Step A: Run smoothing filter and edge detection

%Run smoothing fitler

maskunweighted = [1/9 1/9 1/9; 1/9 1/9 1/9; 1/9 1/9 1/9];

maskGaus5 = 1/273*[1 4  7  4  1;
             4 16 26 16 4; 
             7 26 41 26 7;
             4 16 26 16 4;
             1 4  7  4  1];
    
im_gray = double(im_gray)/255;
im_gray_smoothed{1} = im_gray;

im_tem_gray = double(im_tem_gray)/255;
im_template_smoothed{1} = im_tem_gray;

for i=2:15
    im_gray_smoothed{i} = conv2(im_gray_smoothed{i-1},maskGaus5,'same');
    im_template_smoothed{i} = conv2(im_template_smoothed{i-1},maskGaus5,'same');
    
end

im_gray_edge = edge(im_gray_smoothed{8},'sobel',0.008);
im_template_edge = edge(im_template_smoothed{8},'sobel',0.008);

% imshow(im_gray_edge);

%Step B: Generate R-Table

%get template edge point:
[x,y] = find(temp > 0);
maxP = size(x,1);
%get the gradient of tamplate:
grad = Gradient(temp);
maxA = 180;

%Rtable with rotation:
rtable = zeros(2*maxA, maxP, 2);
binCount = zeros(2*maxA,1);

for i=1:1:maxP
    k = grad(x(i), y(i)) + 180;
    binCount(k) = binCount(k) + 1;
    h = binCount(k);
%get dx and dy:
    delta_x = x(i) - refPointx;
    delta_y = y(i) - refPointy;
%rotation:
    ang = 90*pi()/180;
    rtable(k, h, 1) = round(cos(ang)*delta_x - sin(ang)*delta_y);
    rtable(k, h, 2) = round(sin(ang)*delta_x + cos(ang)*delta_y);
end;



