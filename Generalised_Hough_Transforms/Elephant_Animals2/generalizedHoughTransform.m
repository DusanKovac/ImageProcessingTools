%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File Name:   generaliszedHoughTransform.m
% Description: Using GHT's find the shapes and letters
% Input:       .png files which are unsharpened 
% Output:      .bmp files that have been sharpened
% Environment: Matlab R2019b
% Usage:       In Matlab Command Window, type 'generaliszedHoughTransform'.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;

full_circle_angle = 360;
half_circle_angle = 180;

%Original image, we will need to gray value it later and use smoothing
%filter
im = imread('animals2.jpg');

%Template we searching for, will need to be gray scaled
im_template = imread('template_elephant.png');

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

%Use sobel to tweak and get rid of background elements
im_gray_edge = edge(im_gray_smoothed{2},'sobel',0.03);
%Use canny for template since it doesnt have noise or background
im_temp_edge = edge(im_tem_gray,'sobel',0.03);

imwrite(im_gray_edge, 'Animals_edges.bmp', 'bmp');
imwrite(im_temp_edge, 'Template_edge.bmp','bmp');


%Step B: Start the algo
%Step B.1: Find R-Tables

%Calculate gradients of the template and the image:
gradient_template = atan2(imfilter(double(im_temp_edge),[1; -1],'same'),imfilter(double(im_temp_edge),[1  -1],'same'))*(half_circle_angle)/pi();
gradient_image = atan2(imfilter(double(im_gray_edge),[1; -1],'same'),imfilter(double(im_gray_edge),[1  -1],'same'))*(half_circle_angle)/pi();


%find edge vertices from template
[x_pos, y_pos] = find(im_temp_edge > 0);

%Once we find the edge we can create are R_table for it:

%Dealing with the first rotation
% (-1/6)pi
Angle_The = (-1/11)*pi();
scale = 1.5;
R_Table_The = zeros(full_circle_angle, size(x_pos,1), 2); 

%Find center of the template that had edge detection ran on it
template_x_center = round(size(im_temp_edge,1)/2);
template_y_center = round(size(im_temp_edge,2)/2);

%Create the bins that will be used for the algo
No_Bin = zeros(full_circle_angle,1);

%Values needed when calculating R-Table values:
    a = cos(Angle_The);
    b = sin(Angle_The);

i = 1;
while i <= size(x_pos,1)
    %Calculate the differentials from the center of the template:
    dx = x_pos(i) - template_x_center;
    dy = y_pos(i) - template_y_center;
    
    %Find out which bin we are going to use
    gradient_coe = gradient_template(x_pos(i),y_pos(i))+half_circle_angle;
    No_Bin(gradient_coe) = No_Bin(gradient_coe)+1;
    bin_coe = No_Bin(gradient_coe);
    
    dx = scale * dx;
    dy = scale * dy;
    
    %RTable_The calc:
    %X val
    R_Table_The(gradient_coe,bin_coe,1) = round(a * dx - b * dy);
    %Y val
    R_Table_The(gradient_coe,bin_coe,2) = round(b * dx + a * dy);

    i = i+1;
end

%Step B.2: Accumulator
% 
%Find verticies for image edges
[x_prime, y_prime] = find(im_gray_edge);

The_R_TablePos = zeros(size(im_gray_edge));
for outer=1:1:size(x_prime,1)
    
    %Find the index of the bin we are going to iterate through
    gradient_coe = half_circle_angle + gradient_image(x_prime(outer),y_prime(outer));
    
    %Iterate through bin
    for i=1:1:No_Bin(gradient_coe)
        
        %Set index we are entering data into for Table_The
        x_Rtable_The = x_prime(outer) - R_Table_The(gradient_coe, i, 1);
        y_Rtable_The = y_prime(outer) - R_Table_The(gradient_coe, i, 2);
        
        %Check if index is non negative value && (=/=0)
        if (x_Rtable_The > 0) && (y_Rtable_The > 0)
            %Check if the index is within size constraints
            if(x_Rtable_The < size(im_gray_edge,1)) && (y_Rtable_The < size(im_gray_edge,2))
                %iterator increase ++
                The_R_TablePos(x_Rtable_The, y_Rtable_The) = The_R_TablePos(x_Rtable_The, y_Rtable_The) + 1;
            end;
       end;
        
    end
    
end

%Step C: Display

%Step C.1: Bear 1
Elephant_1 = mat2gray(The_R_TablePos);

imwrite(Elephant_1, 'accumulatorMatrixBear1.bmp', 'bmp');

[first_elephant_x,first_elephant_y] = find(The_R_TablePos == max(max(The_R_TablePos)));
% first_elephant_x = first_elephant_x(1); 
% first_elephant_y = first_elephant_y(1);

%Subplot the first bear and its accumulator matrix
subplot(221),imshow(Elephant_1),title('Elephant 1');
hold on;
circle(first_elephant_y, first_elephant_x, 3);


%plot the original image
subplot(222),imshow(im),title('');
hold on;
circle(first_elephant_y, first_elephant_x, 35);
hold on;
circle(first_elephant_y, first_elephant_x, 4);

% The following link was used as reference on how to create a subplot
% circle, I had no idea how to do one so I followed this individuals
% recommendation.
% https://www.mathworks.com/matlabcentral/answers/3058-plotting-circles

function circle(x,y,r)
    ang=0:0.01:2*pi; 
    xp=r*cos(ang);
    yp=r*sin(ang);
    plot(x+xp,y+yp);
end
