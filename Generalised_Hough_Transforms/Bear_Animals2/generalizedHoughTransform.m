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
im_template = imread('template_bear.png');

%Convert to Gray levels
im_gray = rgb2gray(im);
im_tem_gray = rgb2gray(im_template);

%Step A: Run smoothing filter and edge detection

%Run smoothing fitler

% maskunweighted = [1/9 1/9 1/9; 1/9 1/9 1/9; 1/9 1/9 1/9];
% 
% maskGaus5 = 1/273*[1 4  7  4  1;
%              4 16 26 16 4; 
%              7 26 41 26 7;
%              4 16 26 16 4;
%              1 4  7  4  1];
%     
% im_gray = double(im_gray)/255;
% im_gray_smoothed{1} = im_gray;
% 
% im_tem_gray = double(im_tem_gray)/255;
% im_template_smoothed{1} = im_tem_gray;
% 
% for i=2:15
%     im_gray_smoothed{i} = conv2(im_gray_smoothed{i-1},maskGaus5,'same');
%     im_template_smoothed{i} = conv2(im_template_smoothed{i-1},maskGaus5,'same');
%     
% end


%Used medfilt because I couldnt get rid of block edges for some wierd
%reason with my edges, and I believed that to be screwign with my code
%since the edges of the blocks can look like a bear body
for i=1:1:5
    
    im_gray = medfilt2(im_gray);
    
end

%Use sobel to tweak and get rid of background elements
im_gray_edge = edge(im_gray,'sobel',0.032);
%Use canny for template since it doesnt have noise or background
im_temp_edge = edge(im_tem_gray,'canny');


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
% (-2/9)pi
Angle_The = (-2/9)*pi();
Scale_The = 1.5;
R_Table_The = zeros(full_circle_angle, size(x_pos,1), 2); 
% (-7/12)pi
Angle_Phi = (-7/12)*pi();
Scale_Phi = 1.4;
R_Table_Phi = zeros(full_circle_angle, size(x_pos,1), 2);

%Find center of the template that had edge detection ran on it
template_x_center = round(size(im_temp_edge,1)/2);
template_y_center = round(size(im_temp_edge,2)/2);

%Create the bins that will be used for the algo
No_Bin = zeros(full_circle_angle,1);

%Values needed when calculating R-Table values:
    a = cos(Angle_The);
    b = sin(Angle_The);
    c = cos(Angle_Phi);
    d = sin(Angle_Phi);

i = 1;
while i <= size(x_pos,1)
    %Calculate the differentials from the center of the template:
    dx = (x_pos(i) - template_x_center);
    dy = (y_pos(i) - template_y_center);
    
    %Find out which bin we are going to use
    gradient_coe = gradient_template(x_pos(i),y_pos(i))+half_circle_angle;
    No_Bin(gradient_coe) = No_Bin(gradient_coe)+1;
    bin_coe = No_Bin(gradient_coe);
    
    %RTable_The calc:
    %X val
    R_Table_The(gradient_coe,bin_coe,1) = round(a * Scale_The * dx - b * Scale_The * dy);
    %Y val
    R_Table_The(gradient_coe,bin_coe,2) = round(b * Scale_The *dx + a * Scale_The *dy);
    
    %RTable_Phi calc:
    %X val
    R_Table_Phi(gradient_coe,bin_coe,1) = round(c * Scale_Phi *dx - d * Scale_Phi *dy);
    %Y val 
    R_Table_Phi(gradient_coe,bin_coe,2) = round(d * Scale_Phi *dx + c * Scale_Phi *dy);   
    
    i = i+1;
end

%Step B.2: Accumulator
% 
%Find verticies for image edges
[x_prime, y_prime] = find(im_gray_edge);

The_R_TablePos = zeros(size(im_gray_edge));
Phi_R_TablePos = zeros(size(im_gray_edge));

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
        
        %Set index we are entering data into for Table_Phi
        x_RTable_Phi = x_prime(outer) - R_Table_Phi(gradient_coe, i, 1);
        y_RTable_Phi = y_prime(outer) - R_Table_Phi(gradient_coe, i, 2);
        %Check if index is non negative value && (=/=0)
        if (x_RTable_Phi > 0) && (y_RTable_Phi > 0)
            %Check if the index is within size constraints
            if(y_RTable_Phi < size(im_gray_edge,2)) && (x_RTable_Phi < size(im_gray_edge,1))
                %iterator increase ++
                Phi_R_TablePos(x_RTable_Phi, y_RTable_Phi) = Phi_R_TablePos(x_RTable_Phi, y_RTable_Phi) + 1;
            end;
        end;
        
    end
    
end

%Step C: Display

%Step C.1: Bear 1
Bear_1 = mat2gray(The_R_TablePos);

imwrite(Bear_1, 'accumulatorMatrixBear1.bmp', 'bmp');

[first_bear_x,first_bear_y] = find(The_R_TablePos == max(max(The_R_TablePos)));
first_bear_x = first_bear_x(1);
first_bear_y = first_bear_y(1);

%Subplot the first bear and its accumulator matrix
subplot(222),imshow(Bear_1),title('Bear 1');
hold on;
circle(first_bear_y, first_bear_x, 3);

%Step C.2: Bear 2
Bear_2 = mat2gray(Phi_R_TablePos);

imwrite(Bear_2, 'accumulatorMatrixBear2.bmp', 'bmp');

[second_bear_x,second_bear_y] = find(Phi_R_TablePos == max(max(Phi_R_TablePos)));

%Subplot the second bear and its accumulator matrix
subplot(223),imshow(Bear_2),title('Bear 2');
hold on;
circle(second_bear_y, second_bear_x, 3);

%plot the original image
subplot(224),imshow(im),title('');
hold on;
circle(first_bear_y, first_bear_x, 30);
hold on;
circle(first_bear_y, first_bear_x, 4);
hold on;
circle(second_bear_y, second_bear_x, 30);
hold on;
circle(second_bear_y, second_bear_x, 4);

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