clc
clear 

% Define the paths to the input image and output mask directories
image_dir = fullfile('C:\Users\muham\Pictures\nex\kvid\images\');
mask_dir = fullfile('C:\Users\muham\Pictures\nex\kvid\masks_converted\');

% Load the CSV file with headers
data = readtable('C:\Users\muham\Pictures\nex\kvid\train_master.csv');

headers = data.Properties.VariableDescriptions;

% Select a random image and mask pair
idx = randi(height(data));
img_path = data.(headers{1}){idx};
mask_path = data.(headers{2}){idx};

% Load the image and mask
img = imread(img_path);
mask = imread(mask_path);

% Create a new figure with two subplots
figure;
subplot(1,2,1);
imshow(img);
title('Image');

subplot(1,2,2);
imshow(mask);
title('Mask');
