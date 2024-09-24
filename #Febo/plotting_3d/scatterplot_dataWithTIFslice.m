% Simplified 3D Neuroimaging Visualization Script

% MATLAB R2021b
% This function calculates plots an array of values given XYZ data, and
% assigns marker color based on values, then plots selected TIF in plane

% Author: Jason Coleman <jcoleman@ufl.edu>
% 2024-0829 Created function

% Load data
[filename, pathname] = uigetfile('*.tif', 'Select TIF-STACK file');
fullpath = fullfile(pathname, filename);
img = imread(fullpath);
%%
% Simplified 3D Neuroimaging Visualization Script

% Load data
[filename, pathname] = uigetfile('*.tif', 'Select TIF-STACK file');
fullpath = fullfile(pathname, filename);
FinalImage = load_tif_stack(fullpath);

%%

% Adjust Z as needed (THIS IS SETUP FOR THE VASCULAR IMGAES Z, also fine just pick a TIF and tell where on Z to plot)
% Zmatch = 61; % 
% iFILE=16; %number of file fieldname
% z1_midSLice_tmp = datacell_Tprofiles{1, iFILE}.Z1.Zcoords %m5_roi2_00003... %pull Zcoords from var
% [~, idx] = min(abs(z1_midSLice_tmp - Zmatch));
% sameSLices = find(abs(z1_midSLice_tmp - Zmatch) == min(abs(z1_midSLice_tmp - Zmatch)));
% ztslicenumber=z1_midSLice_tmp(sameSLices);

% just set the slice number of interest and use the correct single-TIF file
ztslicenumber = 100; 
img = FinalImage(:,:,ztslicenumber);
%%
% % Load data
% [filename, pathname] = uigetfile('*.tif', 'Select TIF-STACK file');
% fullpath = fullfile(pathname, filename);
% FinalImage = load_tif_stack(fullpath);

% Load or generate ROI data
% Assuming you have functions to load or generate this data
[x, y, z, fwhm] = load_roi_data();

% Create figure
fig = figure('Position', [100, 100, 800, 600]);

% Subplot 2: 3D scatter plot
scatter3(x, y, z, 50, fwhm, 'filled');
set(gca, 'ZDir', 'reverse'); % Invert Z-axis
colormap(gca, jet);
cb = colorbar;
ylabel(cb, 'FWHM (microns)');
caxis([min(fwhm) max(fwhm)]);

% Fix jet index scale
set(gca, 'CLim', [min(fwhm) max(fwhm)]);

% Set view and labels
view(-34, 14);
xlabel('x (pixels)');
ylabel('y (pixels)');
zlabel('z (microns)');
zlim([-20 300]);
pbaspect([1 1 1]);
title(['3D Scatter Plot, Z slice = ' num2str(ztslicenumber)]);

%%
% Plot image in same plane as 3D scatter plot
g = hgtransform('Matrix', makehgtform('translate', [0 0 ztslicenumber(1)]));
imagesc(g, img);
%set(g, 'Parent', gca); % Set parent axes to current axes
img_1 = img(1,:)
%set(gca, 'CLim', [min(img_1) max(img_1)]);
colormap(gca, gray); % Set colormap to gray - but only TIF - how?

% Overall figure title
title(strrep(filename, '_', ' '));

%% FUNCTIONS
% Helper function to load TIF stack
function tif_stack = load_tif_stack(filepath)
    info = imfinfo(filepath);
    num_images = numel(info);
    tif_stack = zeros(info(1).Height, info(1).Width, num_images, 'uint16');
    for k = 1:num_images
        tif_stack(:,:,k) = imread(filepath, k);
    end
end

% Helper function to load or generate ROI data
% Replace this with your actual data loading method
function [x, y, z, fwhm] = load_roi_data()
    % Example: Generate random data
    num_points = 200;
    x = randi([1, 512], num_points, 1);
    y = randi([1, 512], num_points, 1);
    z = randi([1, 300], num_points, 1);
    fwhm = rand(num_points, 1) * 10; % Random FWHM values between 0 and 10
end