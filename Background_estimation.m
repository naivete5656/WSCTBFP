clear all
close all
clc

%% Read a phase contrast image:
addpath('./func')

debug = 0;
img_path = sprintf('./data/img/*.tif');
output_path = './data/bg';

img_paths = dir(img_path);

for frame = 1:length(img_paths)
    img_path = img_paths(frame);
    phc_img = imread([img_path.folder,'/',img_path.name]);
    
    phc_img=uint8(phc_img);
    if ndims(phc_img)==3
        grayimg=rgb2gray(phc_img);
    else
        grayimg=phc_img;
    end
    img=im2double(grayimg);

	[nrows, ncols] = size(img);
	N = nrows*ncols;

	%for flatten images
	[xx yy] = meshgrid(1:ncols, 1:nrows);
	xx = xx(:); yy = yy(:);
	X = [ones(N,1), xx, yy, xx.^2, xx.*yy, yy.^2];
    
    p = X\img(:); 	
    background = reshape(X*p,[nrows,ncols]);
    
    out = sprintf([output_path '/%05d.tif'], frame-1);;
    imwrite(background,out);
 
end