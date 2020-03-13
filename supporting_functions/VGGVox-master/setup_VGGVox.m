function setup_VGGVox()
%SETUP_VGGVOX Sets up VGGVox, by adding its folders 
% to the Matlab path
%
% Copyright (C) 2017 Arsha Nagrani
  addpath '/home/ziqi/matconvnet-1.0-beta25/matlab'
  vl_setupnn ;
  root = fileparts(mfilename('fullpath')) ;
  addpath(root, [root '/matlab']) ;
  addpath(genpath('mfcc'))
