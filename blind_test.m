% This file is used for testing each blind_"method" functions

%% blind_nn

allFiles = 'allFiles.txt';
trainList = 'train_read_trials.txt';  
testList = 'test_read_trials.txt';
blind_list = 'blind_file_list';
blind_trials = 'blind_trials';

use_pca = 0;
pca_latent_knob = 0.99999;

testScores = blind_nn(allFiles, trainList, testList, blind_list, ...
    blind_trials, use_pca, pca_latent_knob);

%% blind_mfcc
allFiles = 'allFiles.txt';
trainList = 'train_read_trials.txt';  
testList = 'test_read_trials.txt';
blind_list = 'blind_file_list';
blind_trials = 'blind_trials';

use_pca = 0;
pca_latent_knob = 0.99999;

num_coeffs = 33;
use_delta = 0;
use_delta_delta = 0;

testScores = blind_mfcc(allFiles, trainList, testList, ...
    blind_list, blind_trials, use_pca, pca_latent_knob, num_coeffs,...
    use_delta, use_delta_delta);

%% blind_lfcc
allFiles = 'allFiles.txt';
trainList = 'train_read_trials.txt';  
testList = 'test_read_trials.txt';
blind_list = 'blind_file_list';
blind_trials = 'blind_trials';

use_pca = 1;
pca_latent_knob = 0.99999;

num_coeffs = 511;
use_delta = 0;
use_delta_delta = 0;

testScores = blind_lfcc(allFiles, trainList, testList, ...
    blind_list, blind_trials, use_pca, pca_latent_knob, num_coeffs,...
    use_delta, use_delta_delta);