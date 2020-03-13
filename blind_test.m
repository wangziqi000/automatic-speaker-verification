allFiles = 'allFiles.txt';
trainList = 'train_read_trials.txt';  
testList = 'test_read_trials.txt';
blind_list = 'blind_file_list';
blind_trials = 'blind_trials';

use_pca = 0;
pca_latent_knob = 0.99999;

testScores = blind_nn(allFiles, trainList, testList, blind_list, ...
    blind_trials, use_pca, pca_latent_knob);