% This file is used to generated test score for blind test data using
% score-level fusion method

allFiles = 'allFiles.txt';
trainList = 'train_read_trials.txt';  
testList = 'test_read_trials.txt';
blind_list = 'blind_file_list';
blind_trials = 'blind_trials';

eval_prediction = 'ziqi_qiong_yuchun_blind_label_fusion.txt'; 

% nn
use_pca = 0;
pca_latent_knob = 0.99999;

test_score_1 = blind_nn(allFiles, trainList, testList, blind_list, ...
    blind_trials, use_pca, pca_latent_knob);

% lfcc
use_pca = 1;
pca_latent_knob = 0.99999;
num_coeffs = 450;
use_delta = 0;
use_delta_delta = 0;


test_score_2 = blind_lfcc(allFiles, trainList, testList, ...
    blind_list, blind_trials, use_pca, pca_latent_knob, num_coeffs,...
    use_delta, use_delta_delta);

% mfcc
use_pca = 1;
pca_latent_knob = 0.99999;
num_coeffs = 40;
use_delta = 1;
use_delta_delta = 1;

test_score_3 = blind_mfcc(allFiles, trainList, testList, ...
    blind_list, blind_trials, use_pca, pca_latent_knob, num_coeffs,...
    use_delta, use_delta_delta);

test_score_1_norm = (test_score_1 - min(test_score_1))/(max(test_score_1) - min(test_score_1));
test_score_2_norm = (test_score_2 - min(test_score_2))/(max(test_score_2) - min(test_score_2));
test_score_3_norm = (test_score_3 - min(test_score_3))/(max(test_score_3) - min(test_score_3));

testScores = (3 * test_score_1_norm + 1 * test_score_2_norm + 1 * test_score_3_norm)/ 5;
fid=fopen(eval_prediction,'w');
fprintf(fid,'%f\n',testScores);
fclose(fid);
