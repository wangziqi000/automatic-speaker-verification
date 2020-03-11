allFiles = 'allFiles.txt';
trainList = 'train_read_trials.txt';  
testList = 'test_phone_trials.txt';

use_pca = 0;
pca_latent_knob = 0.99999;
enable_fusion = 1;

[train_eer_1, test_score_1, test_label_1, eer_1] = fun_nn(allFiles, ...
    trainList, testList, use_pca, pca_latent_knob, enable_fusion) ;


use_pca = 1;
pca_latent_knob = 0.99999;
num_coeffs = 450;
use_delta = 0;
use_delta_delta = 0;
enable_fusion = 1;

[train_eer_2, test_score_2, test_label_2, eer_2] =  fun_lfcc( ...
    allFiles, trainList, testList, use_pca, pca_latent_knob, ...
    num_coeffs, use_delta, use_delta_delta, enable_fusion);


use_pca = 1;
pca_latent_knob = 0.99999;
num_coeffs = 40;
use_delta = 1;
use_delta_delta = 1;
enable_fusion = 1;

[train_eer_3, test_score_3, test_label_3, eer_3] =  fun_mfcc( ...
    allFiles, trainList, testList, use_pca, pca_latent_knob, ...
    num_coeffs, use_delta, use_delta_delta, enable_fusion);


use_pca = 0;
pca_latent_knob = 0.99999;
num_coeffs = 101;
use_delta = 0;
use_delta_delta = 0;
enable_fusion = 1;

[train_eer_4, test_score_4, test_label_4, eer_4] =  fun_cqcc( ...
    allFiles, trainList, testList, use_pca, pca_latent_knob, ...
    num_coeffs, use_delta, use_delta_delta, enable_fusion);




test_label = test_label_1;

ok = (test_label_1 == test_label_2);
if sum(ok) ~= length(test_label)
    disp("label_error");
end


ok = (test_label_1 == test_label_3);
if sum(ok) ~= length(test_label)
    disp("label_error");
end


ok = (test_label_1 == test_label_4);
if sum(ok) ~= length(test_label)
    disp("label_error");
end


[eer,~] = compute_eer(test_score_1, test_label);
disp(['The EER is ',num2str(eer),'%.']);

[eer,~] = compute_eer(test_score_2, test_label);
disp(['The EER is ',num2str(eer),'%.']);

[eer,~] = compute_eer(test_score_3, test_label);
disp(['The EER is ',num2str(eer),'%.']);

[eer,~] = compute_eer(test_score_4, test_label);
disp(['The EER is ',num2str(eer),'%.']);


test_score_1_norm = (test_score_1 - min(test_score_1))/(max(test_score_1) - min(test_score_1));
test_score_2_norm = (test_score_2 - min(test_score_2))/(max(test_score_2) - min(test_score_2));
test_score_3_norm = (test_score_3 - min(test_score_3))/(max(test_score_3) - min(test_score_3));
test_score_4_norm = (test_score_4 - min(test_score_4))/(max(test_score_4) - min(test_score_4));

alpha_sum = sum([1/ train_eer_1 ,1/ train_eer_2, 1/ train_eer_3, 1/ train_eer_4]);

alpha_1 = (1 / train_eer_1) / alpha_sum;
alpha_2 = (1 / train_eer_2) / alpha_sum;
alpha_3 = (1 / train_eer_3) / alpha_sum;
alpha_4 = (1 / train_eer_4) / alpha_sum;

% testScores = (alpha_1 * test_score_1_norm + alpha_2 * test_score_2_norm + ...
%                     alpha_3 * test_score_3_norm + alpha_4 * test_score_4_norm);

% testScores = (2 * test_score_1_norm + 1 * test_score_2_norm)/ 2;

testScores = (6 * test_score_1_norm + 1 * test_score_2_norm + 1*test_score_3_norm + 1*test_score_4_norm)/ 6;


[eer,~] = compute_eer(testScores, test_label);
disp(['The EER is ',num2str(eer),'%.']);

