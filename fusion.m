allFiles = 'allFiles.txt';
trainList = 'train_read_trials.txt';  
testList = 'test_read_trials.txt';

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


ok = (test_label_1 == test_label_2);
test_label = test_label_1;
if sum(ok) ~= length(test_label)
    disp("label_error");
end

[eer,~] = compute_eer(test_score_1, test_label);
disp(['The EER is ',num2str(eer),'%.']);

[eer,~] = compute_eer(test_score_2, test_label);
disp(['The EER is ',num2str(eer),'%.']);

test_score_1_norm = (test_score_1 - min(test_score_1))/(max(test_score_1) - min(test_score_1));
test_score_2_norm = (test_score_2 - min(test_score_2))/(max(test_score_2) - min(test_score_2));

alpha_1 = (1 / train_eer_1) / ((1/ train_eer_1) + (1/ train_eer_2));
alpha_2 = (1 / train_eer_2) / ((1/ train_eer_1) + (1/ train_eer_2));

testScores = (alpha_1 * test_score_1_norm + alpha_2 * test_score_2_norm)/ 2;

% testScores = (1 * test_score_1_norm + 2 * test_score_2_norm)/ 2;

[eer,~] = compute_eer(testScores, test_label);
disp(['The EER is ',num2str(eer),'%.']);

