allFiles = 'allFiles.txt';
trainList = 'train_read_trials.txt';  
testList = 'test_read_trials.txt';

use_pca = 0;
pca_latent_knob = 0.99999;


use_delta = 0;
use_delta_delta = 0;

enable_fusion = 0;

max_num_coeff = 500;
eer_array = zeros(max_num_coeff, 1);

parfor num_coeffs = 1:max_num_coeff

    [trainEER, testScores, testLabels, eer] =  fun_cqcc( ...
    allFiles, trainList, testList, use_pca, pca_latent_knob, ...
    num_coeffs, use_delta, use_delta_delta, enable_fusion);
    
    eer_array(num_coeffs) = eer;
    
end

figure()
plot(1:max_num_coeff, eer_array);