%% fun_mfcc.m

allFiles = 'allFiles.txt';
trainList = 'train_read_trials.txt';  
testList = 'test_read_trials.txt';

use_pca = 0;
pca_latent_knob = 0.99999;

num_coeffs = 33;
use_delta = 0;
use_delta_delta = 0;

enable_fusion = 0;

[trainEER, testScores, testLabels, eer] =  fun_mfcc( ...
    allFiles, trainList, testList, use_pca, pca_latent_knob, ...
    num_coeffs, use_delta, use_delta_delta, enable_fusion);


%% fun_cqcc.m
allFiles = 'allFiles.txt';
trainList = 'train_read_trials.txt';  
testList = 'test_mismatch_trials.txt';

use_pca = 0;
pca_latent_knob = 0.99999;

num_coeffs = 40;
use_delta = 1;
use_delta_delta = 1;

enable_fusion = 0;

[trainEER, testScores, testLabels, eer] =  fun_cqcc( ...
    allFiles, trainList, testList, use_pca, pca_latent_knob, ...
    num_coeffs, use_delta, use_delta_delta, enable_fusion);

%% fun_lfcc.m

allFiles = 'allFiles.txt';
trainList = 'train_read_trials.txt';  
testList = 'test_read_trials.txt';

use_pca = 1;
pca_latent_knob = 0.99999;

num_coeffs = 450;
use_delta = 0;
use_delta_delta = 0;

enable_fusion = 0;

[trainEER, testScores, testLabels, eer] =  fun_lfcc( ...
    allFiles, trainList, testList, use_pca, pca_latent_knob, ...
    num_coeffs, use_delta, use_delta_delta, enable_fusion);

%% fun_nn.m
allFiles = 'allFiles.txt';
trainList = 'train_read_trials.txt';  
testList = 'test_read_trials.txt';

use_pca = 0;
pca_latent_knob = 0.99999;

enable_fusion = 1;

[trainEER, testScores, testLabels, eer] = fun_nn(allFiles, ...
    trainList, testList, use_pca, pca_latent_knob, enable_fusion) ;



%% VCTK_ivector_given_T.m
allFiles = 'allFiles.txt';
trainList = 'train_read_trials.txt';  
testList = 'test_phone_trials.txt';

use_pca = 0;
pca_latent_knob = 0.99999;

enable_fusion = 0;

nmix = 256;
nWorkers = 8;   % Num for parpool
tvDim = 200;
use_score = 1;

[trainEER, testScores, testLabels, eer] = VCTK_ivector_given_T(allFiles, ...
    trainList, testList, nmix, tvDim, nWorkers, use_score, enable_fusion);

