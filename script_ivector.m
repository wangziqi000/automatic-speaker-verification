%##############################################################
% Sample script to perform short utterance speaker verficiation
% ECE214A: Digital Speech Processing, Winter 2020
%##############################################################
%%
clear all;
clc;

%%
tic

% Define lists
allFiles = 'allFiles.txt';
fid = fopen(allFiles);
myData = textscan(fid,'%s');
fclose(fid);
myFiles = myData{1};

nCoeffs = 40;               % NumCoeffs for MFCC
nMixtures = 512;            % Mixture num for GMM (must be power of 2)
final_niter = 10;           % max iter num for GMM
tvDim = 100;                % Dimensionality of total variability, can adjust
niter = 5;                  % iter num for total variability 
ldaDim = min(tvDim, 69);    % = final feature number, 69 = nSpeakers - 1
niterlda = 10;              % iter num for PLDA
[featureDict, ubm, T] = func_ivector(myFiles, nCoeffs, nMixtures, final_niter, tvDim, niter, ldaDim, niterlda);

%%
tic
trainListList = {'train_read_trials.txt', 'train_phone_trials.txt'};
testListList = {'test_read_trials.txt', 'test_phone_trials.txt', 'test_mismatch_trials.txt'};
for eachTrain = 1:length(trainListList)
    trainList = trainListList{eachTrain};
    Mdl = func_train(trainList, featureDict);
    for eachTest = 1:length(testListList)
        testList = testListList{eachTest};
        disp([trainList, ' ', testList]);
        func_test(testList, featureDict, Mdl);
    end
end

toc
%%