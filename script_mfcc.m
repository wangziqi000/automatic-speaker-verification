%##############################################################
% Sample script to perform short utterance speaker verficiation
% ECE214A: Digital Speech Processing, Winter 2020
%##############################################################
% clear all;
% clc;
%%

% load('featureDictMFCC.mat');

% Define lists
allFiles = 'allFiles.txt';
trainList = 'train_read_trials.txt';  
testList = 'test_read_trials.txt';

use_pca = 1;
pca_latent_knob = 0.999;

num_coeffs = 40;
use_delta = 1;
use_delta_delta = 0;

tic

% Extract features
featureDict = containers.Map;
fid = fopen(allFiles);
myData = textscan(fid,'%s');
fclose(fid);
myFiles = myData{1};
for cnt = 1:length(myFiles)
    [snd,fs] = audioread(myFiles{cnt});
    try
        [coeffs,delta,deltaDelta,loc] = mfcc(snd,fs, "NumCoeffs", num_coeffs);
        if use_delta_delta == 1
            featureDict(myFiles{cnt}) = mean([coeffs,delta,deltaDelta]', 2);
        elseif use_delta == 1
            featureDict(myFiles{cnt}) = mean([coeffs,delta]', 2);
        else 
            featureDict(myFiles{cnt}) = mean(coeffs', 2);
        end
    catch
        disp(["No features for the file ", myFiles{cnt}]);
    end
    
    if(mod(cnt,1)==0)
        disp(['Completed ',num2str(cnt),' of ',num2str(length(myFiles)),' files.']);
    end
end

% save('featureDictMFCC');

%% PCA
old_dim = size(featureDict(myFiles{cnt}), 1);
new_dim = old_dim;
if use_pca
    fid = fopen(allFiles,'r');
    myData = textscan(fid,'%s');
    fclose(fid);
    fileList = myData{1};
    wholeFeatures = zeros(length(fileList), old_dim);

    for cnt = 1:length(fileList)
        wholeFeatures(cnt,:) = featureDict(fileList{cnt});
    end

    [coeff,score,latent] = pca(wholeFeatures);
    new_dim = sum(cumsum(latent)./sum(latent) < pca_latent_knob)+1;
    trans_mat = coeff(:,1:new_dim);

    % apply dimension reduction
    for cnt = 1:length(myFiles)
        featureDict(myFiles{cnt}) = transpose(featureDict(myFiles{cnt}))*trans_mat;
    end
end

%%

% Train the classifier
fid = fopen(trainList,'r');
myData = textscan(fid,'%s %s %f');
fclose(fid);
fileList1 = myData{1};
fileList2 = myData{2};
trainLabels = myData{3};
trainFeatures = zeros(length(trainLabels), new_dim);
parfor cnt = 1:length(trainLabels)
    trainFeatures(cnt,:) = -abs(featureDict(fileList1{cnt})-featureDict(fileList2{cnt}));
end

Mdl = fitcknn(trainFeatures,trainLabels,'NumNeighbors',15000,'Standardize',1);

%%
% Test the classifier
fid = fopen(testList);
myData = textscan(fid,'%s %s %f');
fclose(fid);
fileList1 = myData{1};
fileList2 = myData{2};
testLabels = myData{3};
testFeatures = zeros(length(testLabels), new_dim);
parfor cnt = 1:length(testLabels)
    testFeatures(cnt,:) = -abs(featureDict(fileList1{cnt})-featureDict(fileList2{cnt}));
end

[~,prediction,~] = predict(Mdl,testFeatures);
testScores = (prediction(:,2)./(prediction(:,1)+1e-15));
[eer,~] = compute_eer(testScores, testLabels);
disp(['The EER is ',num2str(eer),'%.']);

toc
%%