%##############################################################
% This script shows the effort to use the concatenation of mfcc and VQual
% as the feature of each utterance.
%##############################################################
% clear all;
% clc;
%%
% Define lists
% load('featureDictConcat.mat');
allFiles = 'allFiles.txt';
trainList = 'train_read_trials.txt';  
testList = 'test_read_trials.txt';

tic
%
% Extract features
featureDict = containers.Map;
fid = fopen(allFiles);
myData = textscan(fid,'%s');
fclose(fid);
myFiles = myData{1};
wholeFeatures = zeros(length(myFiles), 43);
for cnt = 1:length(myFiles)
    [snd,fs] = audioread(myFiles{cnt});
%     if(fs~=8000)
%         ytmp = resample(snd,8000,fs);
%         snd = ytmp;
%         fs = 8000;
%         clear ytmp;
%     end
    B = 96;
    fmax = fs/2;
    fmin = fmax/2^9;
    d = 16;
    cf = 19;
    ZsdD = 'ZsdD';
    try
        lpcs = lpc(snd,8);
        [coeffs,delta,deltaDelta,loc] = mfcc(snd,fs, 'NumCoeffs', 40);
        wholeFeatures(cnt, 42:43)= lpcs(3:4);
        wholeFeatures(cnt, 1:41)= mean(coeffs,1);
        featureDict(myFiles{cnt}) = wholeFeatures(cnt,:);
    catch
        disp(["No features for the file ", myFiles{cnt}]);
        
    end
    
    if(mod(cnt,1)==0)
        disp(['Completed ',num2str(cnt),' of ',num2str(length(myFiles)),' files.']);
    end
end

wholeFeatures = [wholeFeatures(:, 1:30), wholeFeatures(:, 42:43)];
for cnt = 1:length(myFiles)
    featureDict(myFiles{cnt}) = wholeFeatures(cnt,:);
end

% PCA dimemsion reduction
[coeff,score,latent] = pca(wholeFeatures);
new_dim = sum(cumsum(latent)./sum(latent)<0.9999)+1;
trans_mat = coeff(:,1:new_dim);

% apply dimension reduction
for cnt = 1:length(myFiles)
    featureDict(myFiles{cnt}) = featureDict(myFiles{cnt})*trans_mat;
end

% save('featureDictConcat');

%%

% Train the classifier
fid = fopen(trainList,'r');
myData = textscan(fid,'%s %s %f');
fclose(fid);
fileList1 = myData{1};
fileList2 = myData{2};
trainLabels = myData{3};
trainFeatures = zeros(length(trainLabels),new_dim);
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
testFeatures = zeros(length(testLabels),new_dim);
parfor cnt = 1:length(testLabels)
    testFeatures(cnt,:) = -abs(featureDict(fileList1{cnt})-featureDict(fileList2{cnt}));
end

[~,prediction,~] = predict(Mdl,testFeatures);
testScores = (prediction(:,2)./(prediction(:,1)+1e-15));
[eer,~] = compute_eer(testScores, testLabels);
disp(['The EER is ',num2str(eer),'%.']);

toc
%%