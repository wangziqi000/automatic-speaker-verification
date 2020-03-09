%##############################################################
% Sample script to perform short utterance speaker verficiation
% ECE214A: Digital Speech Processing, Winter 2020
%##############################################################
% clear all;
% clc;
%%
% Define lists
allFiles = 'allFiles.txt';
trainList = 'train_read_trials.txt';  
testList = 'test_read_trials.txt';
% 
tic

%%

% Train the classifier
fid = fopen(trainList,'r');
myData = textscan(fid,'%s %s %f');
fclose(fid);
fileList1 = myData{1};
fileList2 = myData{2};
trainLabels = myData{3};
trainFeatures = zeros(length(trainLabels),1);

snd1 = [];
snd2 = [];

parfor cnt = 1:length(trainLabels)
    snd1 = audioread(fileList1{cnt});
    snd2 = audioread(fileList2{cnt});
    trainFeatures(cnt,:) = vcc_vox_net_siamese(snd1, snd2);
end

Mdl = fitcknn(trainFeatures,trainLabels,'NumNeighbors',15000,'Standardize',1);
fprintf("Training finished")
%%
% Test the classifier
fid = fopen(testList);
myData = textscan(fid,'%s %s %f');
fclose(fid);
fileList1 = myData{1};
fileList2 = myData{2};
testLabels = myData{3};
testFeatures = zeros(length(testLabels),1);

parfor cnt = 1:length(testLabels)
    snd1 = audioread(fileList1{cnt});
    snd2 = audioread(fileList2{cnt});
    testFeatures(cnt,:) = vcc_vox_net_siamese(snd1, snd2);
end

save('featureVGGVox_siamese');

[~,prediction,~] = predict(Mdl,testFeatures);
testScores = (prediction(:,2)./(prediction(:,1)+1e-15));
[eer,~] = compute_eer(testScores, testLabels);
disp(['The EER is ',num2str(eer),'%.']);

toc
%%