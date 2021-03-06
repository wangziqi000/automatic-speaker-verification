%##############################################################
% This script tries to extract the x1(128 dimension, the last) layer of
% the ResNet model as utterance feature vector
%##############################################################
% clear all;
% clc;
%%
% Define lists
load('featureVGGVox_x1.mat')


allFiles = 'allFiles.txt';
trainList = 'train_read_trials.txt';  
testList = 'test_mismatch_trials.txt';
% for read-mismatch, the EER is 23%.
tic

% % Extract features
% featureDict = containers.Map;
% fid = fopen(allFiles);
% myData = textscan(fid,'%s');
% fclose(fid);
% myFiles = myData{1};
% for cnt = 1:length(myFiles)
%     [snd,fs] = audioread(myFiles{cnt});
%     try
%         feat = vcc_vox_net(snd);
%         featureDict(myFiles{cnt}) = feat;
%     catch
%         disp(["No features for the file ", myFiles{cnt}]);
%     end
%     
%     if(mod(cnt,1)==0)
%         disp(['Completed ',num2str(cnt),' of ',num2str(length(myFiles)),' files.']);
%     end
% end
% 
% save('featureVGGVox_x1');

%%

% Train the classifier
fid = fopen(trainList,'r');
myData = textscan(fid,'%s %s %f');
fclose(fid);
fileList1 = myData{1};
fileList2 = myData{2};
trainLabels = myData{3};
trainFeatures = zeros(length(trainLabels), 128);
parfor cnt = 1:length(trainLabels)
    trainFeatures(cnt,:) = -abs(featureDict(fileList1{cnt})-featureDict(fileList2{cnt}));
end

Mdl = fitcknn(trainFeatures,trainLabels,'NumNeighbors',15000,'Standardize',1);

%%
% Test the classifier
fid = fopen(testList, "r");
myData = textscan(fid,'%s %s %f');
fclose(fid);
fileList1 = myData{1};
fileList2 = myData{2};
testLabels = myData{3};
testFeatures = zeros(length(testLabels), 128);
parfor cnt = 1:length(testLabels)
    testFeatures(cnt,:) = -abs(featureDict(fileList1{cnt})-featureDict(fileList2{cnt}));
end

[~,prediction,~] = predict(Mdl,testFeatures);
testScores = (prediction(:,2)./(prediction(:,1)+1e-15));
[eer,~] = compute_eer(testScores, testLabels);
disp(['The EER is ',num2str(eer),'%.']);

toc
%%