%##############################################################
% This script tries to extract the x0(2048 dimension) layer of the ResNet 
% model as utterance feature vector, and use the euclidean distance between
% pairs of feature vectors as the final 1-dim feature for KNN 

% No blind test involved
% You can ignore this script
%##############################################################
% clear all;
% clc;
%%
load('featureVGGVox_x0.mat')

% Define lists
allFiles = 'allFiles.txt';
trainList = 'train_read_trials.txt';  
testList = 'test_mismatch_trials.txt';
% for read-mismatch, the EER is 28%.
tic
%
% % Extract features
% featureDict = containers.Map;
% fid = fopen(allFiles);
% myData = textscan(fid,'%s');
% fclose(fid);
% myFiles = myData{1};
% for cnt = 1:length(myFiles)
%     [snd,fs] = audioread(myFiles{cnt});
%     try
%         feat = vcc_vox_net_x0(snd);
%         featureDict(myFiles{cnt}) = feat;
%     catch
%         disp(["No features for the file ", myFiles{cnt}]);
%     end
%     
%     if(mod(cnt,1)==0)
%         disp(['Completed ',num2str(cnt),' of ',num2str(length(myFiles)),' files.']);
%     end
% end
% save('featureVGGVox_x0');


%% pca
% fid = fopen(allFiles,'r');
% myData = textscan(fid,'%s');
% fclose(fid);
% fileList = myData{1};
% wholeFeatures = zeros(length(fileList),2048);
% 
% for cnt = 1:length(fileList)
%     wholeFeatures(cnt,:) = featureDict(fileList{cnt});
% end
% 
% [coeff,score,latent] = pca(wholeFeatures);
% new_dim = sum(cumsum(latent)./sum(latent)<0.99999)+1;
% trans_mat = coeff(:,1:new_dim);
% 
% % apply dimension reduction
% for cnt = 1:length(myFiles)
%     featureDict(myFiles{cnt}) = transpose(featureDict(myFiles{cnt}))*trans_mat;
% end

new_dim = size(feat, 1);
%%

% Train the classifier
fid = fopen(trainList,'r');
myData = textscan(fid,'%s %s %f');
fclose(fid);
fileList1 = myData{1};
fileList2 = myData{2};
trainLabels = myData{3};
trainFeatures = single(zeros(length(trainLabels), 1));
for cnt = 1:length(trainLabels)
    trainFeatures(cnt,:) = -pdist([featureDict(fileList1{cnt}),featureDict(fileList2{cnt})]', 'euclidean');
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
testFeatures = zeros(length(testLabels), 1);
parfor cnt = 1:length(testLabels)
    testFeatures(cnt,:) = -pdist([featureDict(fileList1{cnt}),featureDict(fileList2{cnt})]', 'euclidean');
end

[~,prediction,~] = predict(Mdl,testFeatures);
testScores = (prediction(:,2)./(prediction(:,1)+1e-15));
[eer,~] = compute_eer(testScores, testLabels);
disp(['The EER is ',num2str(eer),'%.']);

toc
%% The scirpt below is much slower

% %##############################################################
% % Sample script to perform short utterance speaker verficiation
% % ECE214A: Digital Speech Processing, Winter 2020
% %##############################################################
% % clear all;
% % clc;
% %%
% % Define lists
% allFiles = 'allFiles.txt';
% trainList = 'train_phone_trials.txt';  
% testList = 'test_read_trials.txt';
% % 
% tic
% 
% %%
% 
% % Train the classifier
% fid = fopen(trainList,'r');
% myData = textscan(fid,'%s %s %f');
% fclose(fid);
% fileList1 = myData{1};
% fileList2 = myData{2};
% trainLabels = myData{3};
% trainFeatures = zeros(length(trainLabels),1);
% 
% snd1 = [];
% snd2 = [];
% 
% for cnt = 1:length(trainLabels)
%     snd1 = audioread(fileList1{cnt});
%     snd2 = audioread(fileList2{cnt});
%     trainFeatures(cnt,:) = vcc_vox_net_siamese(snd1, snd2);
%     fprintf(".");
% end
% 
% Mdl = fitcknn(trainFeatures,trainLabels,'NumNeighbors',15000,'Standardize',1);
% fprintf("Training finished")
% %%
% % Test the classifier
% fid = fopen(testList);
% myData = textscan(fid,'%s %s %f');
% fclose(fid);
% fileList1 = myData{1};
% fileList2 = myData{2};
% testLabels = myData{3};
% testFeatures = zeros(length(testLabels),1);
% 
% parfor cnt = 1:length(testLabels)
%     snd1 = audioread(fileList1{cnt});
%     snd2 = audioread(fileList2{cnt});
%     testFeatures(cnt,:) = vcc_vox_net_siamese(snd1, snd2);
% end
% 
% save('featureVGGVox_siamese');
% 
% [~,prediction,~] = predict(Mdl,testFeatures);
% testScores = (prediction(:,2)./(prediction(:,1)+1e-15));
% [eer,~] = compute_eer(testScores, testLabels);
% disp(['The EER is ',num2str(eer),'%.']);
% 
% toc
% %%