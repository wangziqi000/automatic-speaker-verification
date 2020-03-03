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
%     B = 96;
%     fmax = fs/2;
%     fmin = fmax/2^9;
%     d = 16;
%     cf = 19;
%     ZsdD = 'ZsdD';
%     try
%         [CQcc, LogP_absCQT, TimeVec, FreqVec, Ures_LogP_absCQT, Ures_FreqVec] = ...
%     cqcc(snd, fs, B, fmax, fmin, d, cf, ZsdD);
%         featureDict(myFiles{cnt}) = mean(CQcc,2);
%     catch
%         disp(["No features for the file ", myFiles{cnt}]);
%     end
%     
%     if(mod(cnt,1)==0)
%         disp(['Completed ',num2str(cnt),' of ',num2str(length(myFiles)),' files.']);
%     end
% end
% save('featureDictCQCC');
load('featureDictCQCC.mat');
%%

% Train the classifier
fid = fopen(trainList,'r');
myData = textscan(fid,'%s %s %f');
fclose(fid);
fileList1 = myData{1};
fileList2 = myData{2};
trainLabels = myData{3};
trainFeatures = zeros(length(trainLabels),60);
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
testFeatures = zeros(length(testLabels),60);
parfor cnt = 1:length(testLabels)
    testFeatures(cnt,:) = -abs(featureDict(fileList1{cnt})-featureDict(fileList2{cnt}));
end

[~,prediction,~] = predict(Mdl,testFeatures);
testScores = (prediction(:,2)./(prediction(:,1)+1e-15));
[eer,~] = compute_eer(testScores, testLabels);
disp(['The EER is ',num2str(eer),'%.']);

toc
%%