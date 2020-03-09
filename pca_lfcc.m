%##############################################################
% Sample script to perform short utterance speaker verficiation
% ECE214A: Digital Speech Processing, Winter 2020
%##############################################################
% clear all;
% clc;
%%
% Define lists
allFiles = 'allFiles.txt';
trainList = 'train_phone_trials.txt';  
testList = 'test_read_trials.txt';

tic

% Extract features
featureDict = containers.Map;
fid = fopen(allFiles);
myData = textscan(fid,'%s');
fclose(fid);
myFiles = myData{1};
wholeFeatures = zeros(length(myFiles), 450);
for cnt = 1:length(myFiles)
    [snd,fs] = audioread(myFiles{cnt});
    Window_Length = 20;
    NFFT = 512;
    No_Filter = 450;
    try
        [stat,delta,double_delta] = extract_lfcc(snd,fs,Window_Length,NFFT,No_Filter); 
        featureDict(myFiles{cnt}) = mean(stat,1);
        wholeFeatures(cnt,:) = mean(stat,1);
    catch
        disp(["No features for the file ", myFiles{cnt}]);
    end
    
    if(mod(cnt,1)==0)
        disp(['Completed ',num2str(cnt),' of ',num2str(length(myFiles)),' files.']);
    end
end


% PCA dimemsion reduction
[coeff,score,latent] = pca(wholeFeatures);
new_dim = sum(cumsum(latent)./sum(latent)<0.99999)+1;
trans_mat = coeff(:,1:new_dim);

% apply dimension reduction
for cnt = 1:length(myFiles)
    featureDict(myFiles{cnt}) = featureDict(myFiles{cnt})*trans_mat;
end

% save('featureDictMFCC_delta_delta2');
% load('featureDictMFCC.mat');
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