function [trainEER, testScores, testLabels, eer] = VCTK_ivector_given_T(allFiles, trainList, testList, nmix, tvDim, nWorkers, use_score, enable_fusion)
%%
clear all;clc;

%%
% dimension information
nSpeaker = 103;
nChannel = 420;
nSpeakers = 70;
nChannels = 10;

% nWorkers = 8;   % Num for parpool
% nmix = 256;

% load('VCTK_trainSpeakerData');   % trainSpeakerData
load(['VCTK_nmix',num2str(nmix),'_ubm']);  % ubm
load(['VCTK_nmix',num2str(nmix),'_tvDim',num2str(tvDim),'_T']);   % T

%%
% Extract features
% allFiles = 'allFiles.txt';
fid = fopen(allFiles);
myData = textscan(fid,'%s');
fclose(fid);
myFiles = myData{1};

%%
% Label speaker ID
speakerData = containers.Map;
for cnt = 1:length(myFiles)
    title = split(myFiles{cnt},'/');
    title = title{3}(1:3);
    try
        speakerData(title) = [speakerData(title), cnt];
    catch
        speakerData(title) = [cnt];
    end
end
disp('speakerData done');

%%
% mfcc for i-vector (on small data)
nDims = 13;     % Feature num = 14
rng('default'); % Init the random seeds

mfccsdata = cell(nSpeakers, nChannels);
speakerNameList = keys(speakerData);
for i=1:nSpeakers
    speakerName = char(speakerNameList(i));
    channelList = speakerData(speakerName);
    for j=1:nChannels
        index = channelList(j);
        [snd, fs] = audioread(myFiles{index});
        [mfccsdata{i, j},~] = mfcc(snd, fs);
        speakerID(i, j) = i;
    end
    disp(['MFCC for Speaker ', speakerName, ' done']);
end

%%
% Step 2.1 again: Calculate the statistics needed for the i-Vector model (on small data)
stats_s = cell(nSpeakers, nChannels);
for i=1:nSpeakers
    for j=1:nChannels
        [N, F] = compute_bw_stats(mfccsdata{i, j}, ubm);
        stats_s{i, j} = [N; F];
    end
end

%%
% Step 2.3: Compute the i-Vector for each speaker and channel
% The size of the resulting i-Vector: tvDim x nSpeakers x nChannels

devIVs = zeros(tvDim, nSpeakers, nChannels);
for i=1:nSpeakers
    for j=1:nChannels
        devIVs(:, i, j) = extract_ivector(stats_s{i, j}, ubm, T);
    end
end
disp('i-Vector done');

%%
% Step 3.1: Do LDA on the i-Vector to find the dimensions that matter.
ldaDim = min(tvDim, nSpeakers-1);   % =final feature number
devIVbySpeaker = reshape(devIVs, tvDim, nSpeakers * nChannels);
[V, D] = lda(devIVbySpeaker, speakerID(:));
finalDevIVs = V(:, 1:ldaDim)' * devIVbySpeaker;
disp('LDA done');

%%
% Step 3.2: Train a Gaussian PLDA model with development i-Vectors
nphi = ldaDim;  % should be <= ldaDim
niter = 10;
pLDA = gplda_em(finalDevIVs, speakerID(:), nphi, niter);
disp('PLDA done');

%%
% Step 4.1: Build speaker model using the channel and LDA model
averageIVs = mean(devIVs, 3);   % Average IVs across channels for each speaker and feature
modelIVs = V(:, 1:ldaDim)' * averageIVs;
disp('Speaker model done');

%%
% Apply speaker model on allFiles
featureDict = containers.Map;
for cnt = 1:length(myFiles)
    [snd,fs] = audioread(myFiles{cnt});
    try
        [coeffs,~] = mfcc(snd,fs);

        currIVs = zeros(tvDim, 1, 1);
        [N, F] = compute_bw_stats(coeffs, ubm);
        currIVs(:,1,1) = extract_ivector([N; F], ubm, T);
        currIVbySpeaker = reshape(permute(currIVs, [1 3 2]), tvDim, 1);
        finalCurrIVs = V(:, 1:ldaDim)' * currIVbySpeaker;

        if use_score
            ivScores = score_gplda_trials(pLDA, modelIVs, finalCurrIVs);
            featureDict(myFiles{cnt}) = ivScores(:);
        else
            featureDict(myFiles{cnt}) = finalCurrIVs(:);
        end

    catch
        disp(["No features for the file ", myFiles{cnt}]);
    end
    
    if(mod(cnt,1)==0)
        disp(['Completed ',num2str(cnt),' of ',num2str(length(myFiles)),' files.']);
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
trainFeatures = zeros(length(trainLabels), length(featureDict(fileList1{1})));
parfor cnt = 1:length(trainLabels)
    trainFeatures(cnt,:) = -abs(featureDict(fileList1{cnt})-featureDict(fileList2{cnt}));
end

Mdl = fitcknn(trainFeatures,trainLabels,'NumNeighbors',15000,'Standardize',1);

%%
% Train EER
if enable_fusion
    [~,prediction,~] = predict(Mdl,trainFeatures(1:10000,:));
    testScores = (prediction(:,2)./(prediction(:,1)+1e-15));
    [eer,~] = compute_eer(testScores, trainLabels(1:10000));
    trainEER = eer;
else
    trainEER = NaN;
end

%%
% Test the classifier
fid = fopen(testList);
myData = textscan(fid,'%s %s %f');
fclose(fid);
fileList1 = myData{1};
fileList2 = myData{2};
testLabels = myData{3};
testFeatures = zeros(length(testLabels), length(featureDict(fileList1{1})));
parfor cnt = 1:length(testLabels)
    testFeatures(cnt,:) = -abs(featureDict(fileList1{cnt})-featureDict(fileList2{cnt}));
end

[~,prediction,~] = predict(Mdl,testFeatures);
testScores = (prediction(:,2)./(prediction(:,1)+1e-15));
[eer,~] = compute_eer(testScores, testLabels);
disp(['The EER is ',num2str(eer),'%.']);

end