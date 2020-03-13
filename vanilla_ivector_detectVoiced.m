%##############################################################
% Sample script to perform short utterance speaker verficiation
% ECE214A: Digital Speech Processing, Winter 2020
%##############################################################
%%
clear all;
clc;
%%
% Define lists
% allFiles = 'blind_evaluation\blind_file_list';
trainList = 'train_read_trials.txt';
% trainList = 'train_phone_trials.txt';
% testList = 'test_read_trials.txt';
% testList = 'test_phone_trials.txt';
testList = 'test_mismatch_trials.txt';
% testList = 'blind_evaluation\blind_trials_labelCheat';

%%
tic

%%
% Extract features
% allFiles = 'allFiles.txt';
allFiles = 'blind_evaluation\blind_file_list';
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

nSpeakers = length(keys(speakerData));  % nSpeakers = 70
nChannels = 10;
disp('speakerData done');

%%
% mfcc for i-vector
nDims = 13;     % Feature num = 14
nMixtures = 256; % Mixture num for GMM (must be power of 2)
nChannels = 10; % Channel num per each speaker, 10 wav for each speaker
nFrames = 1000; % Frame num(/length?) per speaker
nWorkers = 8;   % Num for parpool
rng('default'); % Init the random seeds

mfccsdata = cell(nSpeakers, nChannels);
speakerNameList = keys(speakerData);
for i=1:nSpeakers
    speakerName = char(speakerNameList(i));
    channelList = speakerData(speakerName);
    for j=1:nChannels
        index = channelList(j);
        
        % Remove silence
        [seg, fs] = detectVoiced(myFiles{index});
        snd = [];
        for segidx=1:length(seg)
            snd = [snd;seg{segidx}];
        end
        if length(snd)<44100
            snd_new = [];
            while length(snd_new)<44100
                snd_new = [snd_new;snd];
            end
            snd=snd_new(1:44100);
        end
        
        [mfccsdata{i, j},~] = mfcc(snd, fs);
        speakerID(i, j) = i;
    end
    disp(['MFCC for Speaker ', speakerName, ' done']);
end

%%
% Begin i-vector
rng('default');     % Reset random seeds
% Step 1: Create the ubm model from all the training speaker data
nmix = nMixtures;
final_niter = 10;   % max num of iteration
ds_factor = 1;      % downsampling rate
ubm = gmm_em(mfccsdata(:), nmix, final_niter, ds_factor, nWorkers);     % ubm: a struct, has mu(means), sigma(covariances), w(weights)
disp('ubm done');

%%
% Step 2.1: Calculate the statistics needed for the i-Vector model
stats = cell(nSpeakers, nChannels);
for i=1:nSpeakers
    for j=1:nChannels
        [N, F] = compute_bw_stats(mfccsdata{i, j}, ubm);
        stats{i, j} = [N; F];
    end
end

%%
% Step 2.2: Learn the total variability subspace from all the speaker data
tvDim = 200;    % Dimension of total variability, can adjust
niter = 5;
T = train_tv_space(stats(:), ubm, tvDim, niter, nWorkers);
disp('T trained');

%%
% Step 2.3: Compute the i-Vector for each speaker and channel
% The size of the resulting i-Vector: tvDim x nSpeakers x nChannels

devIVs = zeros(tvDim, nSpeakers, nChannels);
for i=1:nSpeakers
    for j=1:nChannels
        devIVs(:, i, j) = extract_ivector(stats{i, j}, ubm, T);
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
% Apply speaker model on all files
featureDict = containers.Map;
for cnt = 1:length(myFiles)
    % Remove silence
    [seg, fs] = detectVoiced(myFiles{cnt});
    snd = [];
    for segidx=1:length(seg)
        snd = [snd;seg{segidx}];
    end
    if length(snd)<44100
        snd_new = [];
        while length(snd_new)<44100
            snd_new = [snd_new;snd];
        end
        snd=snd_new(1:44100);
    end

    try
        [coeffs,~] = mfcc(snd, fs);

        currIVs = zeros(tvDim, 1, 1);
        [N, F] = compute_bw_stats(coeffs, ubm);
        currIVs(:,1,1) = extract_ivector([N; F], ubm, T);
        currIVbySpeaker = reshape(permute(currIVs, [1 3 2]), tvDim, 1);
        finalCurrIVs = V(:, 1:ldaDim)' * currIVbySpeaker;
%         ivScores = score_gplda_trials(pLDA, modelIVs, finalCurrIVs);
        featureDict(myFiles{cnt}) = finalCurrIVs(:);
%         featureDict(myFiles{cnt}) = ivScores(:);
%         wholeFeatures(cnt,:) = currIVs(:);
    catch
        disp(["No features for the file ", myFiles{cnt}]);
    end
    
    if(mod(cnt,1)==0)
        disp(['Completed ',num2str(cnt),' of ',num2str(length(myFiles)),' files.']);
    end
end

% PCA
% [coeff,score,latent] = pca(wholeFeatures);
% new_dim = sum(cumsum(latent)./sum(latent)<0.95)+1;
% trans_mat = coeff(:,1:new_dim);
% 
% for cnt=1:length(myFiles)
%     featureDict(myFiles{cnt}) = featureDict(myFiles{cnt})'*trans_mat;
% end

% save('featureDictMFCC_ivector');
% load('featureDictMFCC_ivector.mat');

%%
% Score, for test
% ivScores = score_gplda_trials(pLDA, modelIVs, finalCurrIVs);
% [~, idx] = max(ivScores);

%%
% Train the classifier
trainList = 'train_read_trials.txt';
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
disp('classifier finished.');

%%
% Test the classifier
% testList = 'test_mismatch_trials.txt';
testList = 'blind_evaluation\blind_trials_labelCheat';
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

toc
%%