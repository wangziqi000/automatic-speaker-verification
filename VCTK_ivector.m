%%
clear all;clc;

%%
allfile_dir = 'VCTK-Corpus\wav48\';
allSpeaker = dir([allfile_dir, 'p*']);
fs = 22050;
nFrame = 44100;

%%
% mfcc on universal speaker dataset VCTK
tic
for speakerIdx=1:length(allSpeaker)
    speaker_dir = allSpeaker(speakerIdx).name;
    allWav = dir([allfile_dir, speaker_dir, '\*.wav']);
    channelCnt = 1;
    for waveIdx=1:length(allWav)
        wavFile = [allfile_dir, speaker_dir, '\', allWav(waveIdx).name];
        [snd, ~] = audioread(wavFile);
        snd = resample(snd, 147, 320);  % resample 48000 to 22050 (22050/48000=147/320)
        nChannel = floor(length(snd)/nFrame);
        for channelIdx=1:nChannel
            [mfccsdata{speakerIdx, channelCnt}, ~] = mfcc(snd((channelIdx-1)*nFrame+1:channelIdx*nFrame), fs);
            channelCnt = channelCnt + 1;
        end
    end
    disp([num2str(speakerIdx), ': ', speaker_dir, ' done.']);
    toc
end

%%
% nChannel to same size
channelMax = size(mfccsdata);
channelMax = channelMax(2);
channelPerSpeaker = ones(1, length(allSpeaker)) * channelMax;
for i=1:length(allSpeaker)
    for j=1:channelMax
        if isempty(mfccsdata{i,j}) == 1
            channelPerSpeaker(i)=j-1;
            break
        end
    end
end

%%
% Find the proper nChannel num with max nSpeaker * nChannel
for i=209:866
    channelScore(i) = i * sum(channelPerSpeaker>=i);
end
[~, nChannel] = max(channelScore);              % nChannel = 420;
nSpeaker = sum(channelPerSpeaker >= nChannel);  % nSpeaker = 103;

%%
% Selected useful mfccsdata from universal dataset
speakerInUse = find(channelPerSpeaker >= nChannel);
trainSpeakerData = cell(nSpeaker, nChannel);
for i=1:nSpeaker
    for j=1:nChannel
        trainSpeakerData{i, j} = mfccsdata{speakerInUse(i), j};
    end
end

%%
% Train ubm
rng('default');
nMixture = 256;
final_niter = 10;
ds_factor = 1;
nWorkers = 8;
tic
ubm = gmm_em(trainSpeakerData(:), nMixture, final_niter, ds_factor, nWorkers);
toc

% save('VCTK_nmix512_ubm.mat', 'ubm');

%%
% Works before this part: to extract trainSpeakerData, train ucm
% Works after this part: to find ivector for small dataset

% dimension information
nSpeaker = 103;
nChannel = 420;
nSpeakers = 70;
nChannels = 10;

nWorkers = 8;   % Num for parpool
nmix = 256;

load('VCTK_trainSpeakerData');              % trainSpeakerData
load(['VCTK_nmix',num2str(nmix),'_ubm']);   % ubm

%%
% Extract features
allFiles = 'allFiles.txt';
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
% Step 2.1: Calculate the statistics needed for the i-Vector model (on universal data)
stats = cell(nSpeaker, nChannel);
for i=1:nSpeaker
    for j=1:nChannel
        [N, F] = compute_bw_stats(trainSpeakerData{i, j}, ubm);
        stats{i, j} = [N; F];
    end
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
% Step 2.2: Learn the total variability subspace from all the speaker data
tvDim = 200;    % Dimension of total variability, can adjust
%%
niter = 5;
T = train_tv_space(stats(:), ubm, tvDim, niter, nWorkers);
disp('T trained');
% save(['VCTK_nmix256_tvDim', num2str(tvDim), '_T'],'T');
%%
load(['VCTK_nmix256_tvDim',num2str(tvDim),'_T']);

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
        ivScores = score_gplda_trials(pLDA, modelIVs, finalCurrIVs);
%         featureDict(myFiles{cnt}) = finalCurrIVs(:);
        featureDict(myFiles{cnt}) = ivScores(:);
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

%%
% Score, for test
% ivScores = score_gplda_trials(pLDA, modelIVs, finalCurrIVs);
% [~, idx] = max(ivScores);

%%
% Train classifier and test EER, given featureDict
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