function T = VCTK_ivector_find_T(nmix, tvDim, nWorkers)
%%
clear all;clc;

%%
% dimension information
nSpeaker = 103;
nChannel = 420;

% nWorkers = 8;   % Num for parpool
% nmix = 256;     % nmix = 256, 512, 1024

load('VCTK_trainSpeakerData');              % trainSpeakerData: mfcc for universal data
load(['VCTK_nmix',num2str(nmix),'_ubm']);   % ubm

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
% Step 2.2: Learn the total variability subspace from all the speaker data
% tvDim = 200;    % Dimension of total variability, can adjust
niter = 5;
T = train_tv_space(stats(:), ubm, tvDim, niter, nWorkers);
disp('T trained');

save(['VCTK_nmix',num2str(nmix),'_tvDim', num2str(tvDim), '_T'],'T');

end
%%
