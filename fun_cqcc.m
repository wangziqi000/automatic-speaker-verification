function[trainEER, testScores, testLabels, eer] =  fun_cqcc(allFiles, ...
    trainList, testList, use_pca, pca_latent_knob, num_coeffs, ...
    use_delta, use_delta_delta, enable_fusion) 

% load('featureDictCQCC.mat');

% Extract features
featureDict = containers.Map;
fid = fopen(allFiles);
myData = textscan(fid,'%s');
fclose(fid);
myFiles = myData{1};
for cnt = 1:length(myFiles)
    [snd,fs] = audioread(myFiles{cnt});
    B = 96;
    fmax = fs/2;
    fmin = fmax/2^9;
    d = 16;
    cf = num_coeffs;
    
    if use_delta_delta == 1
        ZsdD = 'ZsdD';
    elseif use_delta == 1
        ZsdD = 'Zsd';
    else 
        ZsdD = 'Zs';
    end

    try
        [CQcc, LogP_absCQT, TimeVec, FreqVec, Ures_LogP_absCQT, Ures_FreqVec] = ...
    cqcc(snd, fs, B, fmax, fmin, d, cf, ZsdD);
        featureDict(myFiles{cnt}) = mean(CQcc,2);
    catch
        disp(["No features for the file ", myFiles{cnt}]);
    end
    
    if(mod(cnt,100)==0)
        disp(['Completed ',num2str(cnt),' of ',num2str(length(myFiles)),' files.']);
    end
end

% save('featureDictCQCC');

%% PCA
old_dim = size(featureDict(myFiles{cnt}), 1);
new_dim = old_dim;
if use_pca
    fid = fopen(allFiles,'r');
    myData = textscan(fid,'%s');
    fclose(fid);
    fileList = myData{1};
    wholeFeatures = zeros(length(fileList), old_dim);

    for cnt = 1:length(fileList)
        wholeFeatures(cnt,:) = featureDict(fileList{cnt});
    end

    [coeff,score,latent] = pca(wholeFeatures);
    new_dim = sum(cumsum(latent)./sum(latent) < pca_latent_knob)+1;
    trans_mat = coeff(:,1:new_dim);

    % apply dimension reduction
    for cnt = 1:length(myFiles)
        featureDict(myFiles{cnt}) = transpose(featureDict(myFiles{cnt}))*trans_mat;
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
trainFeatures = zeros(length(trainLabels), new_dim);
parfor cnt = 1:length(trainLabels)
    trainFeatures(cnt,:) = -abs(featureDict(fileList1{cnt})-featureDict(fileList2{cnt}));
end

Mdl = fitcknn(trainFeatures,trainLabels,'NumNeighbors',15000,'Standardize',1);

%% Train EER - "Purity of Classification"
if enable_fusion
    [~,prediction,~] = predict(Mdl,trainFeatures(1:10000,:));
    trainScores = (prediction(:,2)./(prediction(:,1)+1e-15));
    [eer,~] = compute_eer(trainScores, trainLabels(1:10000));
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
testFeatures = zeros(length(testLabels), new_dim);
parfor cnt = 1:length(testLabels)
    testFeatures(cnt,:) = -abs(featureDict(fileList1{cnt})-featureDict(fileList2{cnt}));
end

[~,prediction,~] = predict(Mdl,testFeatures);
testScores = (prediction(:,2)./(prediction(:,1)+1e-15));
[eer,~] = compute_eer(testScores, testLabels);
disp(['The EER is ',num2str(eer),'%.']);
