function[trainEER, testScores, testLabels, eer] =  fun_nn(allFiles, ...
    trainList, testList, use_pca, pca_latent_knob, enable_fusion) 

% This function is used to generated test score and calculate eer for local
% test data using neural network based features

featureName = 'featureVGGVox_x1.mat';
ok = exist(featureName, 'file');

if ok
    load(featureName, "featureDict", "myFiles", "cnt");
else
    % Extract features
    featureDict = containers.Map;
    fid = fopen(allFiles);
    myData = textscan(fid,'%s');
    fclose(fid);
    myFiles = myData{1};
    for cnt = 1:length(myFiles)
        [snd, fs] = audioread(myFiles{cnt});
        try
            feat = vcc_vox_net(snd);
            featureDict(myFiles{cnt}) = feat;
        catch
            disp(["No features for the file ", myFiles{cnt}]);
        end

        if(mod(cnt,100)==0)
            disp(['Completed ',num2str(cnt),' of ',num2str(length(myFiles)),' files.']);
        end
    end
    save('featureVGGVox_x1');
end

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


