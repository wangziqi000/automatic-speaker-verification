%##############################################################
% Sample script to perform short utterance speaker verficiation
% ECE214A: Digital Speech Processing, Winter 2020
%##############################################################
% clear all;
% clc;
%%
% load('featureVQtest.mat')

% Define lists
allFiles = 'allFiles.txt';
trainList = 'train_read_trials.txt';  
testList = 'test_read_trials.txt';

tic
%
% Extract features
featureDict = containers.Map;
fid = fopen(allFiles);
myData = textscan(fid,'%s');
fclose(fid);
myFiles = myData{1};

featureTypeDict = containers.Map;

for cnt = 1:length(myFiles)
    feature_save_name = split(myFiles{cnt}, ".");
    feature_save_name = feature_save_name{1} + ".mat";
    ftr = load(feature_save_name);
    feature = [];
    fn = fieldnames(ftr);
    for i = 1:length(fn)
       if fn{i} == "epoch"
           continue;
       end
       frame_cnt = 2000;
       if length(ftr.(fn{i})) ~= frame_cnt
           continue;
       end
       valid_index = ~isnan(ftr.(fn{i}));
       feature_temp = ftr.(fn{i});
       feature_temp = feature_temp( valid_index );
       if isempty(feature_temp)
           continue;
       end
       
       if ismember(fn{i}, keys(featureTypeDict))
           featureTypeDict(fn{i}) = featureTypeDict(fn{i}) + 1;
       else
           featureTypeDict(fn{i}) = 1;
       end
    end
end    

feature_types =  keys(featureTypeDict);
feature_count = 0;
for cnt = 1:length(feature_types)
    if featureTypeDict(feature_types{cnt}) < length(myFiles)
        feature_types{cnt} = [];
    else
        feature_count = feature_count + 1;
    end
end
valid_features = cell(feature_count, 1);
feature_count = 1;
for cnt = 1:length(feature_types)
    if ~isempty(feature_types{cnt})
        valid_features{feature_count} = feature_types{cnt};
        feature_count = feature_count + 1;
    end
end
feature_count = feature_count - 1;
disp(valid_features);
    
for cnt = 1:length(myFiles)
    [snd,fs] = audioread(myFiles{cnt});
    
    try
        feature_save_name = split(myFiles{cnt}, ".");
        feature_save_name = feature_save_name{1} + ".mat";
        ftr = load(feature_save_name);
        feature = [];
        fn = fieldnames(ftr);
        for i = 1:length(fn)
           if ismember(fn{i}, valid_features)
                feature = [feature mean(feature_temp)];
           end
        end
        featureDict(myFiles{cnt}) = feature;
        if size(feature) ~= feature_count
            disp("Invalid feature number!");
        end
        
    catch
        disp(["No features for the file ", myFiles{cnt}]);
    end
    
    if(mod(cnt,100)==0)
        disp(['Completed ',num2str(cnt),' of ',num2str(length(myFiles)),' files.']);
    end
end
% save('featureVQtest');
old_dim = size(feature, 2);
%%

% Train the classifier
fid = fopen(trainList,'r');
myData = textscan(fid,'%s %s %f');
fclose(fid);
fileList1 = myData{1};
fileList2 = myData{2};
trainLabels = myData{3};
trainFeatures = zeros(length(trainLabels), old_dim);
for cnt = 1:length(trainLabels)
    trainFeatures(cnt, :) = -abs(featureDict(fileList1{cnt})-featureDict(fileList2{cnt}));
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
testFeatures = zeros(length(testLabels), old_dim);
for cnt = 1:length(testLabels)
    testFeatures(cnt, :) = -abs(featureDict(fileList1{cnt})-featureDict(fileList2{cnt}));
end

[~,prediction,~] = predict(Mdl,testFeatures);
testScores = (prediction(:,2)./(prediction(:,1)+1e-15));
[eer,~] = compute_eer(testScores, testLabels);
disp(['The EER is ',num2str(eer),'%.']);

toc
%%