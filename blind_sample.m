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
blind_list = 'blind_file_list';
blind_trials = 'blind_trials';
eval_prediction = 'vijay_ravi_blind_label.txt'; % change it to your group member's name

tic
%
%% Extract features
featureDict1 = containers.Map;
fid = fopen(allFiles);
myData = textscan(fid,'%s');
fclose(fid);
myFiles = myData{1};
for cnt = 1:length(myFiles)
    [snd,fs] = audioread(myFiles{cnt});
    try
        [F0,lik] = fast_mbsc_fixedWinlen_tracking(snd,fs);
        featureDict1(myFiles{cnt}) = mean(F0(lik>0.45));
    catch
        disp(["No features for the file ", myFiles{cnt}]);
    end
    
    if(mod(cnt,1)==0)
        disp(['Completed ',num2str(cnt),' of ',num2str(length(myFiles)),' files.']);
    end
end
save('featureDict1');

%% Extract Features for blind data
featureDict2 = containers.Map;
fid = fopen(blind_list);
myData2 = textscan(fid,'%s');
fclose(fid);
myFiles2 = myData2{1};
for cnt = 1:length(myFiles2)
    [snd,fs] = audioread(myFiles2{cnt});
    try
        [F0,lik] = fast_mbsc_fixedWinlen_tracking(snd,fs);
        featureDict2(myFiles2{cnt}) = mean(F0(lik>0.45));
    catch
        disp(["No features for the file ", myFiles2{cnt}]);
    end
    
    if(mod(cnt,1)==0)
        disp(['Completed ',num2str(cnt),' of ',num2str(length(myFiles2)),' files.']);
    end
end
save('featureDict2');

%% Train the classifier
fid = fopen(trainList,'r');
myData = textscan(fid,'%s %s %f');
fclose(fid);
fileList1 = myData{1};
fileList2 = myData{2};
trainLabels = myData{3};
trainFeatures = zeros(length(trainLabels),1);
for cnt = 1:length(trainLabels)
    trainFeatures(cnt) = -abs(featureDict1(fileList1{cnt})-featureDict1(fileList2{cnt}));
end

Mdl = fitcknn(trainFeatures,trainLabels,'NumNeighbors',15000,'Standardize',1);

%%
% Test the classifierground_truth = 'blind_labels';
% fid = fopen(testList);
% myData = textscan(fid,'%f');
% fclose(fid);
testLabels = myData{1};
fid = fopen(testList);
myData = textscan(fid,'%s %s %f');
fclose(fid);
fileList1 = myData{1};
fileList2 = myData{2};
testLabels = myData{3};
testFeatures = zeros(length(testLabels),1);
for cnt = 1:length(testLabels)
    testFeatures(cnt) = -abs(featureDict1(fileList1{cnt})-featureDict1(fileList2{cnt}));
end

[~,prediction,~] = predict(Mdl,testFeatures);
testScores = (prediction(:,2)./(prediction(:,1)+1e-15));
[eer,~] = compute_eer(testScores, testLabels);
disp(['The EER is ',num2str(eer),'%.']);

toc


%% Blind Evaluation

% Test the classifier on eval data
fid = fopen(blind_trials);
myData = textscan(fid,'%s %s');
fclose(fid);
fileList1 = myData{1};
fileList2 = myData{2};
% testLabels = myData{3};
testFeatures2 = zeros(length(fileList1),1);
for cnt = 1:length(fileList1)
    testFeatures2(cnt) = -abs(featureDict2(fileList1{cnt})-featureDict2(fileList2{cnt}));
end

[~,prediction,~] = predict(Mdl,testFeatures2);
testScores = (prediction(:,2)./(prediction(:,1)+1e-15));
fid=fopen(eval_prediction,'w');
fprintf(fid,'%f\n',testScores);
fclose(fid);

filename = 'experiment.mat';
save(filename)

%% EER scoring - FOR TA's only. 

% ground_truth = 'blind_labels';
% fid = fopen(ground_truth);
% myData = textscan(fid,'%f');
% fclose(fid);
% testLabels = myData{1};
% 
% fid = fopen(eval_prediction);
% myData = textscan(fid,'%f');
% fclose(fid);
% testScores = myData{1};
% 
% [eer,~] = compute_eer(testScores, testLabels);
% disp(['The EER is for eval',num2str(eer),'%.']);


