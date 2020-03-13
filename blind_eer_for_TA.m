%% EER scoring - FOR TA's only. 

ground_truth = 'blind_labels';
fid = fopen(ground_truth);
myData = textscan(fid,'%f');
fclose(fid);
testLabels = myData{1};


eval_prediction = 'ziqi_qiong_yuchun_blind_label_fusion.txt'; % best fusion model
% eval_prediction = 'ziqi_qiong_yuchun_blind_label_nn.txt'; % best single model

fid = fopen(eval_prediction);
myData = textscan(fid,'%f');
fclose(fid);
testScores = myData{1};

[eer,~] = compute_eer(testScores, testLabels);
disp(['The EER is for eval',num2str(eer),'%.']);
