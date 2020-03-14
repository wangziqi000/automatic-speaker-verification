# ECEM214A-W20-Final-PJ
This is the final project of ECE M214 Digital Speech Processing, UCLA, Winter 2020

## Methods
1. Source-filter based methods
    - Pitch + LPC
    - LPCC
    - VQual Features

1. Cepstral Coefficients
    - LFCC
    - MFCC
    - CQCC
    - add delta/delta^2
    - apply PCA for dimension reduction

1. NN based methods

1. i-Vector based methods
    - use LDA, PCA
    - use cosine distance
    - detect voiced segments and remove silence

1. Score fusion

## Usage - for Blind Test

### For TA
Please run '[**blind_eer_for_TA.m**](/blind_scores/blind_eer_for_TA.m)'. The predicted scores for blind trials are stored in [/blind_scores](/blind_scores) folder, where '[ziqi_qiong_yuchun_blind_label_fusion.txt](/blind_scores/ziqi_qiong_yuchun_blind_label_fusion.txt)’ is the score from the best fusion model, and '[ziqi_qiong_yuchun_blind_label_nn.txt](/blind_scores/ziqi_qiong_yuchun_blind_label_nn.txt)' is that of the best single model.

Test EER results of the fusion model and the NN model are as below:
- NN model

|  Train/Test | Read-Read | Phone-Phone | Read-Phone |
|:-----------:|:---------:|:-----------:|:----------:|
|  Read-Read  |   16.14%  |    17.49%   |     23%    |
| Phone-Phone |   18.4%   |    17.6%    |   22.31%   |

- Fusion model

|  Train/Test | Read-Read | Phone-Phone | Read-Phone |
|:-----------:|:---------:|:-----------:|:----------:|
|  Read-Read  |   9.28%   |    10.17%   |    24.2%   |
| Phone-Phone |   10.8%   |     10%     |    23.4%   |

### For Fusion Scores
Run '[blind_fusion.m](blind_fusion.m)'. You can play with score fusion weights here. 
If you want to play with each feature, you can write a script with the function 'blind_FEATURE.m'. Sample scripts are included in '[blind_test.m](blind_test.m)'.

## Usage - for Playing with Different Methods
For each feature, there is a 'script_FEATURE.m' which shows all the way from data processing to training and validation. However, for some features which does not yield very good results on validation set, we do not use them for blind test. You can see a brief comment at the beginning of each script. Also, there may be a 'fun_FEATURE.m' function which does roughly the same thing but is encapsulated as a MATLAB Function.


## Data

The UCLA Speaker Variability Database is a database designed to capture variability both between speakers and within a single speaker. Speech utterances by 50 males in two different styles: read speech and phone call conversation are included in the training set. All the utterances are text-independent i.e all speakers are speaking different text. In the case of a phone call recording, only one side of the conversation is recorded in the studio. The microphone and environment conditions are the same for both styles. Each utterance is about 2 seconds long​.

This dataset is created by UCLA, SPAPL. Thus, we cannot distribute or use it outside this project. However, methods used in this work is universal, and you can use your own dataset for this work.

In the script, you have six knobs that you should prepare in advance:

- allFiles = '[allFiles.txt](filelists/allFiles.txt)'; 

    A list including all your sound files dataset for training and validation. A typical line looks like “speaker1_sentence_001.wav\n”.

- trainList = '[train_read_trials.txt](filelists/train_read_trials.txt)';

    including speech pairs and labels for model training.
    Typical lines look like this: 

    “Speaker1_sentence_001.wav  Speaker1_phone_001.wav  1\n”

    “Speaker1_sentence_001.wav  Speaker1_phone_001.wav  1\n”

- testList = '[test_read_trials.txt](filelists/test_read_trials.txt)'; 

    Speech pairs and labels for model testing(validation). 

- blind_list = '[blind_file_list](filelists/blind_file_list)'; 

    A list including all your sound dataset filenames for blind test.

- blind_trials = '[blind_trials](filelists/blind_trials)';

    Speech pairs for your blind test.

- ground_truth = 'blind_labels';

    Corresponding labels for your blind list pairs.


## Credits - References and inspirations of the implementation

1. Source-filter based methods
    - Pitch + LPC
    - LPCC https://www.mathworks.com/help/dsp/ref/dsp.lpctocepstral-system-object.html
    - VQual Features http://www.phonetics.ucla.edu/voicesauce/

1. Cepstral Coefficients
    - LFCC https://www.asvspoof.org/asvspoof2019/ASVspoof_2019_baseline_CM_v1.zip
    - MFCC
    - CQCC https://www.asvspoof.org/asvspoof2019/ASVspoof_2019_baseline_CM_v1.zip
    - add delta/delta^2
    - apply PCA for dimension reduction

1. NN based methods
    - https://github.com/a-nagrani/VGGVox

1. i-Vector based methods
    - https://github.com/SatyamGaba/Speaker-Recognition

