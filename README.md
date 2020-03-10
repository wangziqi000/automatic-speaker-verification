# ECEM214A-W20-Final-PJ
This is the final project of ECE M214 Digital Speech Processing, UCLA, Winter 2020

## Methods
- Pitch + LPC
- LPCC
- LFCC
- MFCC
- PCA + MFCC
- CQCC
- PCA + CQCC

## Results
### Pitch (sample.m, Baseline)
feature num = 1
|  Train/Test | Read-Read | Phone-Phone | Read-Phone |
|:-----------:|:---------:|:-----------:|:----------:|
|  Read-Read  |   39.40%  |   40.7684%  |   45.20%   |
| Phone-Phone |   39.00%  |   40.8974%  |    44.8%   |

### Pitch + LPC (pitch_and_lpc.m)
Resample to 8000Hz

feature num = 10
|  Train/Test | Read-Read | Phone-Phone | Read-Phone |
|:-----------:|:---------:|:-----------:|:----------:|
|  Read-Read  |   35.2%   |   32%       |   43.4%    |
| Phone-Phone |   35%     |   30.5684%  |   43.6%    |

### LPC
feature num = 9
|  Train/Test | Read-Read | Phone-Phone | Read-Phone |
|:-----------:|:---------:|:-----------:|:----------:|
|  Read-Read  |   37.6%   |   40.8%     |   46.6%    |
| Phone-Phone |   37.2%   |   40.4%     |   46.8%    |

### LPCC
feature num = 10
|  Train/Test | Read-Read | Phone-Phone | Read-Phone |
|:-----------:|:---------:|:-----------:|:----------:|
|  Read-Read  | 33.9263%  |      32.8%  | 43.4211%   |
| Phone-Phone |   34%     |      32%    | 42.7158%   |

### LFCC (vanilla_lfcc.m)
#### Include delta, delta^2
Window_Length = 20, NFFT = 512, No_Filter = 30

feature num = 90
|  Train/Test | Read-Read | Phone-Phone | Read-Phone |
|:-----------:|:---------:|:-----------:|:----------:|
|  Read-Read  |  22.8%    |    %   |    %   |
| Phone-Phone |  % |    %    |    %   |

#### Include delta, delta^2
Window_Length = 20, NFFT = 512, No_Filter = 50

feature num = 150
|  Train/Test | Read-Read | Phone-Phone | Read-Phone |
|:-----------:|:---------:|:-----------:|:----------:|
|  Read-Read  |  24.1263% |    %   |    %   |
| Phone-Phone |  % |    %    |    %   |

#### Exclude delta, delta^2
Window_Length = 20, NFFT = 512, No_Filter = 50

feature num = 50
|  Train/Test | Read-Read | Phone-Phone | Read-Phone |
|:-----------:|:---------:|:-----------:|:----------:|
|  Read-Read  |    22%    |    22.8%   |    36.8%   |
| Phone-Phone |  21.7474% |    22.8%    |    37.2%   |

### PCA + LFCC (vanilla_lfcc.m)
#### Exclude delta, delta^2
Window_Length = 20, NFFT = 512, No_Filter = 450

|  Train/Test | Read-Read | Phone-Phone | Read-Phone |
|:-----------:|:---------:|:-----------:|:----------:|
|  Read-Read  |  11.7895% |    20.2%    |    41.8%   |
| Phone-Phone |  13.4%    |    19.4%    |    41.2%   |

### MFCC (vanilla_mfcc.m)
#### Include delta, delta^2
feature num = 42
|  Train/Test | Read-Read | Phone-Phone | Read-Phone |
|:-----------:|:---------:|:-----------:|:----------:|
|  Read-Read  |   21.4%   |   17.6%     |   37.8%    |
| Phone-Phone |   23.2%   |   18.4%     |   37.8%    |

#### Exclude delta, delta^2
feature num = 14
|  Train/Test | Read-Read | Phone-Phone | Read-Phone |
|:-----------:|:---------:|:-----------:|:----------:|
|  Read-Read  |   21.2%   |   18%       |   38.2%    |
| Phone-Phone |   21.6%   |   18.8%     |   38.2%    |

### PCA + MFCC (pca_mfcc.m)
#### NumCoeffs = 40; Exclude delta, delta^2
feature num = 41
|  Train/Test | Read-Read | Phone-Phone | Read-Phone |
|:-----------:|:---------:|:-----------:|:----------:|
|  Read-Read  | 14.7474%  |   18.4%     |   39%      |
| Phone-Phone |   15.8%   |   17.6%     |   38.1579% |

#### NumCoeffs = 40; Include delta, delta^2
feature num = 48
|  Train/Test | Read-Read | Phone-Phone | Read-Phone |
|:-----------:|:---------:|:-----------:|:----------:|
|  Read-Read  |   14.4%   |   18.2737%  |   38.8%    |
| Phone-Phone |   17.4%   |    18%      |   38.1895% |


#### NumCoeffs = 35; Exclude delta, delta^2
feature num = 36
|  Train/Test | Read-Read | Phone-Phone | Read-Phone |
|:-----------:|:---------:|:-----------:|:----------:|
|  Read-Read  | 15.0421%  |   %     |   %      |
| Phone-Phone |   %   |   %     |   % |

#### NumCoeffs = 30; Exclude delta, delta^2
feature num = 31
|  Train/Test | Read-Read | Phone-Phone | Read-Phone |
|:-----------:|:---------:|:-----------:|:----------:|
|  Read-Read  |   15.2%   |   %     |   %      |
| Phone-Phone |   %   |   %     |   % |

#### NumCoeffs = 25; Exclude delta, delta^2
feature num = 26
|  Train/Test | Read-Read | Phone-Phone | Read-Phone |
|:-----------:|:---------:|:-----------:|:----------:|
|  Read-Read  |   17.4%   |   %     |   %      |
| Phone-Phone |   %   |   %     |   % |

#### NumCoeffs = 20; Exclude delta, delta^2
feature num = 21
|  Train/Test | Read-Read | Phone-Phone | Read-Phone |
|:-----------:|:---------:|:-----------:|:----------:|
|  Read-Read  |  19.5368% |   %     |   %      |
| Phone-Phone |   %   |   %     |   % |

### CQCC (vanilla_cqcc.m)
#### ZsdD = 'ZsdD', include delta, delta^2
feature num = 60
|  Train/Test | Read-Read | Phone-Phone | Read-Phone |
|:-----------:|:---------:|:-----------:|:----------:|
|  Read-Read  |   21.2%   |   19.3263%  |   38.4%    |
| Phone-Phone |   22%     |   21.6%     | 39.2105%   |

### PCA + CQCC (pca_cqcc.m)
#### ZsdD = 'ZsdD', include delta, delta^2
feature num = 20 (apply dimension reduction)
|  Train/Test | Read-Read | Phone-Phone | Read-Phone |
|:-----------:|:---------:|:-----------:|:----------:|
|  Read-Read  | 15.6211%  |   19.2%     |   36.8%    |
| Phone-Phone | 16.3368%  |   19.2211%  |   36.3158% |

### MFCC + LPC ()
#### Exclude delta, delta^2
#### (mfcc=40 (1:41)取(1:30), lpcs(3:4))
(apply dimension reduction pca = 0.9999)
|  Train/Test | Read-Read | Phone-Phone | Read-Phone |
|:-----------:|:---------:|:-----------:|:----------:|
|  Read-Read  | 14.7263%  |   17.2%     |   34.6%    |
| Phone-Phone | 17.1789%  |   16.8%     |   35.6%    |

### MFCC + LPC ()
#### Exclude delta, delta^2
#### (mfcc=40 (1:41)取(1:30), lpcs(3:5))
(apply dimension reduction pca = 0.9999)
|  Train/Test | Read-Read | Phone-Phone | Read-Phone |
|:-----------:|:---------:|:-----------:|:----------:|
|  Read-Read  | 15.2%     |   17.0526%  |   35.3263% |
| Phone-Phone | 15.9789%  |   16.8%     |   35.9474% |

### MFCC + LPC ()
#### Exclude delta, delta^2
#### (mfcc=40 (1:41)取(1:35), lpcs(3:5))
(apply dimension reduction pca = 0.9999)
|  Train/Test | Read-Read | Phone-Phone | Read-Phone |
|:-----------:|:---------:|:-----------:|:----------:|
|  Read-Read  | 14.8%     |   17.1263%  |   35.3053% |
| Phone-Phone | 17.3053%  |   16.5263%  |   36.6%    |

### MFCC + cqcc + LPC ()
#### Exclude delta, delta^2
(apply dimension reduction pca = 0.9999)
|  Train/Test | Read-Read | Phone-Phone | Read-Phone |
|:-----------:|:---------:|:-----------:|:----------:|
|  Read-Read  | 15.9368%  |   20%       |   36.6105% |
| Phone-Phone | 16.4%     |   19.3474%  |   36.7474% |

### i-Vector (vanilla_ivector.m)
#### Exclude delta, delta^2
feature num = 70 (use score_gplda)
|  Train/Test | Read-Read | Phone-Phone | Read-Phone |
|:-----------:|:---------:|:-----------:|:----------:|
|  Read-Read  |   29.8%   |   34.4%     |     36%    |
| Phone-Phone |   30.6%   |   30.4%     |    34.8%   |

feature num = 69 (not use score_gplda)
|  Train/Test | Read-Read | Phone-Phone | Read-Phone |
|:-----------:|:---------:|:-----------:|:----------:|
|  Read-Read  |   30%     |   34.4526%  |   35.6421% |
| Phone-Phone |   30.8%   |   33%       |    36.4%   |
