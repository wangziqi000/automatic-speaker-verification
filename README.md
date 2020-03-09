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

### LPCC (vanila_lpcc.m)
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
