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
|  Read-Read  |   %  |   %  |   %   |
| Phone-Phone |   %  |   %  |   %   |

### LPCC (vanila_lpcc.m)
feature num = 10
|  Train/Test | Read-Read | Phone-Phone | Read-Phone |
|:-----------:|:---------:|:-----------:|:----------:|
|  Read-Read  | 33.9263%  |      32.8%  | 43.4211%   |
| Phone-Phone |   34%     |      32%    | 42.7158%   |

### LFCC (vanilla_lfcc.m)
#### Include delta, delta^2
Window_Length = 20, NFFT = 512, No_Filter = 50

feature num = 150
|  Train/Test | Read-Read | Phone-Phone | Read-Phone |
|:-----------:|:---------:|:-----------:|:----------:|
|  Read-Read  |   %  |   %  |   %   |
| Phone-Phone |   %  |   %  |   %   |

### MFCC (vanilla_mfcc.m)
#### Include delta, delta^2
feature num = 
|  Train/Test | Read-Read | Phone-Phone | Read-Phone |
|:-----------:|:---------:|:-----------:|:----------:|
|  Read-Read  |   %  |   %  |   %   |
| Phone-Phone |   %  |   %  |   %   |

### PCA + MFCC (pca_mfcc.m)
#### NumCoeffs = 40; Exclude delta, delta^2
feature num = 41 (apply dimension reduction)
|  Train/Test | Read-Read | Phone-Phone | Read-Phone |
|:-----------:|:---------:|:-----------:|:----------:|
|  Read-Read  |   %  |   %  |   %   |
| Phone-Phone |   %  |   %  |   %   |

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
