# FF Training for Si set

## data source:
https://github.com/materialsvirtuallab/mlearn/tree/master/data/Si

## Summary of results 
- There are 214 structures for training and 25 structures for testing
- Descriptors computation time was not included
- NN training is performed on a CPU

|Scripts| No. des.      | No. epochs | Energy MAE | Force MAE | Stress MAE |  CPU time|  
|:-----:|:-------------:|:----------:|:----------:|:---------:|:----------:|:--------:|
|       |               |            | (meV/atom) | (meV/A)   | (GPa)      |   (hr)  |
|`bp-NN.py`  | 14       | 1000       |   7.81/5.97|106.2/115.0| 0.79/0.72  |   0.31  |
|`eamd-NN.py`| 21       | 1000       |   6.79/5.27| 90.5/98.2 | 0.48/0.59  |. |
|`so4-NN.py` | 30       | 1000       |   6.79/5.27| 90.5/98.2 | 0.48/0.59  |  0.47   |
|`so3-NN.py` | 30       | 1000       |   7.32/6.08| 81.9/92.9 | 0.56/0.68. |  0.38   |

### Another 1000 step of training

|Scripts| No. des.      | No. epochs | Energy MAE | Force MAE | Stress MAE |  CPU time|  
|:-----:|:-------------:|:----------:|:----------:|:---------:|:----------:|:--------:|
|       |               |            | (meV/atom) | (meV/A)   | (GPa)      |   (hr)  |
|`bp-NN.py`  | 14       | 1000       |   7.31/6.12| 98.1/108.7| 0.81/0.70  |   0.32  |
|`eamd-NN.py`| 21       | 1000       |   6.79/5.27| 90.5/98.2 | 0.48/0.59  |.|
|`so4-NN.py` | 30       | 1000       |   6.13/4.87| 82.6/90.4 | 0.50/0.59  |  0.34   |
|`so3-NN.py` | 30       | 1000       |   5.97/5.08| 74.3/84.8 | 0.47/0.56  |  0.37   |

### Another 1000 step of training

|Scripts| No. des.      | No. epochs | Energy MAE | Force MAE | Stress MAE |  CPU time|  
|:-----:|:-------------:|:----------:|:----------:|:---------:|:----------:|:--------:|
|       |               |            | (meV/atom) | (meV/A)   | (GPa)      |   (hr)  |
|`bp-NN.py`  | 14       |  533       |   7.15/6.67| 97.0/106.3| 0.79/0.66  |   0.15  |
|`eamd-NN.py`| 21       | 1000       |   6.79/5.27| 90.5/98.2 | 0.48/0.59  |.|
|`so4-NN.py` | 30       | 1000       |   5.60/4.77| 78.6/87.2 | 0.51/0.60  |  0.33   |
|`so3-NN.py` | 30       | 1000       |   5.51/4.64| 71.9/82.0 | 0.52/0.59  |  0.41   |



## Properties calculation
After the training is done, one can also check the accuracy of the model based on the properties such as equilibrium lattice constant, elastic properties and phoonon dispersion.
```
$ python test_properties.py -f Si-Bispectrum/16-16-checkpoint.pth 
load the precomputed values from  /scratch/qzhu/anaconda3/lib/python3.7/site-packages/pyxtal_ff-0.0.4-py3.7.egg/pyxtal_ff/descriptors/Wigner_coefficients.npy
      Step     Time          Energy         fmax
BFGS:    0 14:53:31       -4.901490        3.7442
BFGS:    1 14:53:32       -4.955509        3.4924
BFGS:    2 14:53:32       -5.006329        3.3129
BFGS:    3 14:53:32       -5.054866        3.1832
BFGS:    4 14:53:32       -5.101676        3.0775
BFGS:    5 14:53:32       -5.146920        2.9695
BFGS:    6 14:53:32       -5.190384        2.8362
BFGS:    7 14:53:32       -5.231559        2.6616
BFGS:    8 14:53:32       -5.267968        2.4521
BFGS:    9 14:53:33       -5.298639        2.2244
BFGS:   10 14:53:33       -5.323744        1.9957
BFGS:   11 14:53:33       -5.343886        1.7788
BFGS:   12 14:53:33       -5.359865        1.5805
BFGS:   13 14:53:33       -5.372475        1.4029
BFGS:   14 14:53:33       -5.382412        1.2454
BFGS:   15 14:53:33       -5.390246        1.1062
BFGS:   16 14:53:34       -5.396429        0.9831
BFGS:   17 14:53:34       -5.401314        0.8741
BFGS:   18 14:53:34       -5.405178        0.7775
BFGS:   19 14:53:34       -5.408234        0.6916
BFGS:   20 14:53:34       -5.410654        0.6153
BFGS:   21 14:53:34       -5.412569        0.5474
BFGS:   22 14:53:34       -5.414084        0.4869
BFGS:   23 14:53:34       -5.415283        0.4330
BFGS:   24 14:53:35       -5.416232        0.3850
BFGS:   25 14:53:35       -5.416981        0.3423
BFGS:   26 14:53:35       -5.417574        0.3043
BFGS:   27 14:53:35       -5.418042        0.2704
BFGS:   28 14:53:35       -5.418412        0.2403
BFGS:   29 14:53:35       -5.418704        0.2135
BFGS:   30 14:53:35       -5.418935        0.1897
BFGS:   31 14:53:35       -5.419117        0.1685
BFGS:   32 14:53:36       -5.419260        0.1497
BFGS:   33 14:53:36       -5.419373        0.1329
BFGS:   34 14:53:36       -5.419463        0.1181
BFGS:   35 14:53:36       -5.419533        0.1048
BFGS:   36 14:53:36       -5.419589        0.0931
BFGS:   37 14:53:36       -5.419633        0.0827
BFGS:   38 14:53:36       -5.419667        0.0734
BFGS:   39 14:53:36       -5.419694        0.0652
BFGS:   40 14:53:37       -5.419716        0.0579
BFGS:   41 14:53:37       -5.419733        0.0514
BFGS:   42 14:53:37       -5.419746        0.0456
BFGS:   43 14:53:37       -5.419757        0.0405
BFGS:   44 14:53:37       -5.419765        0.0360
BFGS:   45 14:53:37       -5.419772        0.0319
BFGS:   46 14:53:37       -5.419777        0.0283
BFGS:   47 14:53:38       -5.419781        0.0252
BFGS:   48 14:53:38       -5.419784        0.0223
BFGS:   49 14:53:38       -5.419786        0.0198
BFGS:   50 14:53:38       -5.419788        0.0176
BFGS:   51 14:53:38       -5.419790        0.0156
BFGS:   52 14:53:38       -5.419791        0.0139
BFGS:   53 14:53:38       -5.419792        0.0123
BFGS:   54 14:53:38       -5.419793        0.0109
BFGS:   55 14:53:39       -5.419794        0.0097
equlirum cell para:  5.472847701313846
equlirum energy:  tensor(-5.4198, grad_fn=<DivBackward0>)
equlirum stress [  0.0671   0.0671   0.0671   0.0000  -0.0000   0.0000]
This is a Cubic system
2 deformations were generated
      Step     Time          Energy         fmax
BFGS:    0 14:53:39       -5.418636        0.0000
      Step     Time          Energy         fmax
BFGS:    0 14:53:39       -5.419483        0.0000
      Step     Time          Energy         fmax
BFGS:    0 14:53:39       -5.419794        0.0000
      Step     Time          Energy         fmax
BFGS:    0 14:53:39       -5.419569        0.0000
      Step     Time          Energy         fmax
BFGS:    0 14:53:39       -5.418811        0.0000
      Step     Time          Energy         fmax
BFGS:    0 14:53:39       -5.419660        0.0464
BFGS:    1 14:53:39       -5.419686        0.0314
BFGS:    2 14:53:40       -5.419708        0.0000
      Step     Time          Energy         fmax
BFGS:    0 14:53:40       -5.419586        0.0580
BFGS:    1 14:53:40       -5.419626        0.0392
BFGS:    2 14:53:40       -5.419660        0.0000
      Step     Time          Energy         fmax
BFGS:    0 14:53:40       -5.419494        0.0696
BFGS:    1 14:53:40       -5.419552        0.0471
BFGS:    2 14:53:40       -5.419601        0.0000
      Step     Time          Energy         fmax
BFGS:    0 14:53:40       -5.419386        0.0812
BFGS:    1 14:53:41       -5.419465        0.0549
BFGS:    2 14:53:41       -5.419531        0.0000
      Step     Time          Energy         fmax
BFGS:    0 14:53:41       -5.419261        0.0928
BFGS:    1 14:53:41       -5.419364        0.0627
BFGS:    2 14:53:41       -5.419451        0.0000
C_11:   167.24(GPa)
C_12:   108.21(GPa)
C_44:    53.64(GPa)
Bulk modulus, Shear modulus, Young's modulus, Poisson's ratio
127.88503896159091 43.987841313073176 118.38960570000751 0.3457082930871417
127.88503896159094 40.421822436537326 109.70678365395833 0.3570242141630191
127.88503896159092 42.20483187480525 114.04819467698292 0.3513662536250804
displacement 1 / 1
The detected space group is Fd-3m with a tolerance of 1.00e-05
```
