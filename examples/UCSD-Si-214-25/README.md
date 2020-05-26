# FF Training for Si set

## data source:
https://github.com/materialsvirtuallab/mlearn/tree/master/data/Si

## Summary of results 

It is generally difficult to estimate how many training steps are needed to get the results be converged.
Therefore, we choose to repeat training the following models for each 1000 epochs with the BFGS algorithm.
The following tables show the results after each 1000 epochs with an NN of [16, 16] hidden layers.

- There are 214 structures for training and 25 structures for testing
- For each structure, the total energy, atomic forces andd stress tensors are trained together with the coefficients of [1. 3e-2, 1e-5]
- The regularization factor is 1e-6.
- Descriptors computation time was not included
- NN training is performed on a CPU

### 0-1000

|Scripts| No. des.      | No. epochs | Energy MAE | Force MAE | Stress MAE |  CPU time|  
|:-----:|:-------------:|:----------:|:----------:|:---------:|:----------:|:--------:|
|       |               |            | (meV/atom) | (meV/A)   | (GPa)      |   (hr)  |
|`bp-NN.py`  | 14       | 1000       |   7.81/5.97|106.2/115.0| 0.79/0.72  |   0.31  |
|`eamd-NN.py`| 21       | 1000       |   6.79/5.27| 90.5/98.2 | 0.48/0.59  |. |
|`so4-NN.py` | 30       | 1000       |   6.79/5.27| 90.5/98.2 | 0.48/0.59  |  0.47   |
|`so3-NN.py` | 30       | 1000       |   7.32/6.08| 81.9/92.9 | 0.56/0.68. |  0.38   |

### 1000-2000

|Scripts| No. des.      | No. epochs | Energy MAE | Force MAE | Stress MAE |  CPU time|  
|:-----:|:-------------:|:----------:|:----------:|:---------:|:----------:|:--------:|
|       |               |            | (meV/atom) | (meV/A)   | (GPa)      |   (hr)  |
|`bp-NN.py`  | 14       | 1000       |   7.31/6.12| 98.1/108.7| 0.81/0.70  |   0.32  |
|`eamd-NN.py`| 21       | 1000       |   6.79/5.27| 90.5/98.2 | 0.48/0.59  |.|
|`so4-NN.py` | 30       | 1000       |   6.13/4.87| 82.6/90.4 | 0.50/0.59  |  0.34   |
|`so3-NN.py` | 30       | 1000       |   5.97/5.08| 74.3/84.8 | 0.47/0.56  |  0.37   |

### 2000-3000

|Scripts| No. des.      | No. epochs | Energy MAE | Force MAE | Stress MAE |  CPU time|  
|:-----:|:-------------:|:----------:|:----------:|:---------:|:----------:|:--------:|
|       |               |            | (meV/atom) | (meV/A)   | (GPa)      |   (hr)  |
|`bp-NN.py`  | 14       |  533       |   7.15/6.67| 97.0/106.3| 0.79/0.66  |   0.15  |
|`eamd-NN.py`| 21       | 1000       |   6.79/5.27| 90.5/98.2 | 0.48/0.59  |.|
|`so4-NN.py` | 30       | 1000       |   5.60/4.77| 78.6/87.2 | 0.51/0.60  |  0.33   |
|`so3-NN.py` | 30       | 1000       |   5.51/4.64| 71.9/82.0 | 0.52/0.59  |  0.41   |

### 3000-4000

|Scripts| No. des.      | No. epochs | Energy MAE | Force MAE | Stress MAE |  CPU time|  
|:-----:|:-------------:|:----------:|:----------:|:---------:|:----------:|:--------:|
|       |               |            | (meV/atom) | (meV/A)   | (GPa)      |   (hr)  |
|`bp-NN.py`  | 14       |  0         |   7.15/6.67| 97.0/106.3| 0.79/0.66  |   0     |
|`eamd-NN.py`| 21       | 1000       |   6.79/5.27| 90.5/98.2 | 0.48/0.59  |.|
|`so4-NN.py` | 30       | 1000       |   5.24/4.48| 77.1/86.5 | 0.51/0.59  |  0.33   |
|`so3-NN.py` | 30       | 1000       |   4.97/4.20| 69.7/80.1 | 0.52/0.60  |  0.36   |

### 4000-5000

|Scripts| No. des.      | No. epochs | Energy MAE | Force MAE | Stress MAE |  CPU time|  
|:-----:|:-------------:|:----------:|:----------:|:---------:|:----------:|:--------:|
|       |               |            | (meV/atom) | (meV/A)   | (GPa)      |   (hr)  |
|`bp-NN.py`  | 14       |  0         |   7.15/6.67| 97.0/106.3| 0.79/0.66  |   0     |
|`eamd-NN.py`| 21       | 1000       |   6.79/5.27| 90.5/98.2 | 0.48/0.59  |.|
|`so4-NN.py` | 30       | 1000       |   5.23/4.50| 75.6/86.1 | 0.49/0.56  |  0.33   |
|`so3-NN.py` | 30       | 195        |   4.83/4.19| 69.7/80.2 | 0.50/0.59  |  0.08   |


## Properties calculation
After the training is done, one can also check the accuracy of the model based on the properties such as equilibrium lattice constant, elastic properties and phoonon dispersion.

|          | Exp   | DFT   | bp-NN |eamd-NN| so4-NN | so3-NN|  
|:--------:|:-----:|:-----:|:-----:|:-----:|:------:|:-----:|
| a(A)     | 5.429 |5.469  | 5.448 |       | 5.471  | 5.462 |
|C11(GPa)  | 167   |156    | 141   |  | 159    | 154   |
|C12(GPa)  |  65   | 65    |  88   |  |  90    |  75   |
|C44(GPa)  |  81   | 76    |  57   |  |  62    |  70   |
|B-VRH(GPa)|  99   | 95    |  106  |  | 113    |  101  |

