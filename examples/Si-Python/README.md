# FF Training for Si set

## Remark
This is the first example to let the users be familiar with PyXtal\_FF.
Last updated by Qiang Zhu on 2021/05/10.

## data source:
https://github.com/materialsvirtuallab/mlearn/tree/master/data/Si

## Summary of results 

It is generally difficult to estimate how many training steps are needed to get the results be converged.
Therefore, we choose to repeat training the following models for each 1000 epochs with the BFGS algorithm.
The following tables show the results after each 1000 epochs with an NN of [16, 16] hidden layers.

- There are 214 structures for training and 25 structures for testing
- For each structure, the total energy, atomic forces andd stress tensors are trained together with the coefficients of [1. 2e-2, 2e-3]
- The regularization factor is 1e-6.
- Descriptors computation time was not included
- NN training is performed on a CPU

### 0-1000

|Scripts| No. des. |No. epochs|Energy MAE | Force MAE |Stress MAE|CPU time|  
|:-----:|:--------:|:--------:|:--------: |:---------:|:--------:|:------:|
|       |          |          |(meV/atom) | (meV/A)   | (GPa)    |  (hr)  |
|`bp-NN.py`  | 14  | 1000     |10.40/12.06|116.9/129.4| 0.44/0.45|  0.30  |
|`eamd-NN.py`| 21  | 1000     | 7.19/6.10 | 96.7/106.9| 0.31/0.37|  0.31  |
|`so4-NN.py` | 30  | 1000     | 5.20/4.44 | 91.4/100.5| 0.26/0.38|  0.34  |
|`so3-NN.py` | 30  | 1000     | 5.80/5.91 | 84.3/97.0 | 0.26/0.33|  0.38  |

### 1000-2000

|Scripts| No. des. |No. epochs|Energy MAE| Force MAE |Stress MAE|CPU time|  
|:-----:|:--------:|:--------:|:--------:|:---------:|:--------:|:------:|
|       |          |          |(meV/atom)| (meV/A)   | (GPa)    |  (hr)  |
|`bp-NN.py`  | 14  | 1000     |8.96/11.21|111.2/124.4| 0.39/0.42|  0.27  |
|`eamd-NN.py`| 21  | 1000     | 6.58/4.87| 86.4/94.4 | 0.31/0.36|  0.31  |
|`so4-NN.py` | 30  | 1000     | 4.55/3.69| 83.7/93.8 | 0.25/0.32|  0.33  |
|`so3-NN.py` | 30  | 1000     | 4.39/4.00| 76.4/87.0 | 0.24/0.32|  0.34  |

### 2000-3000

|Scripts| No. des. |No. epochs|Energy MAE| Force MAE |Stress MAE|CPU time|  
|:-----:|:--------:|:--------:|:--------:|:---------:|:--------:|:------:|
|       |          |          |(meV/atom)| (meV/A)   | (GPa)    |  (hr)  |
|`bp-NN.py`  | 14  | 1000     |7.89/10.13|107.2/101.3| 0.38/0.40|  0.26  |
|`eamd-NN.py`| 21  | 1000     | 6.41/5.47| 85.0/92.0 | 0.30/0.35|  0.29  |
|`so4-NN.py` | 30  | 1000     | 4.43/3.88| 80.8/90.8 | 0.24/0.31|  0.36  |
|`so3-NN.py` | 30  | 1000     | 4.16/3.75| 73.4/84.1 | 0.24/0.32|  0.36  |

### 3000-4000

|Scripts| No. des. |No. epochs|Energy MAE| Force MAE |Stress MAE|CPU time|  
|:-----:|:--------:|:--------:|:--------:|:---------:|:--------:|:------:|
|       |          |          |(meV/atom)| (meV/A)   | (GPa)    |  (hr)  |
|`bp-NN.py`  | 14  | 1000     | 7.60/9.25|105.5/120.0| 0.36/0.40|  0.26  |
|`eamd-NN.py`| 21  | 1000     | 6.16/4.81| 83.4/90.8 | 0.29/0.34|  0.33  |
|`so4-NN.py` | 30  | 1000     | 4.07/3.83| 78.7/89.4 | 0.24/0.30|  0.33  |
|`so3-NN.py` | 30  | 1000     | 4.08/3.95| 72.1/82.6 | 0.24/0.32|  0.33  |

Further training will lead to very minor change. Therefore we terminate the training here.

## Properties calculation
After the training is done, one can also check the accuracy of the model based on the properties such as equilibrium lattice constant, elastic properties and phoonon dispersion.
The command is
`python test_properties.py -f Si-BP/16-16-checkpoint.pth`


|          | Exp   | DFT   | bp-NN |eamd-NN| so4-NN | so3-NN|  
|:--------:|:-----:|:-----:|:-----:|:-----:|:------:|:-----:|
| a(A)     | 5.429 |5.469  | 5.473 | 5.467 | 5.466  | 5.469 |
|C11(GPa)  | 167   | 156   | 129   | 129   | 145    | 149   |
|C12(GPa)  |  65   | 65    |  71   | 67    |  55    |  56   |
|C44(GPa)  |  81   | 76    |  59   | 65    |  68    |  69   |
|B-VRH(GPa)|  99   | 95    |  91   | 88    |  85    |  87   |

![phonon](https://github.com/qzhu2017/PyXtal_FF/blob/dxdr-3D/docs/imgs/Si_phonon.png)
