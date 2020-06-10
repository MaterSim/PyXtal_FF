## data source:
https://github.com/MDIL-SNU/SIMPLE-NN

## Summary of results 
- Descriptors computation time was not included
- NN training is performed on a CPU

|Scripts|No. structures|No. epochs|Energy MAE| Force MAE | CPU time|  
|-------|:------------:|:--------:|:--------:|:---------:|:-------:|
|       |              |          |(meV/atom)| (meV/A)   |   (hr)  |  
|`so4-quick.py` | 250  | 500      |   4.88   |  275.4    | 0.23    |
|`so4-full.py`  |2000  | 5000     |   4.11   |  232.9    | 18.41   |
|`so4-full.py`  |2000  | 10000    |   3.89   |  221.7    | 36.5    |
|`so3-full.py`  |10000 | 500      |   2.34   |  151.6    | 35.5    |

## Traning errors from SO3-mini-batch

Energy            |  Force |
:-------------------------:|:-------------------------:
![E](https://github.com/qzhu2017/PyXtal_FF/blob/master/docs/imgs/SiO2_E.png)  |  ![F](https://github.com/qzhu2017/PyXtal_FF/blob/master/docs/imgs/SiO2_F.png)|
