## data source:
https://github.com/MDIL-SNU/SIMPLE-NN

## Summary of results 
- Descriptors computation time was not included
- NN training is performed on a CPU

|Scripts| No. structures| No. epochs | Energy MAE | Force MAE | CPU time|  
|-------|:-------------:|:----------:|:----------:|:---------:|:-------:|
|       |               |            | (meV/atom) | (meV/A)   |   (hr)  |  
|`so4-quick.py` | 250   | 500        |   4.88     |  275.4    | 0.23    |
|`so4-full.py`  |2000   | 5000       |   4.11     |  232.9    | 18.41   |
|`so4-full.py`  |2000   | 10000      ||||

