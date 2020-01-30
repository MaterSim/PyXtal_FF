# FF Training for Si set

## data source:
https://github.com/materialsvirtuallab/mlearn/tree/master/data/Si

One can download by using the following command.
```
$ wget https://raw.githubusercontent.com/materialsvirtuallab/mlearn/master/data/Si/test.json
$ wget https://raw.githubusercontent.com/materialsvirtuallab/mlearn/master/data/Si/training.json
```
## Gaussian Symmetry function
The parameters were taken from table S5 of of the [arXiv paper](https://arxiv.org/pdf/1906.08888.pdf).

To invoke the training, just execute 

`$ python gaussian.py > gaussian.log &` 

You may complete the entire calculation by 1 hr. The final trained results are
```
============================= Evaluating Training Set ============================
The results for energy: 
    Energy R2     0.999120
    Energy MAE    0.006587
    Energy RMSE   0.009073

The results for force: 
    Force R2      0.970089
    Force MAE     0.098616
    Force RMSE    0.149284
============================= Evaluating Testing Set =============================
The results for energy: 
    Energy R2     0.999221
    Energy MAE    0.006606
    Energy RMSE   0.008865

The results for force: 
    Force R2      0.961980
    Force MAE     0.112763
    Force RMSE    0.171763
```
Compared with the results reported in Fig. 3 from the [arXiv paper](https://arxiv.org/pdf/1906.08888.pdf ). 
The MAE for energy is slightly higher (`5.88/5.60 meV/atom`), but the MAE for force is smaller (`0.12/0.11 eV/A`).
This is expected since we intentionally remove some G4 descriptors which have rather small range.

## Bispectrum
One can also use the SO4 bispectrum coefficients. Below is an example with `lmax=3`, corresponding to 30 descriptors.

To invoke the training, just execute 

`$ python bispectrum.py > bispectrum.log &` 


```
============================= Evaluating Training Set ============================
The results for energy: 
    Energy R2     0.999435
    Energy MAE    0.005236
    Energy RMSE   0.007274

The results for force: 
    Force R2      0.972657
    Force MAE     0.091834
    Force RMSE    0.144449

============================= Evaluating Testing Set =============================
The results for energy: 
    Energy R2     0.999673
    Energy MAE    0.004531
    Energy RMSE   0.005746

The results for force: 
    Force R2      0.970130
    Force MAE     0.099572
    Force RMSE    0.152243
```
Clearly, the MAE values for both energy (`5.24/4.53 meV/atom`) and forces (`0.09/0.10 eV/A`) are better than the previous Gaussian-14 descriptors.
