# FF Training for Si set

## data source:
https://github.com/MDIL-SNU/SIMPLE-NN

One can download by using the following command.
```
$ wget https://raw.githubusercontent.com/MDIL-SNU/SIMPLE-NN/master/examples/SiO2/ab_initio_output/OUTCAR_comp
```

## Bispectrum
The data source contains 10000 structures, which may require a significant amount of CPU and memory. For a quick run, we use only 250 structutes.
Below is an example with `lmax=3`, corresponding to 30 descriptors.

To invoke the training, just execute 

`$ python bispectrum.py > bispectrum.log &` 


```
============================= Evaluating Training Set ============================

The results for energy: 
    Energy R2     0.970248
    Energy MAE    0.003551
    Energy RMSE   0.004429
The energy figure is exported to: Si-O-Bispectrum/Energy_Train.png


The results for force: 
    Force R2      0.967794
    Force MAE     0.252029
    Force RMSE    0.322588
The force figure is exported to: Si-O-Bispectrum/Force_Train.png
```

