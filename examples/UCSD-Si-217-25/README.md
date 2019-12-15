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


Depending on the `runner`, you may complete the entire calculation by 1-3 hrs. The final trained results are
```
Energy R2  : 0.999060
Energy MAE : 0.006885
Force R2   : 0.966034
Force MAE  : 0.104944

Export the figure to: Si-BehlerParrinello/Train.png

Energy R2  : 0.998944
Energy MAE : 0.007627
Force R2   : 0.962472
Force MAE  : 0.116448

Export the figure to: Si-BehlerParrinello/Test.png
```
Compared with the results reported in Fig. 3 from the [arXiv paper](https://arxiv.org/pdf/1906.08888.pdf ). 
The MAE for energy is slightly higher (`5.88/5.60 meV/atom`), but the MAE for force is smaller (`0.12/0.11 eV/A`).
This is expected since we intentionally remove some G4 descriptors which have rather small range.

## Bispectrum

Under construction.
