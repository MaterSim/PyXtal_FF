# Example of SNAP\_NN model for SiO2

## Remark
Last updated by Qiang Zhu on 2021/05/14.

## Validation between python-ASE and lammps results

The `validate.py` will create a randomly perturbed low temperature quartz. Then it will perform a single point energy calculation from both `Python-ASE` and `LAMMPS` calcuator. 
```
$ python validate.py
ASE calculator with pyxtal_ff force field

Energy:  -69.641 eV
Forces (eV/A)
[[ 15.2541   3.0779   4.7569]
 [ -4.7716  -0.4697  -0.1227]
 [  0.0284  -2.1275   0.4802]
 [ -1.2777  -0.5880  -1.4686]
 [ -2.9655  -0.2107  -0.4350]
 [ -0.1211   0.0865  -0.4512]
 [ -1.2014  -2.9408   2.1313]
 [ -0.5169   0.0630   0.1996]
 [ -4.4283   3.1093  -5.0904]]
Stresses (eV/A3)
[  0.0351  -0.0396  -0.0558   0.0601  -0.0649   0.0105]

LAMMPS calculator with pyxtal_ff force field

Energy:  -69.641 eV
Forces (eV/A)
[[ 15.2541   3.0779   4.7569]
 [ -4.7716  -0.4697  -0.1227]
 [  0.0284  -2.1275   0.4802]
 [ -1.2777  -0.5880  -1.4686]
 [ -2.9655  -0.2107  -0.4350]
 [ -0.1211   0.0865  -0.4512]
 [ -1.2014  -2.9408   2.1313]
 [ -0.5169   0.0630   0.1996]
 [ -4.4283   3.1093  -5.0904]]
Stresses (eV/A3)
[  0.0351  -0.0396  -0.0558   0.0601  -0.0649   0.0105]
```
