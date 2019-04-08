# FF-MAPP (Force-Field Machine-learning interAtomic Potential Predictions)

# This is an ongoing project

## Introduction

In representing a material, Cartesian coordinates are not a suitable choice as descriptors. Symmetry functions are descriptors that are invariant with respect to translation and rotation. Symmetry functions also works well for the construction of high-dimensional neural network potential energy surfaces. 

There are 5 types of symmetry functions:
1. G_1 = 
2. G_2 = 
3. G_3 =
4. G_4 = 
5. G_5 =

## Test
Predicting formation energy of Cu:
```
cd test/
```
```
python cu_bispectrum.py
```

## Dependencies
You need lammps to run the snap descriptor.
Also, you need to tell the snap.py to find your lammps executable.
