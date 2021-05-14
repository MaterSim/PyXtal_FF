# Example of SNAP\_NN model for elemental silicon

## Remark
This is an example to train the neural networks model based on the SNAP/SO3 descriptor based on an online [dataset](https://github.com/materialsvirtuallab/mlearn/tree/master/data/Si)

Note that the [zbl potential](https://lammps.sandia.gov/doc/pair_zbl.html) is included during the training.

At the moment, SO3 descriptor is disabled.

Last updated by Qiang Zhu on 2021/05/14.

## Force field training
```
python train.py
```

After training is complete, you expect to find a folder called `Si-snap` with several important files
- `12-12-checkpoint.pth`: NN model in Pytorch format
- `DescriptorParams.txt`: parameters to compute the descriptor in LAMMPS format
- `NN_weights.txt`: weight parameters of the model in LAMMPS format

The training takes about 20-30 minutes. 
**If you cannot wait, please skip this step and use the existing files for the following steps.**

## Atomistic simulation via ASE 
PyXtal\_FF provides a built in `ASE` calculator. You can simply use it for some light weight calculations.
Below is an example of running NVT MD simulation for 1000 Si atoms.
```
$ python md.py -f Si-snap-zbl/12-12-checkpoint.pth 
MD simulation for  1000  atoms
Step:    0 [ 31.17]: Epot = -5.357eV  Ekin = 0.117eV (T=903K)  Etot = -5.241eV 
Step:    1 [ 16.30]: Epot = -5.329eV  Ekin = 0.091eV (T=702K)  Etot = -5.238eV 
Step:    2 [ 16.27]: Epot = -5.297eV  Ekin = 0.061eV (T=471K)  Etot = -5.236eV 
Step:    3 [ 16.25]: Epot = -5.276eV  Ekin = 0.041eV (T=318K)  Etot = -5.235eV 
Step:    4 [ 16.17]: Epot = -5.272eV  Ekin = 0.038eV (T=294K)  Etot = -5.234eV 
Step:    5 [ 16.13]: Epot = -5.279eV  Ekin = 0.047eV (T=366K)  Etot = -5.232eV 
Step:    6 [ 16.12]: Epot = -5.289eV  Ekin = 0.058eV (T=452K)  Etot = -5.231eV 
Step:    7 [ 16.15]: Epot = -5.296eV  Ekin = 0.066eV (T=514K)  Etot = -5.230eV 
Step:    8 [ 16.10]: Epot = -5.296eV  Ekin = 0.068eV (T=525K)  Etot = -5.228eV 
Step:    9 [ 16.06]: Epot = -5.291eV  Ekin = 0.064eV (T=494K)  Etot = -5.227eV 
```
The time cost is about 16 seconds for each time step.
For more types of simulations based on `Python-ASE`, checkout [this example](https://github.com/qzhu2017/PyXtal_FF/blob/master/pyxtal_ff/test_properties.py)

## LAMMPS Installation
Run the simulation with `Python` is not recommended for large scale systems. For practical simulation, you can plug in the force field to `LAMMPS` as follows.

First, follow the steps below to install LAMMPS-MLIAP and python wrapper

```
$ git clone https://github.com/pedroantoniosantosf/lammps.git
$ cd lammps/src
$ make yes-snap
$ make yes-mliap
$ make yes-python
$ make mpi -j 8 mode=shlib  #speed up the compilation
$ make install-python
```
At the end, you expect to get an executable called `lmp_mpi` and `liblammps_mpi.so` in the src directory, as well as a soft link liblammps.so, which is what the Python wrapper will load by default.
Then one just need to add the path of src to the `.bashrc` file as follows,
```
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/scratch/qzhu/soft/lammps/src 
```
**Note that you need to modify the path**


## Validation between python-ASE and lammps results

The `validate.py` will create a randomly perturbed silicon diamond structure. Then it will perform a single point energy calculation from both `Python-ASE` and `LAMMPS` calcuator for 100 randomly perturbed structures. The last three columns present the differences in energy, forces and stress tensors. Ideally, the values should be very close to zero.
```
$ python validate.py 
  0  -42.260 eV   17.729 GPa   -0.000    0.000    0.000
  1  -42.217 eV   17.732 GPa    0.000    0.000    0.000
  2  -42.257 eV   17.730 GPa   -0.000    0.000    0.000
  3  -42.266 eV   17.729 GPa   -0.000    0.000    0.000
  4  -41.976 eV   17.734 GPa   -0.000    0.000    0.000
  5  -40.524 eV   17.057 GPa   -0.000    0.000    0.000
  6  -41.971 eV   17.734 GPa   -0.000    0.000    0.000
  7  -40.811 eV   17.278 GPa    0.000    0.000    0.000
  8  -41.759 eV   17.709 GPa    0.000    0.000    0.000
  9  -41.963 eV   17.733 GPa    0.000    0.000    0.000
```
If you cannot get the similar results, please check your `LAMMPS` installation.


## Fast simulation via LAMMPS

Finally, you can run LAMMPS directly. 
It will work as any usual `LAMMPS` calculation, except that you need to specify a new `pair style`
```
# Potential
pair_style hybrid/overlay &
mliap model nn Si-snap/NN_weights.txt &
descriptor sna Si-snap/DescriptorParam.txt &
zbl 1.5 2.0
pair_coeff 1 1 zbl 14.0 14.0
pair_coeff * * mliap Si
```

The `md.in` file gives an example to run NPT MD simulation at 500 K for 1000 Si atoms.
```
lmp_mpi < md.in > md.out
```
On a single CPU, it needs ~120 seconds to complete 1000 steps, which is abouot **100x** faster than the `ASE` caculator.
