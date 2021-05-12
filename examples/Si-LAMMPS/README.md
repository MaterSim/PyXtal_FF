# Example of SNAP\_NN model for elemental silicon

## Remark
This is an example to train the neural networks model based on the SNAP/SO3 descriptor based on an online [dataset](https://github.com/materialsvirtuallab/mlearn/tree/master/data/Si)

Note that the [zbl potential](https://lammps.sandia.gov/doc/pair_zbl.html) is included during the training.

Last updated by Qiang Zhu on 2021/05/10.

## Force field training
```
python train.py
```

After training is complete, you expect to find a folder called `Si-snap` with several important files
- `16-16-checkpoint.pth`: NN model in Pytorch format
- `DescriptorParams.txt`: parameters to compute the descriptor in LAMMPS format
- `NN_weights.txt`: weight parameters of the model in LAMMPS format

The training takes about 20-30 minutes. 
**If you cannot wait, please skip this step and use the existing files for the following steps.**

## Atomistic simulation via ASE 
PyXtal\_FF provides a built in `ASE` calculator. You can simply use it for some light weight calculations.
Below is an example of running NVT MD simulation for 1000 Si atoms.
```
$ python md.py -f Si-snap-zbl/16-16-checkpoint.pth 
MD simulation for  1000  atoms
Step:    0 [152.96]: Epot = -5.359eV  Ekin = 0.120eV (T=928K)  Etot = -5.239eV 
Step:    1 [ 77.42]: Epot = -5.330eV  Ekin = 0.094eV (T=724K)  Etot = -5.237eV 
Step:    2 [ 77.26]: Epot = -5.298eV  Ekin = 0.063eV (T=485K)  Etot = -5.235eV 
Step:    3 [ 77.00]: Epot = -5.277eV  Ekin = 0.044eV (T=338K)  Etot = -5.233eV 
Step:    4 [ 77.33]: Epot = -5.273eV  Ekin = 0.042eV (T=323K)  Etot = -5.231eV 
Step:    5 [ 77.01]: Epot = -5.281eV  Ekin = 0.051eV (T=398K)  Etot = -5.230eV 
```
The time cost is about 77 seconds for each time step.
For more types of simulations based on `Python-ASE`, checkout [this example](https://github.com/qzhu2017/PyXtal_FF/blob/master/pyxtal_ff/test_properties.py)

## LAMMPS Installation
Run the simulation with `Python` is not recommended for large scale systems. For practical simulation, you can plug in the force field to `LAMMPS` as follows.

First, follow the steps below to install LAMMPS-MLIAP and python wrapper

```
$ git clone https://github.com/pedroantoniosantosf/lammps.git
$ cd lammps/src
$ make yes-snap
$ make yes-so3  #experimental
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
  0  -39.460  -39.460    0.000    0.000    0.000
  1  -37.263  -37.263    0.000    0.001    0.000
  2  -39.422  -39.422    0.000    0.000    0.000
  3  -36.937  -36.937    0.000    0.001    0.000
  4  -37.184  -37.184    0.000    0.001    0.000
  5  -38.072  -38.072    0.000    0.000    0.000
  6  -37.702  -37.702    0.000    0.000    0.000
  7  -39.097  -39.097    0.000    0.000    0.000
  8  -39.407  -39.407    0.000    0.000    0.000
  9  -38.228  -38.228    0.000    0.000    0.000
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
zbl 2.0 4.0
pair_coeff 1 1 zbl 14.0 14.0
pair_coeff * * mliap Si
```

The `md.in` file gives an example to run NPT MD simulation at 500 K for 1000 Si atoms.
```
lmp_mpi < md.in > md.out
```
On a single CPU, it needs ~120 seconds to complete 1000 steps, which is abouot **700x** faster than the `ASE` caculator.

