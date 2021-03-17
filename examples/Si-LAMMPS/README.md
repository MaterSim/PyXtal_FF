# Example of SNAP\_NN model for elemental silicon

This is an example to train the neural networks model based on the SNAP/SO3 descriptor based on an online [dataset](https://github.com/materialsvirtuallab/mlearn/tree/master/data/Si)

## Force field training
```
python snap_train.py
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
$ python md.py -f Si-snap/16-16-checkpoint.pth 
MD simulation for  1000  atoms
Step:    0 [ 27.83]: Epot = -5.423eV  Ekin = 0.038eV (T=294K)  Etot = -5.385eV 
Step:    1 [ 14.67]: Epot = -5.423eV  Ekin = 0.038eV (T=291K)  Etot = -5.385eV 
Step:    2 [ 14.63]: Epot = -5.422eV  Ekin = 0.037eV (T=284K)  Etot = -5.385eV 
Step:    3 [ 14.62]: Epot = -5.421eV  Ekin = 0.036eV (T=276K)  Etot = -5.385eV 
Step:    4 [ 14.68]: Epot = -5.420eV  Ekin = 0.034eV (T=265K)  Etot = -5.385eV 
Step:    5 [ 14.62]: Epot = -5.418eV  Ekin = 0.033eV (T=253K)  Etot = -5.385eV 
Step:    6 [ 14.64]: Epot = -5.416eV  Ekin = 0.031eV (T=239K)  Etot = -5.385eV 
Step:    7 [ 14.68]: Epot = -5.414eV  Ekin = 0.029eV (T=224K)  Etot = -5.385eV 
Step:    8 [ 14.62]: Epot = -5.412eV  Ekin = 0.027eV (T=208K)  Etot = -5.385eV 
Step:    9 [ 14.65]: Epot = -5.410eV  Ekin = 0.025eV (T=192K)  Etot = -5.385eV 
```
The time cost is about 14.6 seconds for each time step.
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

The `validate.py` will create a randomly perturbed silicon diamond structure. Then it will perform a single point energy calculation from both `Python-ASE` and `LAMMPS` calcuator.

```
$ python validate.py 

ASE calculator with pyxtal_ff force field

Energy:  -37.879 eV/atom
Forces (eV/A)
[[  7.0649   0.0000   0.0000]
 [ -1.6986  -1.4435  -1.4435]
 [  0.6213  -0.0000   0.0000]
 [ -1.6986   1.4435   1.4435]
 [ -0.4587  -0.0000   0.0000]
 [ -1.6857   2.2228  -2.2228]
 [ -0.4587  -0.0000  -0.0000]
 [ -1.6857  -2.2228   2.2228]]
Stresses (GPa)
[ -0.3004  -0.3308  -0.3308   0.0495   0.0000   0.0000]

LAMMPS calculator with pyxtal_ff force field

Energy:  -37.879 eV/atom
Forces (eV/A)
[[  7.0649   0.0000  -0.0000]
 [ -1.6986  -1.4435  -1.4435]
 [  0.6213  -0.0000   0.0000]
 [ -1.6986   1.4435   1.4435]
 [ -0.4587   0.0000   0.0000]
 [ -1.6857   2.2228  -2.2228]
 [ -0.4587  -0.0000  -0.0000]
 [ -1.6857  -2.2228   2.2228]]
Stresses (GPa)
[ -0.3004  -0.3308  -0.3308   0.0495  -0.0000  -0.0000]
```
If you cannot get the similar results, please check your `LAMMPS` installation.


## Fast simulation via LAMMPS

Finally, you can run LAMMPS directly. 
It will work as any usual `LAMMPS` calculation, except that you need to specify a new `pair style`
```
pair_style mliap model nn Si-snap/NN_weights.txt descriptor sna Si-snap/DescriptorParams.txt
pair_coeff * * Si Si
```

The `md.in` file gives an example to run NPT MD simulation at 500 K for 1000 Si atoms.
```
lmp_mpi < md.in > md.out
```
On a single CPU, it needs ~120 seconds to complete 1000 steps, which is abouot **120x** faster than the `ASE` caculator.

