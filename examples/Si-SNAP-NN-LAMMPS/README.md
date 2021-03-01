# FF Training for Si set

## data source:
https://github.com/materialsvirtuallab/mlearn/tree/master/data/Si

## Training
```
python snap_train.py
```

## LAMMPS calculation
After training, you expect to find the trained weights in `weight.txt`. This file can be used by lammps to conduct a large scale MD simulation by following the steps below.

### Install the LAMMPS-MLIAP
```
git clone https://github.com/pedroantoniosantosf/lammps.git
cd lammps/src
make yes-mliap
make yes-snap
make mpi
```

### Run LAMMPS through unix file
Create an input file by specifying 

```
pair_style mliap model nn Cu.mliap.model descriptor sna Cu.mliap.descriptor
```

Below is an input lammps file for NPT MD simulation at 500 K.
```
lmp_mpi < in.md > out.md
```

### Run LAMMPS through Python wrapper
