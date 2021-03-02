# FF Training for Si set

## data source:
https://github.com/materialsvirtuallab/mlearn/tree/master/data/Si

## Training
```
python snap_train.py
```

## LAMMPS calculation
After training is complete, you expect to find 
- `DescriptorParams.txt`: parameters to compute the descriptor
- `NN_weights.txt`: weight parameters of the model

These file can be used by `LAMMPS` to conduct a large scale MD simulation by following the steps below.

### 00. Installation of LAMMPS-MLIAP
```
git clone https://github.com/pedroantoniosantosf/lammps.git
cd lammps/src
make yes-mliap
make yes-snap
make mpi
```

After the installation, you can run LAMMPS via two ways
- one time run through LAMMPS command
- more complicated workflow though LAMMPS-Python wrapper

### 01. Run LAMMPS through unix file
Create an input file by specifying 

```
pair_style mliap model nn Cu.mliap.model descriptor sna Cu.mliap.descriptor
```

Below is an input lammps file for NPT MD simulation at 500 K.
```
lmp_mpi < in.md > out.md
```

### 02. Run LAMMPS through Python wrapper
