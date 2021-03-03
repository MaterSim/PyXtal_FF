# A Tutorial for Running a PyXtal\_FF Potential (SNAP) with LAMMPS

## Step 1
One will need to train their potentials using PyXtal\_FF. A script to run PyXtal\_FF code has been prepared in [PyXtalFF.py](https://github.com/qzhu2017/PyXtal_FF/blob/master/examples/LAMMPS/PyXtalFF.py). 

In the script, neural networks with an architecture of 30-16-16-1 is initiated to train [Si dataset](https://github.com/materialsvirtuallab/mlearn/tree/master/data/Si). The 30 inputs (`lmax = 3`) are constructed with **SNAP descriptor**, which is equivalent to the SNAP descriptor used in LAMMPS. It should be emphasized that `lmax = 3` in PyXtal\_FF is equivalent to `twojmax = 6` in LAMMPS.

To train the neural networks with Si dataset, one can execute the following:
```
python PyXtalFF.py
```

Once the training is finished, PyXtal\_FF will generate two files in the `Si-snap` directory: **DescriptorParams.txt** and **NN_weights.txt**. Both files are necessary to perform neural networks prediction with LAMMPS software. DescriptorParams.txt contains the descriptors information, while the NN\_weights.txt contains neural networks parameters along with the scaling factors.

## Step 2
One will need to install LAMMPS software:
```
git clone https://github.com/pedroantoniosantosf/lammps.git
cd lammps/src
make yes-mliap
make yes-snap
make mpi
```

## Step 3
Ensure this command is included into your LAMMPS input file to run neural networks with SNAP descriptor:
```
pair_style mliap model nn Si-snap/NN_weights.txt descriptor sna Si-snap/DescriptorParams.txt
```

Typically, there are 2 ways to run LAMMPS:
1. Run LAMMPS through unix file. Below is an input lammps file for NPT MD simulation at 500 K:
```
lmp_mpi < in.md > out.md
```

2. Run LAMMPS through Python wrapper
