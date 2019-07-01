## PyXtal FF

FF-MAPP is an open-source Python library for developing machine learning interatomic potential of materials. In the current state, there are two models available: 
- [SNAP](https://www.sciencedirect.com/science/article/pii/S0021999114008353?via%3Dihub)
- [Neural Network](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.98.146401)

## Dependencies
1. [LAMMPS](https://lammps.sandia.gov/doc/Install.html)
2. [ASE](https://wiki.fysik.dtu.dk/ase/)
3. [Pymatgen](https://pymatgen.org/)

## Test Run
```python runSnap.py```

## Result
```
   energy_r2  energy_mae  force_r2  force_mae
0   0.999781    0.005192  0.976695   0.201534
Running time: 102.16s
```

**This is an ongoing project.**
