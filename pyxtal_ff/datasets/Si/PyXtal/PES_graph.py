import matplotlib.pyplot as plt
import numpy as np
from pymatgen import Structure
from monty.serialization import loadfn

pyxtal_data = loadfn("Si_train.json")
#ucsd_train_data = loadfn("pyxtal_ff/datasets/Si/training.json")
#ucsd_test_data = loadfn("pyxtal_ff/datasets/Si/test.json")

pyxtal_energy = []
pyxtal_volume = []
#pyxtal_force = []
for i, d in enumerate(pyxtal_data):
    lat = d['lattice']
    species = d['elements']
    coords = d['coords']
    struc = Structure(lat, species, coords)
    pyxtal_energy.append(d['energy']/len(d['elements']))
    pyxtal_volume.append(struc.volume/len(d['elements']))

#u_train_energy = []
#u_train_volume = []
#for i, d in enumerate(ucsd_train_data):
#    u_train_energy.append(d['outputs']['energy']/len(d['structure']))
#    u_train_volume.append(d['structure'].volume/len(d['structure']))

#u_test_energy = []
#u_test_volume = []
#for i, d in enumerate(ucsd_test_data):
#    u_test_energy.append(d['outputs']['energy']/len(d['structure']))
#    u_test_volume.append(d['structure'].volume/len(d['structure']))


#plt.scatter(u_test_volume, u_test_energy, s=15, label='UCSD_Test', color='red')
plt.scatter(pyxtal_volume, pyxtal_energy, s=7, label='PyXtal', color='k')
#plt.scatter(u_train_volume, u_train_energy, s=7, color='deepskyblue', label='UCSD_Train')
plt.xlabel('Volume/atom (A\N{SUPERSCRIPT THREE}/atom)')
plt.ylabel('Energy/atom (eV/atom)')
plt.legend(loc='lower right')
plt.show()
