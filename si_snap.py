import time
import json
import numpy as np

import matplotlib.pyplot as plt

from pymatgen import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from mapp.models.snap import Snap

# The only parameters needed to be changed
directory = "mapp/datasets/Si/"
files = ["Si.json"]
profile = dict(Si=dict(r=0.75, w=1.0))
# Rc, force weights, force coefficient, stress weight, stress coeff
Rc = 4.9
bounds = [(0.00001, 0.1)] 
optimizer_kwargs = {'strategy': 'best1bin', 'popsize': 15}


##################### Do not change!!!! ############################
for file in files:
    with open(directory+file) as f:
        datas = json.load(f)

structures = []
y = []
styles = []

for struc in datas:
    lat = struc['lattice']['data']
    species = []
    positions = []
    for i in range(len(struc['coords']['data'])):
        species.append(struc['elements'][i])
        positions.append(struc['coords']['data'][i])
    #print(positions)

    #fcoords = np.dot(positions, np.linalg.inv(lat))
    structure = Structure(lat, species, positions)
    #sa = SpacegroupAnalyzer(structure,)
    #sg = sa.get_space_group_number()
    #if sg == 227:

    structures.append(structure)

    # append energies in y
    y.append((struc['energy']/len(struc['coords']['data'])))
    styles.append('energy')

    fs = np.ravel(struc['force'])
    for f in fs:
        y.append(f)
        styles.append('force')

X = "Bispectrum.txt"

# Perform SNAP
t0 = time.time()
Predictor = Snap(element_profile=profile, optimizer_kwargs=optimizer_kwargs, Rc=Rc)
Predictor.fit(structures=structures, 
              features=y, 
              feature_styles=styles, 
              bounds=bounds,
              X=X)
t1 = time.time()
print(f"Running time: {round(t1-t0, 2)}s")

optimized_parameters = Predictor.result
print(optimized_parameters)
