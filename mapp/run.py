import time
import json
import numpy as np

from pymatgen import Structure
from models.snap import Snap


directory = "./datasets/Mo/training/"
files = ["AIMD_NPT.json"]
profile = dict(Mo=dict(r=0.5, w=1.0))
bounds = [(4, 5), (1000,2000), (0.1, 10), (500, 2000), (0.1,10)]


for file in files:
    with open(directory+file) as f:
        datas = json.load(f)

structures = []
y = []
styles = []

for struc in datas:
    lat = struc['structure']['lattice']['matrix']
    species = []
    positions = []
    for site in struc['structure']['sites']:
        species.append(site['label'])
        positions.append(site['xyz'])
    fcoords = np.dot(positions, np.linalg.inv(lat))
    structure = Structure(lat, species, fcoords)
    structures.append(structure)

    # append energies in y
    y.append(struc['data']['energy_per_atom'])
    styles.append('energy')

    fs = np.ravel(struc['data']['forces'])
    for f in fs:
        y.append(f)
        styles.append('force')

# Perform SNAP
t0 = time.time()
Predictor = Snap(element_profile=profile)
Predictor.fit(structures=structures, 
              features=y, 
              feature_styles=styles, 
              bounds=bounds)
t1 = time.time()
print(f"Running time: {round(t1-t0, 2)}s")

optimized_parameters = Predictor.result
print(optimized_parameters)
