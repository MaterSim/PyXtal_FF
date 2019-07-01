import time
import json
import numpy as np

from pymatgen import Structure
from mapp.models.snap import Snap

# The only parameters needed to be changed
directory = "mapp/datasets/Mo/training/"
files = ["AIMD_NVT.json"]
profile = dict(Mo=dict(r=0.50, w=1.0))
# Rc, force weights, force coefficient, stress weight, stress coeff
Rc = 4.615858
bounds = [(0.0001, 0.1)] 
optimizer_kwargs = {'strategy': 'best1bin', 'popsize': 15}


##################### Do not change!!!! ############################
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
Predictor = Snap(element_profile=profile, optimizer_kwargs=optimizer_kwargs, Rc=Rc)
Predictor.fit(structures=structures, 
              features=y, 
              feature_styles=styles, 
              bounds=bounds,)
t1 = time.time()
print(f"Running time: {round(t1-t0, 2)}s")
