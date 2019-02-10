import sys
import time
import numpy as np
import json
from pymatgen import Structure

sys.path.append("../")
from descriptors.snap import bispectrum
from utilities.assembler import assembler


# Read json file and convert them to pymatgen structure object 
# and the corresponding energies and volumes.

with open("../datasets/Cu/training/AIMD.json") as f:
    data = json.load(f)

structures = []
energies = []
volumes = []

files = ["AIMD.json", "Elastic.json", "Surface.json", "Vacancy.json"]

convert_data0 = time.time()
for file in files:
    with open("../datasets/Cu/training/"+file) as f:
        data = json.load(f)

    for struc in data:
        lat = struc['structure']['lattice']['matrix']
        species = []
        positions = []
        for site in struc['structure']['sites']:
            species.append(site['label'])
            positions.append(site['xyz'])

        structure = Structure(lat, species, positions)
    
        structures.append(structure)
        energies.append(struc['outputs']['energy'])
        volumes.append(structure.volume)
convert_data1 = time.time()
convert_data = round(convert_data1 - convert_data0, 2)
print(f"This is the time it takes to convert json files to structures: {convert_data} s")

n_samples = len(structures)

# Get bispectrum coefficients for all structures

convert_bispectrum0 = time.time()
sna = []

profile = dict(Cu=dict(r=1.0, w=1.0))

for i in range(len(structures)):
    bispectrum(structures[i], 4.6, 6, profile, diagonal=3)
    bispec = assembler(atom_type=['Cu'], volume=volumes[i], force=False, stress=False)
    if sna == []:
        sna = bispec.bispectrum_coefficients
    else:
        sna = np.vstack((sna, bispec.bispectrum_coefficients))

convert_bispectrum1 = time.time()
bispectrum_gen = round(convert_bispectrum1 - convert_bispectrum0, 2)
print(f"This is the time it takes to generate bispectrum coefficients for {n_samples} structures: {bispectrum_gen} s")

# Apply linear regression and evaluate

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(sna, energies, test_size=0.4, random_state=13)

lr_time0 = time.time()
reg = LinearRegression().fit(X_train, y_train)
predict_train = reg.predict(X_train)
predict_test = reg.predict(X_test)
lr_time1 = time.time()
lr_time = round(lr_time1 - lr_time0, 2)
print(f"This is the time it takes running the linear regression: {lr_time} s")

r2_train = reg.score(X_train, y_train)
mae_train = mean_absolute_error(y_train, predict_train)
r2_test = reg.score(X_test, y_test)
mae_test = mean_absolute_error(y_test, predict_test)


# Print
import pandas as pd

d = {'train_r2': [r2_train], 
     'train_mae': [mae_train],
     'test_r2': [r2_test],
     'test_mae': [mae_test]}

df = pd.DataFrame(d)
print(df)
