import sys
import time
import numpy as np
import json
from pymatgen import Structure
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

sys.path.append("../")
from descriptors.snap import bispectrum
from utilities.assembler import assembler
import pandas as pd


# Read json file and convert them to pymatgen structure object 
# and the corresponding energies and volumes.

files = ["AIMD.json"] #"Elastic.json", "Surface.json", "Vacancy.json"]
profile = dict(Cu=dict(r=1.0, w=1.0))
Rc = 3.7
twojmax = 6
diagonal = 3
force = True

class Cu_bispectrum(object):
    """
    
    """
    def __init__(self, files, profile, Rc, twojmax, diagonal, force):
        self.files = files
        self.profile = profile
        self.Rc = Rc
        self.twojmax = twojmax
        self.diagonal = diagonal
        self.force = force
        
        self.structures, self.y = self.get_structures_energies()
        self.X = self.convert_to_bispectrum()
        
        
    def get_structures_energies(self):
        time0 = time.time()
        structures = []
        y = []
        self.volumes = []

        
        for file in files:
            with open("../datasets/Cu/training/"+file) as f:
                data = json.load(f)
        
            for struc in data[:5]:
                lat = struc['structure']['lattice']['matrix']
                species = []
                positions = []
                for site in struc['structure']['sites']:
                    species.append(site['label'])
                    positions.append(site['xyz'])
                structure = Structure(lat, species, positions)
                structures.append(structure)
                
                if self.force == False:
                    y.append(struc['outputs']['energy'])
                else:
                    y.append(struc['outputs']['energy'])
                    fs = np.ravel(struc['outputs']['forces'])
                    for f in fs:
                        y.append(f)

                self.volumes.append(structure.volume)
        
        time1 = time.time()
        t = round(time1 - time0, 2)
        print(f"This is the time it takes to convert json files to \
              structures: {t} s")
        
        return structures, y
    
    
    def convert_to_bispectrum(self):
        time0 = time.time()
        
        sna = []        
        for i in range(len(self.structures)):
            bispectrum(self.structures[i], self.Rc, self.twojmax, self.profile, 
                       diagonal=self.diagonal)
            bispec = assembler(atom_type=['Cu'], volume=self.volumes[i], 
                               force=self.force, stress=False)
            if sna == []:
                sna = bispec.bispectrum_coefficients
            else:
                sna = np.vstack((sna, bispec.bispectrum_coefficients))
        
        
        time1 = time.time()
        t = round(time1 - time0, 2)
        print(f"This is the time it takes to generate bispectrum coefficients \
              for {len(self.structures)} structures: {t} s")

        return sna


# Apply linear regression and evaluate



#X_train, X_test, y_train, y_test = train_test_split(sna, energies, test_size=0.4, random_state=13)
#
#lr_time0 = time.time()
#reg = LinearRegression().fit(X_train, y_train)
#predict_train = reg.predict(X_train)
#predict_test = reg.predict(X_test)
#lr_time1 = time.time()
#lr_time = round(lr_time1 - lr_time0, 2)
#print(f"This is the time it takes running the linear regression: {lr_time} s")
#
#r2_train = reg.score(X_train, y_train)
#mae_train = mean_absolute_error(y_train, predict_train)
#r2_test = reg.score(X_test, y_test)
#mae_test = mean_absolute_error(y_test, predict_test)


# Print

#d = {'train_r2': [r2_train], 
#     'train_mae': [mae_train],
#     'test_r2': [r2_test],
#     'test_mae': [mae_test]}
#
#df = pd.DataFrame(d)
#print(df)

if __name__ == '__main__':
    Cu_bispectrum(files, profile, Rc, twojmax, diagonal, force)
