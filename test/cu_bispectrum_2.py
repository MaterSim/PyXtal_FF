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


files = ["AIMD_NPT.json"] #["AIMD.json", "Elastic.json", "Surface.json", "Vacancy.json"]
profile = dict(Mo=dict(r=1.0, w=1.0))
Rc = 4.615858
twojmax = 6
diagonal = 3
force = True
save = False


class Cu_bispectrum(object):
    """
    
    """
    def __init__(self, files, profile, Rc, twojmax, diagonal, force, save=False):
        self.files = files
        self.profile = profile
        self.Rc = Rc
        self.twojmax = twojmax
        self.diagonal = diagonal
        self.force = force
        self.save = save
        
        self.structures, self.y = self.get_structures_energies()
        self.X = self.convert_to_bispectrum(self.save)
        self.linear_regression()
        
        
    def get_structures_energies(self):
        time0 = time.time()
        structures = []
        y = []
        self.volumes = []
        n_atoms = []

        
        for file in files:
            with open("../datasets/Mo/training/"+file) as f:
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
                
                if self.force == False:
                    y.append(struc['data']['energy_per_atom'])
                    n_atoms.append(structure.num_sites)
                else:
                    y.append(struc['data']['energy_per_atom'])
                    n_atoms.append(structure.num_sites)
                    fs = np.ravel(struc['data']['forces'])
                    for f in fs:
                        y.append(f)
                        n_atoms.append(1.)

                self.volumes.append(structure.volume)
        
        time1 = time.time()
        t = round(time1 - time0, 2)
        print(f"This is the time it takes to convert json files to \
              structures: {t} s")
        
        return structures, np.vstack((y, n_atoms))
    
    
    def convert_to_bispectrum(self, save):
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

        if save:            
            np.savetxt("bispectrum.csv", sna, delimiter=",")
            np.savetxt("output.csv", self.y, delimiter=",")

        
        return sna


    def linear_regression(self):
        time0 = time.time()
        ts = 0.4
        rs = 13
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y[0], test_size=ts, random_state=rs)
        X_train1, X_test1, y_train1, y_test1 = train_test_split(self.X, self.y[1], test_size=ts, random_state=rs)

        reg = LinearRegression().fit(X_train, y_train)
        
        # Training evalution
        X_forces_train, X_energies_train = [], []
        y_forces_train, y_energies_train = [], []
        n_atoms_train = []
        for i, natom in enumerate(y_train1):
            if natom == 1.:
                X_forces_train.append(X_train[i])
                y_forces_train.append(y_train[i])
            else:
                X_energies_train.append(X_train[i])
                y_energies_train.append(y_train[i])
                n_atoms_train.append(natom)
        print(n_atoms_train)
        yp_forces_train = reg.predict(X_forces_train)
        mae_forces_train = mean_absolute_error(y_forces_train, yp_forces_train)
        r2_forces_train = reg.score(X_forces_train, y_forces_train)
        yp_energies_train = reg.predict(X_energies_train)
        mae_energies_train = mean_absolute_error(y_energies_train, yp_energies_train)
        r2_energies_train = reg.score(X_energies_train, y_energies_train)
        
        print(mae_energies_train, r2_energies_train, mae_forces_train, r2_forces_train)

        # Training evalution
        X_forces_test, X_energies_test = [], []
        y_forces_test, y_energies_test = [], []
        n_atoms_test = []
        for i, natom in enumerate(y_test1):
            if natom == 1.:
                X_forces_test.append(X_test[i])
                y_forces_test.append(y_test[i])
            else:
                X_energies_test.append(X_test[i])
                y_energies_test.append(y_test[i])
                n_atoms_test.append(natom)
        yp_forces_test = reg.predict(X_forces_test)
        mae_forces_test = mean_absolute_error(y_forces_test, yp_forces_test)
        r2_forces_test = reg.score(X_forces_test, y_forces_test)
        yp_energies_test = reg.predict(X_energies_test)
        mae_energies_test = mean_absolute_error(np.asarray(y_energies_test)/np.asarray(n_atoms_test), np.asarray(yp_energies_test)/np.asarray(n_atoms_test))
        r2_energies_test = reg.score(X_energies_test, y_energies_test)

        print(mae_energies_test, r2_energies_test, mae_forces_test, r2_forces_test)


        #y_pred = reg.predict(X_test)
        #mae = mean_absolute_error(y_pred, y_test)
        #r2 = reg.score(X_test, y_test)
        #print(mae, r2)

        #d = {'train_r2': [r2_train],'train_mae': [mae_train], 'test_r2': [r2_test], 'test_mae': [mae_test]}
        #df = pd.DataFrame(d)
        #print(df)

        
if __name__ == '__main__':
    Cu_bispectrum(files, profile, Rc, twojmax, diagonal, force)
