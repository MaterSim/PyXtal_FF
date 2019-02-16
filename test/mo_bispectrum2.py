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

directory = "../datasets/Mo/training/"
files = ["AIMD_NPT.json", "AIMD_NVT.json"]#, "Elastic.json", "GB.json", "Surface.json", "Vacancy.json"]
profile = dict(Mo=dict(r=1.0, w=1.0))
Rc = 4.615858
twojmax = 6
diagonal = 3
force = True
stress = False
save = False
w_energy = 1537.72250
w_force = 1.61654910
w_stress = 0.

########################### From UCSD #########################################


class Cu_bispectrum(object):
    """
    
    """
    def __init__(self, files, profile, Rc, twojmax, diagonal, 
                 w_energy, w_force, w_stress, 
                 force=False, stress=False, save=False):
        self.files = files
        self.profile = profile
        self.Rc = Rc
        self.twojmax = twojmax
        self.diagonal = diagonal
        self.w_energy = w_energy
        self.w_force = w_force
        self.w_stress = w_stress
        self.force = force
        self.stress = stress
        self.save = save
        
#        self.structures, self.y = self.get_structures_energies()
#        self.X = self.convert_to_bispectrum(self.save)
        self.UCSD()
        self.linear_regression()
    
    
    def UCSD(self):
        
        for file in files:
            with open(directory+file) as f:
                datas = json.load(f)
                
        X = []
        y = []
        n_atoms = []
        w = []
        for i, data in enumerate(datas):
            y.append(data['data']['energy_per_atom'])
            w.append(data['weights']['energy'])
            n_atoms.append(49.)
            fs = np.ravel(data['data']['forces'])
            for f in fs:
                y.append(f)
                w.append(data['weights']['force'])
                n_atoms.append(1.)
                
            e_des = data['optimized parameters']['bispectrum coefficients']['energy descriptor']
            f_des = data['optimized parameters']['bispectrum coefficients']['force descriptor']
                
            if i == 0:
                self.X = np.vstack((e_des, f_des))
                
            else:
                x = np.vstack((e_des, f_des))
                self.X = np.vstack((self.X,x))
                
        self.y = np.vstack((y,w,n_atoms))        
        
        
    def get_structures_energies(self):
        time0 = time.time()
        structures = []
        y = []
        self.volumes = []
        n_atoms = []
        weights = []

        
        for file in files:
            with open(directory+file) as f:
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
                
                # append energies in y
                y.append(struc['data']['energy_per_atom'])
                n_atoms.append(structure.num_sites)
                weights.append(self.w_energy)
                
                # append force in y
                if self.force == True:                    
                    fs = np.ravel(struc['data']['forces'])
                    for f in fs:
                        y.append(f)
                        n_atoms.append(1.)
                        weights.append(self.w_force)
                    # append stress in y
                    if self.stress == True:
                        ss = np.ravel(struc['data']['virial_stress'])
                        for s in ss:
                            y.append(s)
                            n_atoms.append(1.)
                            weights.append(self.w_stress)

                self.volumes.append(structure.volume)

        time1 = time.time()
        t = round(time1 - time0, 2)
        print(f"The time it takes to convert json files to structures: {t} s")

        return structures, np.vstack((y, n_atoms, weights))
    
    
    def convert_to_bispectrum(self, save):
        time0 = time.time()
        
        snap = []        
        for i in range(len(self.structures)):
            bispectrum(self.structures[i], self.Rc, self.twojmax, self.profile,
                       diagonal=self.diagonal)
            bispec = assembler(atom_type=['Cu'], volume=self.volumes[i], 
                               force=self.force, stress=self.stress)
            if snap == []:
                snap = bispec.bispectrum_coefficients
            else:
                snap = np.vstack((snap, bispec.bispectrum_coefficients))
        
        
        time1 = time.time()
        t = round(time1 - time0, 2)
        print(f"The time it takes to generate bispectrum coefficients \
              for {len(self.structures)} structures: {t} s")
        
        return snap


    def linear_regression(self):
        """
        perform linear regression
        """
        time0 = time.time()
        ts = 0.2
        rs = 103


        X_train, X_test, y_train, y_test = train_test_split(self.X, 
                                                            self.y[0], 
                                                            test_size=ts, 
                                                            random_state=rs)
        # To obtain # of atoms in a unit cell
        _, _, n_atoms_train, n_atoms_test = train_test_split(self.X, 
                                                             self.y[2], 
                                                             test_size=ts, 
                                                             random_state=rs)
        # To obtain weights
        _, _, weights_train, weights_test = train_test_split(self.X,
                                                           self.y[1], 
                                                           test_size=ts, 
                                                           random_state=rs)

        reg = LinearRegression().fit(X_train, y_train, weights_train)
        
        # Evaluate training dataset
        mae_E_train, r2_E_train, mae_F_train, r2_F_train = \
            self.evaluate_mae_rsquare(reg, n_atoms_train, X_train, y_train,
                                      weights_train)
            
        # Evaluate test dataset
        mae_E_test, r2_E_test, mae_F_test, r2_F_test = \
            self.evaluate_mae_rsquare(reg, n_atoms_test, X_test, y_test,
                                      weights_test)
            
        # Print train
        print(f"Score for training dataset")
        d_train = {'energy_r2': [r2_E_train], 
                   'energy_mae': [mae_E_train], 
                   'force_r2': [r2_F_train], 
                   'force_mae': [mae_F_train]}
        df_train = pd.DataFrame(d_train)
        print(df_train)
        
        # Print test
        print(f"Score for test dataset")
        d_test = {'energy_r2': [r2_E_test], 
                   'energy_mae': [mae_E_test], 
                   'force_r2': [r2_F_test], 
                   'force_mae': [mae_F_test]}
        df_test = pd.DataFrame(d_test)
        print(df_test)
        
        time1 = time.time()
        t = round(time1 - time0, 2)
        print(f"The time it takes to perform linear regression: {t} s")
        
        
    def evaluate_mae_rsquare(self, regression, n_atoms, X, y, weights):
        """
        Evaluate the train or test dataset.
        
        Returns
        -------
             mean absolute error and r2 for energies and force.
        """
        X_forces, X_energies = [], []
        y_forces, y_energies = [], []
        w_forces, w_energies = [], []
        natoms = []

        for i, atom in enumerate(n_atoms):
            if atom == 1.:
                X_forces.append(X[i])
                y_forces.append(y[i])
                w_forces.append(weights[i])
            else:
                X_energies.append(X[i])
                y_energies.append(y[i])
                natoms.append(atom)
                w_energies.append(weights[i])
                
        # Evaluate energy
        yp_energies = regression.predict(X_energies)
        mae_energies = mean_absolute_error(y_energies, yp_energies)
        r2_energies = regression.score(X_energies, y_energies, w_energies)

        # Evaluate force
        yp_forces = regression.predict(X_forces)
        mae_forces = mean_absolute_error(y_forces, yp_forces)
        r2_forces = regression.score(X_forces, y_forces, w_forces)
        
        return mae_energies, r2_energies, mae_forces, r2_forces        

        
if __name__ == '__main__':
    Cu_bispectrum(files, profile, Rc, twojmax, diagonal, 
                  w_energy, w_force, w_stress, force, stress, save)
