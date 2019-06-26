import sys
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

#sys.path.append("..")
from ..descriptors.bispectrum import Bispectrum, Assembler
from ..utilities.regression import gRegressor as Regressor


class Snap:
    """A class that performs machine learning techniques to describe the local
    environment of each atom using bispectrum components as the descriptor.
    These machine learning techniques are called Snap model.
    
    Parameters
    ----------
    element_profile: dict
        Elemental descriptions of each atom type in the structure.
        i.e. dict(Na=dict(r=0.3, w=0.9), Cl=dict(r=0.7, w=3.0)).
    twojmax: int
        Band limit for bispectrum components.
    diagonal: int
        diagonal value = 0 or 1 or 2 or 3.
        0 = all j1, j2, j <= twojmax, j2 <= j1
        1 = subset satisfying j1 == j2
        2 = subset satisfying j1 == j2 == j3
        3 = subset satisfying j2 <= j1 <= j        
    rfac0: float
        Parameter in distance to angle conversion (0 < rcutfac < 1).
        Default value: 0.99363.
    rmin0: float
        Parameter in distance to angle conversion.
        Default value: 0.
    energy_coefficient: float
        The energy scaling parameter to evaluate the loss value.
    optimizer: str
        Choose the desired global optimization scheme.
        - 'DifferentialEvolution'
        - add 'BasinHopping'
    optimizer_kwargs: dict
        The parameters for the global optimization scheme.
        i.e. {'strategy': 'best1bin'}
    """
    def __init__(self, element_profile, twojmax=8, diagonal=3, rfac0=0.99363, 
                 Rc=5., rmin0=0.0, optimizer='DifferentialEvolution', 
                 optimizer_kwargs=None):
        self.profile = element_profile
        self.twojmax = twojmax
        self.diagonal = diagonal
        self.rfac0 = rfac0
        self.Rc = Rc
        self.rmin0 = rmin0
        
        # Global optimization
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs

        self.atom_types = list(self.profile.keys()) # ['Na', 'Cl']


    def fit(self, structures, features, feature_styles, bounds, 
            force=True, stress=False, save=False, X=None):
        """Run the Snap fitting with linear regression and a global 
        optimization algorithm.

        Parameters
        ----------
        structures: list
            List of Pymatgen crystal structure object.
        feature: array
            An array of the energy, force and stress (if applicable).
        feature_style: list of str
            For identifying the feature whether it's energy, force, or stress.
        bounds: list of tuples
            Defining the machine learning parameters bounds. The parameters 
            must consist of:
            - Rc (radial cutoff)
            - Force weight*
            - Force coefficient*
            - Stress weight**
            - Stress coefficient**
        force: bool
            If True, perform Snap fitting with force.
        stress: bool
            If True, perform Snap fitting with stress.

        *  If force is True.
        ** If stress if True.
        """
        self.structures = structures
        self.y = features
        self.styles = feature_styles
        self.bounds = bounds
        self.force = force
        self.stress = stress
        
        # Calculate the volume for each structure
        self.volumes = []
        for structure in self.structures:
            self.volumes.append(structure.volume)
        
        if X == None:
            t0 = time.time()
            print("Calculating Bispectrum")
            # Bispectrum calculation
            self.X = []
            for i in range(len(self.structures)):
                Bispectrum(self.structures[i], self.Rc, self.profile, twojmax=self.twojmax, diagonal=self.diagonal, rfac0=self.rfac0,rmin0=self.rmin0)
                b = Assembler(atom_type=self.atom_types, volume=self.volumes[i], stress=self.stress)
                if self.X == []:
                    self.X = b.bispectrum_coefficients
                else:
                    self.X = np.vstack((self.X, b.bispectrum_coefficients))
            t1 = time.time()
            print("Calculating Bispectrum Done!")
            print(f"Bispectrum calculation time: {t1-t0}s")
        else:
            self.X = np.loadtxt(X)

        if save:
            np.savetxt("Bispectrum.txt", self.X)

        # Perform the SNAP model
        self.regressor = Regressor(method=self.optimizer, 
                                   user_kwargs=self.optimizer_kwargs)
        self.result = self.regressor.regress(model=self, bounds=self.bounds)
        
        # Print result
        self.print_mae_r2square()
        
        # Save the Bispectrum coefficients
        self.save_regression_coefficients()
    
    
    def calculate_loss(self, parameters):
        """Calculating the loss in the Linear Regression scheme.
        
        Parameters
        ----------
        parameters: list
            A vectorized list provided by Scipy global optimizer.
        
        Returns
        -------
        float
            The loss value.
        """
        loss = 0.
        
        # Construct the weights into an array based on the features.
        self.w = []
        for style in self.styles:
            if style == 'energy':
                self.w.append(1.)
            elif style == 'force':
                self.w.append(parameters[0])
            elif style == 'stress':
                if self.force == True:
                    self.w.append(parameters[1])
                else:
                    self.w.append(parameters[0])
            else:
                raise NotImplementedError(f"This {style} is not acceptable")

        # Separate energies, forces, and stress for MAE and r2 evaluations.
        X_energies, X_forces, X_stress = [], [], []
        y_energies, y_forces, y_stress = [], [], []
        w_energies, w_forces, w_stress = [], [], []
        
        for i in range(len(self.X)):
            if self.styles[i] == 'energy':
                X_energies.append(self.X[i])
                y_energies.append(self.y[i])
                w_energies.append(self.w[i])
            elif self.styles[i] == 'force':
                X_forces.append(self.X[i])
                y_forces.append(self.y[i])
                w_forces.append(self.w[i])
            elif self.styles[i] == 'stress':
                X_stress.append(self.X[i])
                y_stress.append(self.y[i])
                w_stress.append(self.w[i])
                
        # Perform the linear regression here.
        regression = LinearRegression(fit_intercept=False)
        self.regression = regression.fit(self.X, self.y, self.w)
        
        # Calculate the MAE here.
        self.yp_energies = self.regression.predict(X_energies)
        self.mae_energies = mean_absolute_error(y_energies,
                                                self.yp_energies)
        self.r2_energies = self.regression.score(X_energies, y_energies,
                                                 w_energies)

        if self.force == True:
            self.yp_forces = self.regression.predict(X_forces)
            self.mae_forces = mean_absolute_error(y_forces, 
                                                  self.yp_forces)
            self.r2_forces = self.regression.score(X_forces, y_forces,
                                                   w_forces)
        else:
            self.mae_forces = 0.
            self.r2_forces = None

        if self.stress == True:
            self.yp_stress = self.regression.predict(X_stress)
            self.mae_stress = mean_absolute_error(y_stress, self.yp_stress)
            self.r2_stress = self.regression.score(X_stress, y_stress, 
                                                   w_stress)
        else:
            self.mae_stress = 0.
            self.r2_stress = None

        # Evaluate loss
        loss = self.mae_energies
        loss += parameters[0] * self.mae_forces
        if self.force == True and self.stress == True:
            loss += parameters[1] * self.mae_stress
        elif self.force == False and self.stress == True:
            loss += parameters[0] * self.mae_stress

        return loss


    def save_regression_coefficients(self,):
        """Saving the bispectrum coefficients to a textfile."""
        self.coef = {}
        coef = self.regression.coef_
        split_len = len(self.atom_types)
        coef_ = np.split(coef, split_len)
        
        filename = ''
        for i, atype in enumerate(self.atom_types):
            self.coef[atype] = coef_[i]
            filename += atype
        
        f = open(filename+".snapcoeff", "a")
        f.write("# SNAP coefficients for "+filename+"\n\n")
        f.write(f"{len(self.atom_types)} {int(len(coef)/split_len)}\n")
        for k, v in self.profile.items():
            f.write(k+" ")
            for key, value in v.items():
                f.write(str(value)+" ")
            f.write("\n")
            for c in self.coef[k]:
                f.write(str(c))
                f.write("\n")
        f.close()

        
    def print_mae_r2square(self,):
        """Print the evalution criteria such as mae and r2 values"""
        if self.stress:
            d = {'energy_r2': [self.r2_energies], 
                 'energy_mae': [self.mae_energies], 
                 'force_r2': [self.r2_forces], 
                 'force_mae': [self.mae_forces],
                 'stress_r2': [self.r2_stress],
                 'stress_mae': [self.mae_stress]}
        else:
            d = {'energy_r2': [self.r2_energies], 
                 'energy_mae': [self.mae_energies], 
                 'force_r2': [self.r2_forces], 
                 'force_mae': [self.mae_forces]}
        
        df = pd.DataFrame(d)

        print(df)
        

########################## Not needed? ####################################
#    def get_coefficients(self):
#        """
#        Get the linearly fitted coefficients.
#        """
#        coeff = {}
#
#        coeff['intercept'] = [self.reg.intercept_]
#        coeff['slope'] = reg.coef_
#
#        return coeff
#    
#        def get_mae_rsquare(self, X, y, w, styles):
#        """
#        Calculate the mae and rsquare of energies, forces, and stress (if applicable).
#        """
#        X1, X2, X3 = [], [], []
#        y1, y2, y3 = [], [], []
#        w1, w2, w3 = [], [], []
#
#        for i, style in styles:
#            if style == 'energy':
#                X1.append(X[i])
#                y1.append(y[i])
#                w1.append(w[i])
#            elif style == 'force':
#                X2.append(X[i])
#                y2.append(y[i])
#                w2.append(w[i])
#            else:
#                X3.append(X[i])
#                y3.append(y[i])
#                w3.append(w[i])
#
#        # Evaluate the mae and r square of energy
#        y1_ = self.reg.predict(X1)
#        mae1 = mean_absolute_error(y1, y1_)
#        rsquare1 = reg.score(X1, y1, w1)
#
#        # Evaluate the mae and r square of force
#        y2_ = self.reg.predict(X2)
#        mae2 = mean_absolute_error(y2, y2_)
#        rsquare2 = reg.score(X2, y2, w2)
#
#        # Evaluate the mae and r square of stress
#        if self.stress == True:
#            y3_ = self.reg.predict(X3)
#            mae3 = mean_absolute_error(y3, y3_)
#            rsquare3 = reg.score(X3, y3, w3)
#
#            result = {'energy_r2': [rsquare1],
#                      'energy_mae': [mae1],
#                      'force_r2': [rsquare2],
#                      'force_mae': [mae2],
#                      'stress_r2': [rsquare3],
#                      'stress_mae': [mae3]}
#        
#            return result
#
#        else:
#            result = {'energy_r2': [rsquare1],
#                      'energy_mae': [mae1],
#                      'force_r2': [rsquare2],
#                      'force_mae': [mae2]}
#
#            return result
