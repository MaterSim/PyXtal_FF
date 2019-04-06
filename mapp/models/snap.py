import sys
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error



sys.path.append("..")

from descriptors.bispectrum import Bispectrum
from utilities.assembler import Assembler
from utilities.gregression import Regressor



class Snap:
    """
    The class that performs machine learning techniques to predict interatomic potentials.
    
    Parameters
    ----------
    energy_coefficient: float
        The energy scaling parameter to evaluate the loss value.
    force_coefficinet: float
        The force scaling parameter to evaluate the loss value.
    stress_coefficient: float
        The stress scaling parameter to evaluate the loss value.
    """
    def __init__(self, element_profile, twojmax=6, diagonal=3, rfac0=0.99363, rmin0=0.0, energy_coefficient=1., force_coefficient=0.03, stress_coefficient=None):
        self.profile = element_profile
        self.twojmax = twojmax
        self.diagonal = diagonal
        self.rfac0 = rfac0
        self.rmin0 = rmin0
        self.energy_coefficient = energy_coefficient
        self.force_coefficient = force_coefficient
        self.stress_coefficient = stress_coefficient
        self.atom_types = list(self.profile.keys())


    def fit(self, structures, features, feature_styles, bounds, force=True, stress=False):
        """
        Run the SNAP fitting with linear regression and a global optimization algorithm.

        Parameters
        ----------
        structures: list
            Pymatgen crystal structure object.
        feature: array
            An array of the energy, force and stress (if applicable).
        feature_style: list
            List of the str identifying the feature whether if it's energy, force, or stress.
        bounds: list of tuples
            Define the linear regression parameters bounds. The parameters must consist of:
            - Rc (radial cutoff)
            - Energy weight
            - Force weight
            - Stress weight (if and only if the stress_coefficient is not None.
        """
        self.structures = structures
        self.y = features
        self.styles = feature_styles
        self.bounds = bounds
        self.force = force
        self.volumes = []
        for structure in self.structures:
            self.volumes.append(structure.volume)

        if stress == True and self.stress_coefficient == None:
            msg = "You must input the stress coefficient in snap"
            raise ValueError(msg)
        elif stress == True and self.stress_coefficient != None:
            self.stress = stress
            self.volumes = []
            for structure in self.structures:
                self.volumes.append(structure.volume)
        else:
            self.stress = stress


        # Perform the SNAP regression model (i.e. Linear Regression and Global Optimization method)
        self.regressor = Regressor()
        self.result = self.regressor.regress(model=self, bounds=self.bounds)


    def get_coefficients(self):
        """
        Get the linearly fitted coefficients.
        """
        coeff = {}

        coeff['intercept'] = [self.reg.intercept_]
        coeff['slope'] = reg.coef_

        return coeff


    def calculate_loss(self, parameters, lossprime=False):
        """
        Calculating the loss in the Linear Regression prediction scheme. 
        """
        loss = 0.
        
        # Get bispectrum coefficients with the initial structures and the predicted Rc
        self.X = []
        for i in range(len(self.structures)):
            Bispectrum(self.structures[i], parameters[0], self.profile, twojmax=self.twojmax, diagonal=self.diagonal, rfac0=self.rfac0, rmin0=self.rmin0)
            bispec = Assembler(atom_type=self.atom_types, volume=self.volumes[i],
                               force=self.force, stress=self.stress)
        
            if self.X == []:
                self.X = bispec.bispectrum_coefficients
            else:
                self.X = np.vstack((self.X, bispec.bispectrum_coefficients))


        # Construct the weights into an array based on the features.
        self.w = []
        for style in self.styles:
            if style == 'energy':
                self.w.append(parameters[1])
            elif style == 'force':
                self.w.append(parameters[2])
            elif style == 'stress':
                self.w.append(parameters[3])
            else:
                raise NotImplementedError(f"This {style} is not acceptable")

        # Separate energies, forces, and stress for MAE and r2 evaluations.
        X_forces, X_energies = [], []
        y_forces, y_energies = [], []
        w_forces, w_energies = [], []
        
        for i in range(len(self.w)):
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
        regression = LinearRegression().fit(self.X, self.y, self.w)
        
        
        # Calculate the MAE here.
        yp_energies = regression.predict(X_energies)
        mae_energies = mean_absolute_error(y_energies, yp_energies)
        r2_energies = regression.score(X_energies, y_energies, w_energies)

        yp_forces = regression.predict(X_forces)
        mae_forces = mean_absolute_error(y_forces, yp_forces)
        r2_forces = regression.score(X_forces, y_forces, w_forces)

        # Evaluate loss

        loss = self.energy_coefficient * mae_energies + self.force_coefficient * mae_forces

        return loss


    def get_mae_rsquare(self, X, y, w, styles):
        """
        Calculate the mae and rsquare of energies, forces, and stress (if applicable).
        """
        X1, X2, X3 = [], [], []
        y1, y2, y3 = [], [], []
        w1, w2, w3 = [], [], []

        for i, style in styles:
            if style == 'energy':
                X1.append(X[i])
                y1.append(y[i])
                w1.append(w[i])
            elif style == 'force':
                X2.append(X[i])
                y2.append(y[i])
                w2.append(w[i])
            else:
                X3.append(X[i])
                y3.append(y[i])
                w3.append(w[i])

        # Evaluate the mae and r square of energy
        y1_ = self.reg.predict(X1)
        mae1 = mean_absolute_error(y1, y1_)
        rsquare1 = reg.score(X1, y1, w1)

        # Evaluate the mae and r square of force
        y2_ = self.reg.predict(X2)
        mae2 = mean_absolute_error(y2, y2_)
        rsquare2 = reg.score(X2, y2, w2)

        # Evaluate the mae and r square of stress
        if self.stress == True:
            y3_ = self.reg.predict(X3)
            mae3 = mean_absolute_error(y3, y3_)
            rsquare3 = reg.score(X3, y3, w3)

            result = {'energy_r2': [rsquare1],
                      'energy_mae': [mae1],
                      'force_r2': [rsquare2],
                      'force_mae': [mae2],
                      'stress_r2': [rsquare3],
                      'stress_mae': [mae3]}
        
            return result

        else:
            result = {'energy_r2': [rsquare1],
                      'energy_mae': [mae1],
                      'force_r2': [rsquare2],
                      'force_mae': [mae2]}

            return result
