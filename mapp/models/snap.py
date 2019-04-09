import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

sys.path.append("..")
from descriptors.bispectrum import Bispectrum
from utilities.assembler import Assembler
from utilities.gregression import Regressor


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
    force_coefficinet: float
        The force scaling parameter to evaluate the loss value.
    stress_coefficient: float
        The stress scaling parameter to evaluate the loss value.
    optimizer: str
        Choose the desired global optimization scheme.
        - 'DifferentialEvolution'
        - 'BasinHopping'
    optimizer_kwargs: dict
        The parameters for the global optimization scheme.
        i.e. {'strategy': 'best1bin'}
    """
    def __init__(self, element_profile, twojmax=6, diagonal=3, rfac0=0.99363, 
                 rmin0=0.0, energy_coefficient=1., force_coefficient=0.03, 
                 stress_coefficient=None, optimizer='DifferentialEvolution', 
                 optimizer_kwargs=None):
        self.profile = element_profile
        self.twojmax = twojmax
        self.diagonal = diagonal
        self.rfac0 = rfac0
        self.rmin0 = rmin0
        self.energy_coefficient = energy_coefficient
        self.force_coefficient = force_coefficient
        self.stress_coefficient = stress_coefficient
        
        # Global optimization
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs

        # Eventually we want to develop kwargs for:
        # 1. Global optimization arguments
        # 2. Linear regression arguments
        # 3. Bispectrum arguments. i.e. include electrostatics potential
        
        self.atom_types = list(self.profile.keys())


    def fit(self, structures, features, feature_styles, bounds, stress=False,
            save=True):
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
            - Energy weight
            - Force weight
            - Stress weight (Included if the stress_coefficient is not None)
        stress: bool
            If True, stress_coefficient must be included.
        """
        self.structures = structures
        self.y = features
        self.styles = feature_styles
        self.bounds = bounds
        
        # Calculate the volume for each structure
        self.volumes = []
        for structure in self.structures:
            self.volumes.append(structure.volume)

        if stress == True and self.stress_coefficient == None:
            msg = "You must input the stress coefficient in snap"
            raise ValueError(msg)
        elif stress == True and self.stress_coefficient != None:
            if len(bounds) == 4:
                pass
            else:
                msg = "The bounds doesn't match. " \
                        "Please check if you included all the necessary "\
                        "parameters."
                raise ValueError(msg)
        else:
            self.stress = stress

        # Perform the SNAP model
        self.regressor = Regressor(method=self.optimizer, 
                                   user_kwargs=self.optimizer_kwargs)
        self.result = self.regressor.regress(model=self, bounds=self.bounds)
        
        if save:
            self.save_to_textfile()
    
    
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
        
        # Get bispectrum coefficients with the initial structures 
        # and the predicted Rc.
        self.X = []
        for i in range(len(self.structures)):
            Bispectrum(self.structures[i], parameters[0], 
                       self.profile, twojmax=self.twojmax, 
                       diagonal=self.diagonal, rfac0=self.rfac0, 
                       rmin0=self.rmin0)
            bispec = Assembler(atom_type=self.atom_types, 
                               volume=self.volumes[i], stress=self.stress)
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
        X_energies, X_forces, X_stress  = [], [], []
        y_energies, y_forces, y_stress = [], [], []
        w_energies, w_forces, w_stress = [], [], []
        
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
        self.regression = LinearRegression().fit(self.X, self.y, self.w)
        
        # Calculate the MAE here.
        if self.stress == False:
            self.yp_energies = self.regression.predict(X_energies)
            self.mae_energies = mean_absolute_error(y_energies, 
                                                    self.yp_energies)
            self.r2_energies = self.regression.score(X_energies, y_energies, 
                                                     w_energies)
    
            self.yp_forces = self.regression.predict(X_forces)
            self.mae_forces = mean_absolute_error(y_forces, 
                                                  self.yp_forces)
            self.r2_forces = self.regression.score(X_forces, y_forces, 
                                                   w_forces)
            
            # Evaluate loss
            loss = self.energy_coefficient * self.mae_energies
            loss += self.force_coefficient * self.mae_forces
            
        else:
            self.yp_energies = self.regression.predict(X_energies)
            self.mae_energies = mean_absolute_error(y_energies, 
                                                    self.yp_energies)
            self.r2_energies = self.regression.score(X_energies, y_energies, 
                                                     w_energies)
    
            self.yp_forces = self.regression.predict(X_forces)
            self.mae_forces = mean_absolute_error(y_forces, 
                                                  self.yp_forces)
            self.r2_forces = self.regression.score(X_forces, y_forces, 
                                                   w_forces)
            
            self.yp_stress = self.regression.predict(X_stress)
            self.mae_stress = mean_absolute_error(y_stress, self.yp_stress)
            self.r2_stress = self.regression.score(X_stress, y_stress, 
                                                   w_stress)

            # Evaluate loss
            loss = self.energy_coefficient * self.mae_energies 
            loss += self.force_coefficient * self.mae_forces
            loss += self.stress_coefficient * self.mae_stress

        return loss


    def save_to_textfile(self,):
        """Saving the bispectrum coefficients to a textfile."""
        self.coeff = {}
        
        self.coeff['intercept'] = [self.regression.intercept_]
        self.coeff['slope'] = self.regression.coef_
        
        coeff = [self.regression.intercept_] + self.regression.coef_
        
        filename = ''
        for atype in self.atom_types:
            filename += atype
        
        f = open(filename+".snapcoeff", "a")
        f = open(filename+".snapcoeff", "a")
        f.write("# SNAP coefficients for "+filename+"\n\n")
        f.write(f"{len(self.atom_types)} {len(coeff)}\n")
        for k, v in self.profile.items():
            f.write(k+" ")
            for key, value in v.items():
                f.write(str(value)+" ")
            f.write("\n")
            for c in coeff:
                f.write(str(c))
                f.write("\n")
        f.close()

        
    def print_mae_r2square(self,):
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
