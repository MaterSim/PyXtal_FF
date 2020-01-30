import sys
import time
import json
import numpy as np
from monty.serialization import loadfn, MontyEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
np.set_printoptions(threshold=sys.maxsize)

class PR:
    """ Atom-centered Polynomial Regression (PR) model. PR uses 
    linear regression to predict the energy and forces with Bispectrum or 
    BehlerParrinello descriptors as the imput values. 
    A machine learning interatomic potential can developed by optimizing the 
    weights of the PR for a given system.
        
    Parameters
    ----------
    elements: list
         A list of atomic species in the crystal system.
    force_coefficient: float
        This parameter is used in the penalty function to scale the force
        contribution relative to the energy.
    order: int
        The order of the polynomial. Order 1 is for linear and order 2 is for 
        quadratic.
    path: str
        The path of the directory where everything is saved.
    """
    def __init__(self, elements, force_coefficient, order, path, N_max=None):

        self.w = force_coefficient
        self.elements = sorted(elements)
        self.order = order
        if order == 1:
            self.quadratic = False
        else:
            self.quadratic = True
        self.path = path
        self.N_max = N_max

    def train(self, TrainDescriptors, TrainFeatures, use_force=None):
        """ Training of Polynomial Regression with Scikit-Learn. """

        if use_force is None:
            # create the default value
            if 'dxdr' in TrainDescriptors[0]:
                self.use_force = True
            else:
                self.use_force = False
        else:
            if use_force and 'dxdr' not in TrainDescriptors[0]:
                msg = "use_force cannot be True if dxdr is not present"
                raise ValueError(msg)
            else:
                self.use_force = use_force

        if self.N_max is None:
            no_of_descriptors = len(TrainDescriptors[0]['x'][0])
        else: 
            no_of_descriptors = min(len(TrainDescriptors[0]['x'][0]), self.N_max)
        print("\n")
        print("========================== Training ==========================")
        print(f"Order: {self.order}")
        print(f"No_of_descriptors: {no_of_descriptors}")
        print(f"No_of_structures: {len(TrainDescriptors)}")
        print(f"use_force: {self.use_force}")
        if self.use_force:
            print(f"force_coeff: {self.w}")
        print("\n")

        t0 = time.time()
        X = self.parse_descriptors(TrainDescriptors)
        y, w = self.parse_features(TrainDescriptors, TrainFeatures)
                
        regression = LinearRegression(fit_intercept=False)
        self.regression = regression.fit(X, y, w)
        self.coef_ = self.regression.coef_
        #self.intercept_ = self.regression.intercept_
        self.export_parameters()
        t1 = time.time()
        
        print("The training time: {:.2f} s".format(t1-t0))
        print(f"\nThe training results is exported in {self.path}.")
        print("==================== Training is Completed ===================")


    def export_parameters(self):
        d = {}
        
        params = ['elements', 'w', 'path', 'quadratic', 'coef_']
        for param in params:
            d[param] = eval('self.'+param)

        filename = self.path+'PolyReg-parameters.json'
        with open(filename, 'w') as f:
            json.dump(d, f, indent=2, cls=MontyEncoder)


    def load_parameters(self, filename=None):
        """ Load linear regression coefficients from json file. """
        parameters = loadfn(self.path+"PolyReg-parameters.json")
        
        return parameters

    def calculate_energy_forces(self, descriptor):
        """
        A routine to compute energy and forces only.
        Used for the purpose of evalution.
        See detailed math in documentation

        Args:
        -----------
        x: 2D array
            input descriptors
        dxdr: 4d array (optional)
            the primes of inputs

        Returns:
        --------
        energy: float, 
            the predicted energy
        forces: 2D array [N_atom, 3] (if dxdr is provided)
            the predicted forces
        """
        no_of_atoms = len(descriptor['x'])
        parameters = self.load_parameters()
        self.quadratic = parameters['quadratic']
        
        X = self.parse_descriptors([descriptor])
        y = np.dot(X, parameters['coef_'])
        energy = y[0]
        if self.use_force:
            force = np.reshape(y[1:], (no_of_atoms, 3))
        else:
            force = None
        
        return energy, force


    def evaluate(self, descriptors, features, figname):
        """ Evaluating the train or test data set. """

        energy, force = [], [] # true
        _energy, _force = [], [] # predicted
 
        print("======================== Evaluating ==========================")
        for i in range(len(descriptors)):
            no_of_atoms = len(features[i]['force'])
            nnEnergy, nnForce = self.calculate_energy_forces(descriptors[i])
            #nnEnergy /= no_of_atoms #already averaged
            
            # Store energy into list
            true_energy = features[i]['energy'] / no_of_atoms
            energy.append(true_energy)
            _energy.append(nnEnergy)
            if self.use_force:
                true_force = np.ravel(features[i]['force'])
                nnForce = np.ravel(nnForce)
                for m in range(len(true_force)):
                    force.append(true_force[m])
                    _force.append(nnForce[m])
 
        E_mae = mean_absolute_error(energy, _energy)
        E_mse = mean_squared_error(energy, _energy)
        E_r2 = r2_score(energy, _energy)
        print("Energy R2  : {:8.6f}".format(E_r2))
        print("Energy MAE : {:8.6f}".format(E_mae))
        print("Energy MSE : {:8.6f}".format(E_mse))

        if self.use_force:
            F_mae = mean_absolute_error(force, _force)
            F_mse = mean_squared_error(force, _force)
            F_r2 = r2_score(force, _force)
            print("Force R2   : {:8.6f}".format(F_r2))
            print("Force MAE  : {:8.6f}".format(F_mae))
            print("Force MSE  : {:8.6f}".format(F_mse))
        else:
            F_mae = None
            F_mse = None
            F_r2 = None
        return (E_mae, E_mse, E_r2, F_mae, F_mse, F_r2)


    def parse_descriptors(self, descriptors):
        """ Parse descriptors to Linear Regression format. """
        no_of_structures = len(descriptors)
        #no_of_descriptors = len(descriptors[0]['x'][0])
        if self.N_max is None:
            no_of_descriptors = len(descriptors[0]['x'][0])
        else: 
            no_of_descriptors = min(len(descriptors[0]['x'][0]), self.N_max)
       
        atom_types = self.elements
        
        X = None
        for i in range(no_of_structures):
            descriptors[i]['x'] = descriptors[i]['x'][:, :no_of_descriptors]
            descriptors[i]['dxdr'] = descriptors[i]['dxdr'][:, :, :no_of_descriptors, :]
            if self.quadratic:
                # (d^2 + 3*d) / 2
                total_descriptors = int((no_of_descriptors**2+3*no_of_descriptors)/2) 
                x = np.zeros((len(descriptors[i]['x']), total_descriptors))
                dxdr = np.zeros((len(descriptors[i]['x']),
                                 len(descriptors[i]['x']),
                                 total_descriptors,
                                 3))
            
                x[:, :no_of_descriptors] += descriptors[i]['x']
                dxdr[:, :, :no_of_descriptors, :] += descriptors[i]['dxdr']

                dcount = no_of_descriptors
                
                # self term for x
                x_square = 0.5 * (descriptors[i]['x'] ** 2)
                
                # self term for dxdr
                dxdr_square = np.einsum('ijkl,ik->ijkl',
                                        descriptors[i]['dxdr'],
                                        descriptors[i]['x'])
                
                for d1 in range(no_of_descriptors):
                    # Cross term for x
                    x_d1_d2 = np.einsum('i, ij->ij', descriptors[i]['x'][:, d1],
                                        descriptors[i]['x'][:, (d1+1):])

                    # Cross term for dxdr
                    dxdr_d1_d2 = np.einsum('ijl,ik->ijkl',
                                           descriptors[i]['dxdr'][:, :, d1, :],
                                           descriptors[i]['x'][:, d1+1:]) + \
                                 np.einsum('ijkl,i->ijkl',
                                           descriptors[i]['dxdr'][:, :, d1+1:, :],
                                           descriptors[i]['x'][:, d1])
                    
                    
                    # Append for x and dxdr
                    x[:, dcount] += x_square[:, d1]
                    dxdr[:, :, dcount, :] += dxdr_square[:, :, d1, :]
                    dcount += 1
                    
                    x[:, dcount:dcount+len(x_d1_d2[0])] += x_d1_d2
                    dxdr[:, :, dcount:dcount+len(x_d1_d2[0]), :] += dxdr_d1_d2
                    dcount += len(x_d1_d2[0])
                
            else:
                # Linear term only
                total_descriptors = no_of_descriptors
                x = descriptors[i]['x']
                dxdr = descriptors[i]['dxdr']
            
            elements = descriptors[i]['elements']

            bias_weights = np.zeros((len(atom_types), 1))
            bias_weights += 1.0/len(atom_types)
            
            sna = np.zeros((len(atom_types), 1+total_descriptors))
            _sna = {}
            _count = {}
            for atom in atom_types:
                _sna[atom] = None
                _count[atom] = 0
                
            for e, element in enumerate(elements):
                if _sna[element] is None:
                    _sna[element] = x[e][:total_descriptors]
                else:
                    _sna[element] += x[e][:total_descriptors]
                _count[element] += 1
            
            for a, atom in enumerate(atom_types):
                _sna[element] /= _count[element]
                temp = np.hstack((bias_weights[a], _sna[element]))
                sna[a, :] += temp
            sna = sna.reshape(1, sna.size)
                
            if X is None:
                X = sna
            else:
                X = np.concatenate((X, sna))

            # This sum is for 1 element type; doesn't work for SiO2.
            if self.use_force:
                #_dxdr = np.einsum('ijkl->jkl', dxdr)
                _dxdr = -1 * np.einsum('ijkl->jkl', dxdr[:,:,:total_descriptors,:])
                count = 0
                snad = np.zeros((np.shape(_dxdr)[0]*np.shape(_dxdr)[2], 
                                 1+total_descriptors))
                for j in range(_dxdr.shape[0]):
                    for k in range(_dxdr.shape[2]):
                        snad[count, 1:] += _dxdr[j, :, k]
                        count += 1
                X = np.concatenate((X, snad))
                
        return X


    def parse_features(self, descriptors, features):
        """ Parse features to Linear Regression format. """
        
        no_of_structures = len(features)
        y = None #store the features
        w = None #weight of each sample
        for i in range(no_of_structures):
            energy = features[i]['energy']/len(descriptors[i]['x'])
            if self.use_force:
                energy = np.array([energy])
                w_energy = np.array([1.])
                force = np.array(features[i]['force']).ravel()
                w_force = np.array([self.w]*len(force))

                if y is None:
                    y = np.concatenate((energy, force))
                    w = np.concatenate((w_energy, w_force))
                else:
                    y = np.concatenate((y, energy, force))
                    w = np.concatenate((w, w_energy, w_force))
            else:
                if y is None:
                    y = [energy]
                    w = [1]
                else:
                    y.append(energy)
                    w.append(1)
                    
        return y, w
