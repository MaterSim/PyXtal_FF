#!/usr/bin/env  python
# encoding: utf-8
import sys, os
import time
import json
import numpy as np
from scipy.linalg import lstsq
from monty.serialization import loadfn, MontyEncoder
np.set_printoptions(threshold=sys.maxsize)

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mticker
plt.style.use("ggplot")


class PR():
    """ Atom-centered Polynomial Regression (PR) model. PR utilizes
    linear regression to predict the energy and forces based on 
    the atom-centered descriptors as the input values.
        
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
    alpha: float
        L2 penalty (regularization term) parameter.
    norm: int (*PR)
        This argument defines a model to calculate the regularization
        term. It takes only 1 or 2 as its value: Manhattan or Euclidean 
        norm, respectively. If alpha is None, norm is ignored.
    d_max: int
        The maximum number of descriptors (d) used in Linear Regression model.
    """
    def __init__(self, elements, force_coefficient, order, path, alpha, norm,
                 d_max=None):
        self.force_coefficient = force_coefficient
        self.elements = sorted(elements)
        self.order = order
        if order == 1:
            self.quadratic = False
        elif order == 2:
            self.quadratic = True
        else:
            raise ValueError("Order must be 1 or 2")
        self.path = path

        self.alpha = alpha
        self.norm = norm
        self.d_max = d_max
        self.filename = os.path.join(self.path, 'PolyReg-parameters.json')
        self.unit = 'eV'


    def train(self, TrainDescriptors, TrainFeatures):
        """ Fitting Linear Regression model. """

        # d_max is the total number of descriptors used.
        if self.d_max is None:
            self.d_max = len(TrainDescriptors[0]['x'][0])
        else:
            # d_max has to be less or equal than total descriptors.
            assert self.d_max <= len(TrainDescriptors[0]['x'][0]),\
                    "d_max is larger than total descriptors."

        print("==================================== Training ====================================")
        print("\n")
        print(f"Order: {self.order}")
        print(f"No_of_descriptors: {self.d_max}")
        print(f"No_of_structures: {len(TrainDescriptors)}")
        print(f"force_coeff: {self.force_coefficient}")
        print(f"alpha: {self.alpha}")
        print(f"norm: {self.norm}\n")

        t0 = time.time()
        y, w = self.parse_features(TrainFeatures)
        X = self.parse_descriptors(TrainDescriptors)
        
        self.coef_ = self.LinearRegression(X, y, w, self.alpha, self.norm)
        
        t1 = time.time()
        print("The training time: {:.2f} s".format(t1-t0))
        
        self.export_parameters()
        print(f"\nThe training results is exported in {self.path}.")
        print("\n============================== Training is Completed =============================\n")
    
    
    def evaluate(self, descriptors, features, figname):
        """ Evaluating the train or test data set. """

        energy, force = [], [] # true
        _energy, _force = [], [] # predicted
 
        # If-else for consistent separations in printing.
        if figname[:-4] == 'Train':
            print(f"============================= Evaluating {figname[:-4]}ing Set ============================\n")
        else:
            print("============================= Evaluating Testing Set =============================\n")

        for i in range(len(descriptors)):
            no_of_atoms = len(features[i]['force'])
            nnEnergy, nnForce = self.calculate_energy_forces(descriptors[i])
            
            # Store energy into list
            true_energy = features[i]['energy'] / no_of_atoms
            energy.append(true_energy)
            _energy.append(nnEnergy)
            if self.force_coefficient:
                true_force = np.ravel(features[i]['force'])
                nnForce = np.ravel(nnForce)
                for m in range(len(true_force)):
                    force.append(true_force[m])
                    _force.append(nnForce[m])

        energy, force = np.asarray(energy), np.asarray(force)
        _energy, _force = np.asarray(_energy), np.asarray(_force)
        
        # Calculate the statistical metrics for energy.
        E_mae = self.mean_absolute_error(energy, _energy)
        E_mse = self.mean_squared_error(energy, _energy)
        E_r2 = self.r2_score(energy, _energy)
        print("Energy R2  : {:8.6f}".format(E_r2))
        print("Energy MAE : {:8.6f}".format(E_mae))
        print("Energy MSE : {:8.6f}".format(E_mse))

        # Plotting the energy results.
        energy_str = 'Energy: r2({:.4f}), MAE({:.4f} {}/atom)'. \
                     format(E_r2, E_mae, self.unit)
        plt.title(energy_str)
        plt.scatter(energy, _energy, label='Energy', s=5)
        plt.legend(loc=2)
        plt.xlabel('True ({}/atom)'.format(self.unit))
        plt.ylabel('Prediction ({}/atom)'.format(self.unit))
        plt.tight_layout()
        plt.savefig(self.path+'Energy_'+figname)
        plt.close()
        print("The energy figure is exported to: {:s}".format(self.path+'Energy_'+figname))
        print("\n")

        if self.force_coefficient:
            F_mae = self.mean_absolute_error(force, _force)
            F_mse = self.mean_squared_error(force, _force)
            F_r2 = self.r2_score(force, _force)
            print("Force R2   : {:8.6f}".format(F_r2))
            print("Force MAE  : {:8.6f}".format(F_mae))
            print("Force MSE  : {:8.6f}".format(F_mse))

            # Plotting the forces results.
            length = 'A'
            if self.unit == 'Ha':
                length == 'Bohr'
            force_str = 'Force: r2({:.4f}), MAE({:.3f} {}/{})'. \
                        format(F_r2, F_mae, self.unit, length)
            plt.title(force_str)
            plt.scatter(force, _force, s=5, label='Force')
            plt.legend(loc=2)
            plt.xlabel('True ({}/{})'.format(self.unit, length))
            plt.ylabel('Prediction ({}/{})'.format(self.unit, length))
            plt.tight_layout()
            plt.savefig(self.path+'Force_'+figname)
            plt.close()
            print("The force figure is exported to: {:s}".format(self.path+'Force_'+figname))
            print("\n")

        else:
            F_mae, F_mse, F_r2 = None, None, None

        print("============================= Evaluation is Completed ============================")
        print("\n")

        return (E_mae, E_mse, E_r2, F_mae, F_mse, F_r2)


    def LinearRegression(self, X, y, w, alpha, norm=2):
        """ Perform linear regression. """
        m = X.shape[1] # The shape of the descriptors

        _X = X * np.sqrt(np.expand_dims(w, axis=1))
        _y = y * np.sqrt(w)

        if self.alpha:
            if norm == 1:
                theta = np.linalg.lstsq(_X.T.dot(_X), _X.T.dot(_y) - alpha, rcond=None)[0]
            elif norm == 2:
                theta = np.linalg.lstsq(_X.T.dot(_X) + alpha * np.identity(m), _X.T.dot(_y), rcond=None)[0]
            else:
                msg = f"Regularization with {norm} norm is not implemented yet."
                raise NotImplementedError(msg)
        else:
            theta = lstsq(_X, _y, cond=None)[0]
            #theta = np.linalg.lstsq(_X, _y, rcond=None)[0]
        
        return theta


    def export_parameters(self):
        """ Save parameters to a json file. """
        d = {}
        
        params = ['elements', 'force_coefficient', 'path', 'quadratic', 'coef_']
        for param in params:
            d[param] = eval('self.'+param)

        with open(self.filename, 'w') as f:
            json.dump(d, f, indent=2, cls=MontyEncoder)


    def load_parameters(self, filename=None):
        """ Load linear regression parameters from json file. """
        if filename is None:
            filename = self.filename
        parameters = loadfn(filename)
        return parameters


    def calculate_energy_forces(self, descriptor):
        """
        A routine to compute energy and forces.

        Parameters:
        -----------
        descriptor: list
            list of x and dxdr (optional).

        Returns:
        --------
        energy: float, 
            the predicted energy
        forces: 2D array [N_atom, 3] (if dxdr is provided)
            the predicted forces
        """
        no_of_atoms = len(descriptor['x'])
        parameters = self.load_parameters()
        
        X = self.parse_descriptors([descriptor], train=False)
        _y = np.dot(X, parameters['coef_'])

        energy = _y[0]
        if self.force_coefficient:
            force = np.reshape(_y[1:], (no_of_atoms, 3))
        else:
            force = None
        
        return energy, force


    def mean_absolute_error(self, true, predicted):
        """ Calculate mean absolute error of energy or force. """
        return sum(abs(true-predicted)/len(true))


    def mean_squared_error(self, true, predicted):
        """ Calculate mean square error of energy or force. """
        return np.sqrt(sum((true-predicted) ** 2 /len(true)))


    def r2_score(self, true, predicted):
        """ Calculate the r square of energy or force. """
        t_bar = sum(true)/len(true)
        square_error = sum((true-predicted) ** 2)
        true_variance = sum((true-t_bar) ** 2)
        return 1 - square_error / true_variance

    
    def parse_descriptors(self, descriptors, train=True):
        """ Parse descriptors and its gradient to 2-D array. 
        
        Returns
        -------
        X: 2-D array [n+m*3, d]
            d is the total number of descriptors, n is the total
            number of structures, and m is the total number atoms
            in the entire structures. If force_coefficient is None,
            X has the shape of [n, d].
        """
        if train:
            no_of_structures = self.no_of_structures
            no_of_atoms = self.no_of_atoms
        else:
            no_of_structures = len(descriptors)
            no_of_atoms = len(descriptors[0]['x'])

        # Determine the total number of descriptors based on SNAP or qSNAP.
        # Note: d_max != self.d_max
        if self.d_max is None:
            self.d_max = len(descriptors[0]['x'][0])

        if self.quadratic:
            d_max = (self.d_max**2+3*self.d_max)//2
        else:
            d_max = self.d_max
        
        # Determine the size of X.
        if self.force_coefficient:
            X = np.zeros([no_of_structures+no_of_atoms*3,
                          (1+d_max)*len(self.elements)])
        else:
            X = np.zeros([no_of_structures, (1+d_max)*len(self.elements)])
        
        # Fill in X.
        xcount = 0
        for i in range(no_of_structures):
            _x = descriptors[i]['x'][:, :self.d_max]
            _dxdr = descriptors[i]['dxdr'][:, :, :self.d_max, :]

            if self.quadratic:
                # Total descriptors: (d^2 + 3*d) / 2
                
                x = np.zeros((len(_x), d_max))
                dxdr = np.zeros((len(_x), len(_x), d_max, 3))

                x[:, :self.d_max] += _x
                dxdr[:, :, :self.d_max, :] += _dxdr
            
                # self-term for x and dxdr
                x_square = 0.5 * _x ** 2
                dxdr_square = np.einsum('ijkl,ik->ijkl', _dxdr, _x)
                
                dcount = self.d_max
                for d1 in range(self.d_max):
                    # Cross term for x and dxdr
                    x_d1_d2 = np.einsum('i, ij->ij', _x[:, d1], _x[:, d1+1:])
                    dxdr_d1_d2 = np.einsum('ijl,ik->ijkl', _dxdr[:, :, d1, :], _x[:, d1+1:]) + \
                                 np.einsum('ijkl,i->ijkl', _dxdr[:, :, d1+1:, :], _x[:, d1])
                    
                    # Append for x and dxdr
                    x[:, dcount] += x_square[:, d1]
                    dxdr[:, :, dcount, :] += dxdr_square[:, :, d1, :]
                    dcount += 1
                    
                    x[:, dcount:dcount+len(x_d1_d2[0])] += x_d1_d2
                    dxdr[:, :, dcount:dcount+len(x_d1_d2[0]), :] += dxdr_d1_d2
                    dcount += len(x_d1_d2[0])
                
            else:
                x = descriptors[i]['x'][:, :d_max]
                dxdr = descriptors[i]['dxdr'][:, :, :d_max, :]
            
            elements = descriptors[i]['elements']
            
            # Arranging x and dxdr for energy and forces.
            bias_weights = 1.0/len(self.elements)
            
            sna = np.zeros((len(self.elements), 1+d_max))
            snad = np.zeros((len(self.elements), len(_x), 1+d_max, 3))
            _sna = {}
            _snad = {}
            _count = {}
            for element in self.elements:
                _sna[element] = None
                _snad[element] = None
                _count[element] = 0
            
            # Loop over the number of atoms in a structure.
            for e, element in enumerate(elements):
                if _sna[element] is None:
                    _sna[element] = x[e]
                    if self.force_coefficient:
                        _snad[element] = -1 * dxdr[e]
                else:
                    _sna[element] += x[e]
                    if self.force_coefficient:
                        _snad[element] += -1 * dxdr[e]
                _count[element] += 1

            for e, element in enumerate(self.elements):
                if _count[element] > 0:
                    _sna[element] /= _count[element]
                    sna[e, :] += np.hstack(([bias_weights], _sna[element]))
                
                    if self.force_coefficient:
                        snad[e, :, 1:, :] += _snad[element]

            # X for energy
            X[xcount, :] += sna.ravel()
            xcount += 1

            # X for forces.
            if self.force_coefficient:
                for j in range(snad.shape[1]):
                    for k in range(snad.shape[3]):
                        X[xcount, :] += snad[:, j, :, k].ravel()
                        xcount += 1
                            
        return X


    def parse_features(self, features):
        """ Parse features (energy and forces) into 1-D array.
        
        Returns
        -------
        y: 1-D array [n+m*3,]
            y contains the energy and forces of structures in 1-D array.
            If force_coefficient is None, y has the shape of [n,].

        w: 1-D array [n+m*3,]
            w contains the relative importance between energy and forces.
            If force_coefficient is None, w has the shape of [n,].
        """
        self.no_of_structures = len(features)
        self.no_of_atoms = 0

        y = None #store the features (energy+force)
        w = None #weight of each sample
        for i in range(len(features)):
            energy = features[i]['energy']/len(features[i]['force'])
            
            if self.force_coefficient:
                energy = np.array([energy])
                w_energy = np.array([1.])
                force = np.array(features[i]['force']).ravel()
                w_force = np.array([self.force_coefficient]*len(force))

                if y is None:
                    y = np.concatenate((energy, force))
                    w = np.concatenate((w_energy, w_force))
                else:
                    y = np.concatenate((y, energy, force))
                    w = np.concatenate((w, w_energy, w_force))
                
                # Count the number of atoms for the entire structures.
                self.no_of_atoms += len(features[i]['force'])
            
            else:
                if y is None:
                    y = [energy]
                    w = [1]
                else:
                    y.append(energy)
                    w.append(1)
                    
        return np.asarray(y), np.asarray(w)
