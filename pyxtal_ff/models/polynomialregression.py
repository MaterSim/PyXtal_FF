#!/usr/bin/env  python
# encoding: utf-8
import gc
import sys, os
import time
import json
import shelve
import numpy as np
from torch import save, load
np.set_printoptions(threshold=sys.maxsize)

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mticker
plt.style.use("ggplot")

from pyxtal_ff.utilities.elements import Element
eV2GPa = 160.21766

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
    stress_coefficient: float
        This parameter is used as the balance parameter scaling
        the stress contribution relative to the energy.
    stress_group: list of strings
        Only the intended group will be considered in stress training.
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
    def __init__(self, elements, force_coefficient, stress_coefficient, 
                 stress_group, order, path, alpha, norm, d_max=None):
        self.force_coefficient = force_coefficient
        self.stress_coefficient = stress_coefficient
        self.stress_group = stress_group
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
        self.unit = 'eV'


    def train(self, TrainData, optimizer):
        """ Fitting Linear Regression model. """
        db = shelve.open(self.path+TrainData)
        self.no_of_structures = len(list(db.keys()))

        # d_max is the total number of descriptors used.
        if self.d_max is None:
            self.d_max = db['0']['x'].shape[1]
        else:
            # d_max has to be less or equal than total descriptors.
            assert self.d_max <= len(db['0']['x'].shape[1]),\
                    "d_max is larger than total descriptors."

        if self.stress_coefficient and (self.stress_group is None):
            sg = []
            for i in range(self.no_of_structures):
                if db[str(i)]['group'] not in sg:
                    sg.append(db[str(i)]['group'])
            self.stress_group = sg

        db.close()
        
        print(f"Order              : {self.order}")
        if self.order == 1:
            print(f"No of parameters   : {self.d_max+1}")
        else:
            print(f"No of parameters   : {(self.d_max+1)*(self.d_max+2)//2}")
        print(f"No of structures   : {self.no_of_structures}")
        print(f"Force_coefficient  : {self.force_coefficient}")
        print(f"Stress_coefficient : {self.stress_coefficient}")
        print(f"alpha              : {self.alpha}")
        print(f"norm               : {self.norm}\n")

        t0 = time.time()
        y, w = self.parse_features(TrainData)

        X = self.parse_descriptors(TrainData,
            fc=self.force_coefficient, sc=self.stress_coefficient)
        
        self.coef_ = self.LinearRegression(X, y, w, self.alpha, self.norm)
        
        t1 = time.time()
        print("The training time: {:.2f} s".format(t1-t0))
        
    
    def evaluate(self, data, figname):
        """ Evaluating the train or test data set. """
        db = shelve.open(self.path+data)
        
        energy, force, stress = [], [], [] # true
        _energy, _force, _stress = [], [], [] # predicted
 
        for i in range(len(list(db.keys()))):
            no_of_atoms = len(db[str(i)]['force'])
            Energy, Force, Stress = self.calculate_properties(db[str(i)],       # Energy per atom
                                    self.force_coefficient, self.stress_coefficient)
            
            # Store energy into list
            true_energy = db[str(i)]['energy'] / no_of_atoms
            energy.append(true_energy)
            _energy.append(Energy.sum() / no_of_atoms)

            if self.force_coefficient:
                true_force = np.ravel(db[str(i)]['force'])
                Force = np.ravel(Force)
                for m in range(len(true_force)):
                    force.append(true_force[m])
                    _force.append(Force[m])

            if self.stress_coefficient and (db[str(i)]['group'] in self.stress_group):
                true_stress = np.array(db[str(i)]['stress'])#.flat[[0,3,5,3,1,4,5,4,2]]
                Stress = np.ravel(Stress)
                for m in range(len(true_stress)):
                    stress.append(true_stress[m])
                    _stress.append(Stress[m])

        energy, force, stress = np.asarray(energy), np.asarray(force), np.asarray(stress)
        _energy, _force, _stress = np.asarray(_energy), np.asarray(_force), np.asarray(_stress)

        # Dump the true and predicted values into text file.
        self.dump_evaluate(_energy, energy, filename=figname[:-4]+'Energy.txt')
        if self.force_coefficient:
            self.dump_evaluate(_force, force, filename=figname[:-4]+'Force.txt')
        if self.stress_coefficient:
            self.dump_evaluate(_stress, stress, filename=figname[:-4]+'Stress.txt')
        
        # Calculate the statistical metrics for energy.
        E_mae = self.mean_absolute_error(energy, _energy)
        E_mse = self.mean_squared_error(energy, _energy)
        E_r2 = self.r2_score(energy, _energy)
        print("The results for energy: ")
        print("    Energy R2     {:8.6f}".format(E_r2))
        print("    Energy MAE    {:8.6f}".format(E_mae))
        print("    Energy RMSE   {:8.6f}".format(E_mse))

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
            print("The results for force: ")
            print("    Force R2      {:8.6f}".format(F_r2))
            print("    Force MAE     {:8.6f}".format(F_mae))
            print("    Force RMSE    {:8.6f}".format(F_mse))

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

        if self.stress_coefficient:
            S_mae = self.mean_absolute_error(stress, _stress)
            S_mse = self.mean_squared_error(stress, _stress)
            S_r2 = self.r2_score(stress, _stress)
            print("The results for stress: ")
            print("    Stress R2      {:8.6f}".format(S_r2))
            print("    Stress MAE     {:8.6f}".format(S_mae))
            print("    Stress RMSE    {:8.6f}".format(S_mse))

            # Plotting the stress results.
            stress_str = 'Stress: r2({:.4f}), MAE({:.3f} GPa)'. \
                        format(S_r2, S_mae)
            plt.title(stress_str)
            plt.scatter(stress, _stress, s=5, label='Stress')
            plt.legend(loc=2)
            plt.xlabel('True (GPa)')
            plt.ylabel('Prediction (GPa)')
            plt.tight_layout()
            plt.savefig(self.path+'Stress_'+figname)
            plt.close()
            print("The stress figure is exported to: {:s}".format(self.path+'Stress_'+figname))
            print("\n")
        else:
            S_mae, S_mse, S_r2 = None, None, None

        return (E_mae, E_mse, E_r2, F_mae, F_mse, F_r2, S_mae, S_mse, S_r2)


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
            theta = np.linalg.lstsq(_X, _y, rcond=None)[0]

        return theta


    def save_checkpoint(self, des_info, filename=None):
        """ Save Polynomial Regression model to PyTorch. """
        _filename = self.path

        if filename:
            _filename += filename
        else:
            _filename += 'PolyReg-checkpoint.pth'

        checkpoint = {'elements': self.elements,
                      'algorithm': 'PR',
                      'force_coefficient': self.force_coefficient,
                      'path': self.path,
                      'quadratic': self.quadratic,
                      'coef_': self.coef_,
                      'des_info': des_info}

        save(checkpoint, _filename)
        if des_info['type'] in ['SNAP', 'snap', 'SO3', 'SOAP']:
            self.save_weights_to_txt(des_info)
        print("The Linear Regression Potential is exported to {:s}".format(_filename))
        print("\n")


    def save_weights_to_txt(self, des_info):
        """ Saving the model weights to txt file. """
        with open(self.path+"PR_weights.txt", "w") as f:
            f.write("# Polynomial Regression weights generated in PyXtal_FF \n")
            f.write("# total_species ncoefficient \n\n")
            f.write(f"{len(self.elements)} {self.d_max+1} \n")
            count = 0
            for element in self.elements:
                #if des_info['type'] in ['SNAP', 'snap']:
                #    f.write(f"{element} 0.5 {des_info['weights'][element]} \n")
                #else:
                #    f.write(f"{element} \n")
                for _ in range(self.d_max+1):
                    f.write(f"{self.coef_[count]} \n")
                    count += 1

        with open(self.path+"DescriptorParam.txt", "w") as f:
            f.write("# Descriptor parameters generated in PyXtal_FF \n\n")
            f.write("# Required \n")
            f.write(f"rcutfac {des_info['Rc']} \n")
            
            if des_info['type'] in ['SO3', 'SOAP']:
                f.write(f"nmax {des_info['parameters']['nmax']} \n")
                f.write(f"lmax {des_info['parameters']['lmax']} \n")
                f.write(f"alpha {des_info['parameters']['alpha']} \n\n")
            else:
                f.write(f"twojmax {des_info['parameters']['lmax']*2} \n\n")

            f.write("# Elements \n\n")
            f.write(f"nelems {len(self.elements)} \n")
            f.write("elems ")
            for element in self.elements:
                f.write(f"{element} ")
            f.write("\n")

            if des_info['type'] in ['snap', 'SNAP', 'SO3', 'SOAP']:
                f.write("radelems ")
                for element in self.elements:
                    f.write("0.5 ")
                f.write("\n")

                if des_info['type'] in ['snap', 'SNAP']:
                    f.write("welems ")
                    for element in self.elements:
                        f.write(f"{des_info['weights'][element]} ")
                    f.write("\n\n")
                else:
                    f.write("welems ")
                    ele = Element(self.elements)
                    atomic_numbers = ele.get_Z()
                    for num in atomic_numbers:
                        f.write(f"{num} ")
                    f.write("\n\n")

            if des_info['type'] in ['snap', 'SNAP']:
                f.write(f"rfac0 {des_info['parameters']['rfac']} \n")
                f.write(f"rmin0 0 ")
                f.write("\n")
                f.write("switchflag 1 \n")
                f.write("bzeroflag 0 \n")


    def load_checkpoint(self, filename=None):
        """ Load Polynomial Regression file from PyTorch. """
        checkpoint = load(filename)

        # Inconsistent algorithm.
        if checkpoint['algorithm'] != 'PR':
            msg = "The loaded algorithm is not Polynomial Regression."
            raise NotImplementedError(msg)
        
        # Check the consistency with the system of elements
        msg = f"The system, {self.elements}, are not consistent with "\
                    +"the loaded system, {checkpoint['elements']}."

        if len(self.elements) != len(checkpoint['elements']):
            raise ValueError(msg)
        
        for i in range(len(self.elements)):
            if self.elements[i] != checkpoint['elements'][i]:
                raise ValueError(msg)
        
        self.coef_ = checkpoint['coef_']
        self.quadratic = checkpoint['quadratic']

        return checkpoint['des_info']


    def calculate_properties(self, descriptor, bforce=True, bstress=False):
        """ A routine to compute energy, forces, and stress.
        
        Parameters:
        -----------
        descriptor: list
            list of x, dxdr, and rdxdr.
        benergy, bforce, bstress: bool
            If False, excluding the property from calculation.

        Returns:
        --------
        energy: float
            The predicted energy
        forces: 2D array [N_atom, 3] (if dxdr is provided)
            The predicted forces
        stress: 2D array [3, 3] (if rdxdr is provided)
            The predicted stress
        """
        no_of_atoms = len(descriptor['elements'])
        energies, force, stress = np.zeros([no_of_atoms]), np.zeros([no_of_atoms, 3]), np.zeros([6])
        
        X = self.parse_descriptors({'0': descriptor}, fc=bforce, sc=bstress, train=False)
        
        _y = np.dot(X, self.coef_) # Calculate properties

        energies = _y[:no_of_atoms]

        #energy = _y[0] / no_of_atoms # get energy/atom
        
        if bforce: # get force
            force += np.reshape(_y[no_of_atoms:no_of_atoms+(no_of_atoms*3)], (no_of_atoms, 3))

        if bstress: # get stress
            stress += _y[-6:]*eV2GPa # in GPa
        
        return energies, force, stress


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


    def dump_evaluate(self, predicted, true, filename):
        """ Dump the evaluate results to text files. """
        absolute_diff = np.abs(np.subtract(predicted, true))
        combine = np.vstack((predicted, true, absolute_diff)).T
        np.savetxt(self.path+filename, combine, header='Predicted True Diff', fmt='%.7e')


    def parse_descriptors(self, data, fc=True, sc=False, train=True):
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
            db = shelve.open(self.path+data)
            no_of_structures = self.no_of_structures
            no_of_atoms = self.no_of_atoms
            stress_components = self.stress_components
        else:
            db = data
            no_of_structures = 1 # 1 for train is false
            no_of_atoms = len(data['0']['elements']) if fc else 0
            stress_components = 6 if sc else 0
        
        # Determine the total number of descriptors based on SNAP or qSNAP.
        # Note: d_max != self.d_max
        if self.d_max is None: #enable calculator works
            self.d_max = len(db['0']['x'][0])

        if self.quadratic:
            d_max = (self.d_max**2+3*self.d_max)//2 # (d^2 + 3*d) / 2
        else:
            d_max = self.d_max

        columns = (1+d_max)*len(self.elements)
        rows = no_of_structures if train else no_of_atoms
        rows += no_of_atoms * 3 if fc else 0 # x, y, and z
        rows += stress_components if sc else 0 # xx, xy, xz, ..., zz
        
        X = np.zeros([rows, columns])

        # Fill in X.
        xcount = 0
        for i in range(no_of_structures):
            data = db[str(i)]
            if train:
                try:
                    _group = True if data['group'] in self.stress_group else False
                except:
                    _group = False
            else:
                _group = True
           
            if self.quadratic:
                _x = data['x'][:, :self.d_max]
                x = np.zeros((len(_x), d_max))
                x[:, :self.d_max] += data['x'][:, :self.d_max]
                if fc:
                    _dxdr = data['dxdr'][:, :self.d_max, :]
                    dxdr = np.zeros([_dxdr.shape[0], d_max, 3])
                    dxdr[:, :self.d_max, :] += _dxdr
                    seq = data['seq']
                
                #if train:
                if sc and _group: #(data['group'] in self.stress_group):
                    _rdxdr = data['rdxdr'][:, :self.d_max, :]
                    rdxdr = np.zeros([len(_x), d_max, 6])
                    rdxdr[:, :self.d_max, :] += _rdxdr
                
                # self-term for x, dxdr, and rdxdr
                x_square = 0.5 * _x ** 2
                if fc:
                    dxdr_square = np.zeros_like(_dxdr)
                    for i in range(len(_x)):
                        arr = np.where(seq[:, 0]==i)[0]
                        dxdr_square[arr] = np.einsum('jkl, k->jkl', _dxdr[arr], _x[i])
                    #dxdr_square = np.einsum('ijkl,ik->ijkl', _dxdr, _x)
                if sc and _group:
                    rdxdr_square = np.einsum('ijk,ij->ijk', _rdxdr, _x)
                
                dcount = self.d_max
                for d1 in range(self.d_max):
                    # Cross term for x, dxdr, and rdxdr
                    x_d1_d2 = np.einsum('i, ij->ij', _x[:, d1], _x[:, d1+1:])
                    if fc:
                        shp = _dxdr.shape
                        dxdr_d1_d2 = np.zeros([shp[0], shp[1]-1-d1, shp[2]])
                        for i in range(len(_x)):
                            arr = np.where(seq[:, 0]==i)[0]
                            dxdr_d1_d2[arr] = np.einsum('ij,k->ikj', _dxdr[arr][:, d1, :], _x[i, d1+1:]) + \
                                    (_dxdr[arr][:, d1+1:, :] * _x[i, d1])

                        #dxdr_d1_d2 = np.einsum('ijl,ik->ijkl', _dxdr[:, :, d1, :], _x[:, d1+1:]) + \
                                #             np.einsum('ijkl,i->ijkl', _dxdr[:, :, d1+1:, :], _x[:, d1])
                    if sc and _group:
                        rdxdr_d1_d2 = np.einsum('ik,ij->ijk', _rdxdr[:, d1, :], _x[:, d1+1:]) + \
                                      np.einsum('ijk,i->ijk', _rdxdr[:, d1+1:, :], _x[:, d1])
                    
                    # Append for x, dxdr, rdxdr
                    x[:, dcount] += x_square[:, d1]
                    if fc:
                        dxdr[:, dcount, :] += dxdr_square[:, d1, :]
                    if sc and _group:
                        rdxdr[:, dcount, :] += rdxdr_square[:, d1, :]

                    dcount += 1
                    
                    x[:, dcount:dcount+len(x_d1_d2[0])] += x_d1_d2
                    if fc:
                        dxdr[:, dcount:dcount+len(x_d1_d2[0]), :] += dxdr_d1_d2
                    if sc and _group:
                        rdxdr[:, dcount:dcount+len(x_d1_d2[0]), :] += rdxdr_d1_d2
                    dcount += len(x_d1_d2[0])
                
            else:
                x = data['x'][:, :d_max]
                if fc:
                    seq = data['seq']   
                    dxdr = data['dxdr'][:, :d_max, :]
                if sc and _group:
                    rdxdr = data['rdxdr'][:, :d_max, :]
            
            elements = data['elements']
            
            # Arranging x and dxdr for energy and forces.
            bias_weights = 1.0
            
            if train:
                sna = np.zeros([len(self.elements), 1+d_max])
                if fc:
                    snad = np.zeros([len(self.elements), len(x), 1+d_max, 3])
                if sc and _group:
                    snav = np.zeros([len(self.elements), 1+d_max, 6])
                
                _sna, _snad, _snav, _count = {}, {}, {}, {}
                for element in self.elements:
                    _sna[element] = None
                    _snad[element] = None
                    _snav[element] = None
                    _count[element] = 0
                
                # Loop over the number of atoms in a structure.
                for e, element in enumerate(elements):
                    if _sna[element] is None:
                        _sna[element] = 1 * x[e]
                        if fc:
                            shp = snad.shape
                            _snad[element] = np.zeros([shp[1], shp[2]-1, shp[3]])
                            arr = np.where(seq[:, 0]==e)[0]
                            _seq = seq[arr][:, 1]
                            _snad[element][_seq] = -1 * dxdr[arr]
                        if sc and _group:
                            _snav[element] = -1 * rdxdr[e]  # [d, 6]
                    else:
                        _sna[element] += x[e]
                        if fc:
                            arr = np.where(seq[:, 0]==e)[0]
                            _seq = seq[arr][:, 1]
                            _snad[element][_seq] -= dxdr[arr]
                        if sc and _group: 
                            _snav[element] -= rdxdr[e]
                    _count[element] += 1

                for e, element in enumerate(self.elements):
                    if _count[element] > 0:
                        #_sna[element] /= _count[element]
                        sna[e, :] += np.hstack(([bias_weights*_count[element]], _sna[element]))
                        if fc:
                            snad[e, :, 1:, :] += _snad[element]
                        if sc and _group:
                            snav[e, 1:, :] += _snav[element]
                        
                # X for energy
                X[xcount, :] += sna.ravel()
                xcount += 1

                # X for forces.
                if fc:
                    for j in range(snad.shape[1]):
                        for k in range(snad.shape[3]):
                            X[xcount, :] += snad[:, j, :, k].ravel()
                            xcount += 1
                
                # X for stress.
                if sc and _group: 
                    shp = snav.shape
                    X[xcount:xcount+6, :] = snav.reshape([shp[0]*shp[1], shp[2]]).T
                    xcount += 6

            else:
                if fc:
                    snad = np.zeros([len(self.elements), len(x), 1+d_max, 3])
                if sc and _group:
                    snav = np.zeros([len(self.elements), 1+d_max, 6])
                _snad, _snav, _count = {}, {}, {}
                for element in self.elements:
                    _snad[element] = None
                    _snav[element] = None
                    _count[element] = 0

                # Loop over the number of atoms in a structure.
                for e, element in enumerate(elements):
                    elem_cnt = self.elements.index(element)
                    X[xcount, elem_cnt*(1+d_max):(elem_cnt+1)*(1+d_max)] = np.hstack(([bias_weights], x[e]))
                    xcount += 1
                    if _snad[element] is None:
                        if fc:
                            shp = snad.shape
                            _snad[element] = np.zeros([shp[1], shp[2]-1, shp[3]])
                            arr = np.where(seq[:, 0]==e)[0]
                            _seq = seq[arr][:, 1]
                            _snad[element][_seq] = -1 * dxdr[arr]
                        if sc and _group:
                            _snav[element] = -1 * rdxdr[e]  # [d, 6]
                    else:
                        if fc:
                            arr = np.where(seq[:, 0]==e)[0]
                            _seq = seq[arr][:, 1]
                            _snad[element][_seq] -= dxdr[arr]
                        if sc and _group: 
                            _snav[element] -= rdxdr[e]
                    _count[element] += 1

                for e, element in enumerate(self.elements):
                    if _count[element] > 0:
                        if fc:
                            snad[e, :, 1:, :] += _snad[element]
                        if sc and _group:
                            snav[e, 1:, :] += _snav[element]

                # X for forces.
                if fc:
                    for j in range(snad.shape[1]):
                        for k in range(snad.shape[3]):
                            X[xcount, :] += snad[:, j, :, k].ravel()
                            xcount += 1
                
                # X for stress.
                if sc and _group: 
                    shp = snav.shape
                    X[xcount:xcount+6, :] = snav.reshape([shp[0]*shp[1], shp[2]]).T
                    xcount += 6
        
        if train:
            db.close()

        return X


    def parse_features(self, data):
        """ Parse features (energy, forces, and stress) into 1-D array.
        
        Returns
        -------
        y: 1-D array [n+m*3+n*3*3,]
            y contains the energy, forces, and stress of structures 
            in 1-D array. Force and stress may not be present.
            
            n = # of structures
            m = # of atoms in a unit cell

        w: 1-D array [n+m*3+n*3*3,]
            w contains the relative importance between energy, forces, 
            and stress.
        """
        db = shelve.open(self.path+data)
        self.no_of_atoms = 0
        self.stress_components = 0

        y = None # store the features (energy+forces+stress)
        w = None # weight of each sample
        
        for i in range(self.no_of_structures):
            data = db[str(i)]
            #energy = np.array([data['energy']/len(data['elements'])])
            energy = np.array([data['energy']])
            w_energy = np.array([1.])

            if self.force_coefficient:
                force = np.array(data['force']).ravel()
                w_force = np.array([self.force_coefficient]*len(force))

                if self.stress_coefficient and (data['group'] in self.stress_group):     # energy + forces + stress
                    stress = np.array(data['stress'])#.flat[[0,3,5,3,1,4,5,4,2]]
                    w_stress = np.array([self.stress_coefficient]*len(stress))
                    self.stress_components += 6
                    
                    if y is None:
                        y = np.concatenate((energy, force, stress))
                        w = np.concatenate((w_energy, w_force, w_stress))
                    else:
                        y = np.concatenate((y, energy, force, stress))
                        w = np.concatenate((w, w_energy, w_force, w_stress))
                else:                                                                           # energy + forces
                    if y is None:
                        y = np.concatenate((energy, force))
                        w = np.concatenate((w_energy, w_force))
                    else:
                        y = np.concatenate((y, energy, force))
                        w = np.concatenate((w, w_energy, w_force))
                
                # Count the number of atoms for the entire structures.
                self.no_of_atoms += len(data['force'])

            else:
                if self.stress_coefficient and (data['group'] in self.stress_group):    # energy + stress
                    stress = np.array(data['stress'])#.flat[[0,3,5,3,1,4,5,4,2]]
                    w_stress = np.array([self.stress_coefficient]*len(stress))
                    self.stress_components += 6
                    
                    if y is None:
                        y = np.concatenate((energy, stress))
                        w = np.concatenate((w_energy, w_stress))
                    else:
                        y = np.concatenate((y, energy, stress))
                        w = np.concatenate((w, w_energy, w_stress))

                else:                                                                           # energy only
                    if y is None:
                        y = energy
                        w = w_energy
                    else:
                        y = np.concatenate((y, energy))
                        w = np.concatenate((w, w_energy))
        
        db.close()
        gc.collect()

        return y, w
