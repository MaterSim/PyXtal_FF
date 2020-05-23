#!/usr/bin/env  python
# encoding: utf-8
import os
import gc
import time
import math
import torch
torch.set_default_tensor_type(torch.DoubleTensor)
import numpy as np

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mticker
plt.style.use("ggplot")

from scipy.linalg import cholesky, cho_solve

from pyxtal_ff.models.optimizers.regressor import Regressor

class GaussianProcess():
    """ Gaussian Process Regressor. """
    def __init__(self, elements, force_coefficient, path, epoch, noise, kernel):
        self.elements = sorted(elements)

        self.force_coefficient = force_coefficient
        self.path = path
        self.epoch = epoch
        self.noise = noise
        self.unit = 'eV'
        self.device = 'cpu'

        self.kernel = kernel


    def train(self, TrainDescriptors, TrainFeatures, optimizer=None, use_force=None):
        
        # Preprocess descriptors and features.
        self.preprocess(TrainDescriptors, TrainFeatures)
        gc.collect()

        # Define model
        if self.kernel == 'RBF':
            self.models = {'model': RBFKernel(d=self.no_of_descriptors,
                                              device=self.device)}

        # Set up optimizer to train the GPR
        self.regressor = Regressor(optimizer['method'], optimizer['parameters'])
        self.optimizer = self.regressor.regress(models=self.models)

        print("==================================== Training ====================================")
        print("\n")
        print(f"Optimizer         : {optimizer['method']}")
        print(f"Force_coefficient : {self.force_coefficient}\n")
        
        # Training GPR
        t0 = time.time()
        for epoch in range(self.epoch):
            print("Initial state : ")
            def closure():
                loss, E_mae, F_mae = self.log_marginal_likelihood()
                print("    Loss: {:10.6f}     Energy MAE: {:10.10f}     Force MAE: {:10.10f}".\
                        format(loss, E_mae, F_mae))
                self.optimizer.zero_grad()
                loss.backward()
                return loss
            self.optimizer.step(closure)
        t1 = time.time()
        print("\nThe training time: {:.2f} s".format(t1-t0))

        # Save_checkpoint! 
        #print("Here is the optimal parameters: ")
        #print(self.models['model'].parameters())

        print("\n============================== Training is Completed =============================\n")


    def predict(self,):
        pass
    
            
    def evaluate(self, TestDescriptors, TestFeatures, figname):
        if figname[:-4] == 'Train':
            print(f"============================= Evaluating {figname[:-4]}ing Set ============================\n")
        else:
            self.force_coefficient = 0.1
            print("============================= Evaluating Testing Set =============================\n")
        
        TestDescriptors = self.normalized(TestDescriptors,
                                          self.drange, self.unit)
        no_of_structures = TestDescriptors['no_of_structures']

        # Parse descriptors and features
        X = [{} for _ in range(len(TestDescriptors['x']))]
        if self.force_coefficient:
            DXDR = [{} for _ in range(len(TestDescriptors['dxdr']))]
        else:
            DXDR = [None]*len(X)
        
        energy, force = [], []
        for i in range(no_of_structures):
            energy.append(TestFeatures[i]['energy']/len(TestFeatures[i]['force']))
            force.append(np.ravel(TestFeatures[i]['force']))
            for element in self.elements:
                X[i][element] = torch.Tensor(TestDescriptors['x'][i][element])
                if self.force_coefficient:
                    DXDR[i][element] = torch.DoubleTensor(TestDescriptors['dxdr'][i][element])
        energy = np.array(energy)
        force = np.array([x for i in force for x in i])

        del(TestDescriptors)

        # Perform GPR prediction
        Ks = self.get_covariance_matrix(self.x, X, self.dxdr, DXDR)
        pred = Ks.T @ self.alpha

        _energy = np.asarray(pred[:no_of_structures, 0].detach().numpy())
        if self.force_coefficient:
            _force = np.asarray(pred[no_of_structures:, 0].detach().numpy())

        # Dump the true and predicted values into text file.
        self.dump_evaluate(_energy, energy, filename=figname[:-4]+'Energy.txt')
        if self.force_coefficient:
            self.dump_evaluate(_force, force, filename=figname[:-4]+'Force.txt')

        # Calculate the MAE and MSE values and r2
        E_mae = self.mean_absolute_error(energy, _energy)
        E_mse = self.mean_squared_error(energy, _energy)
        E_r2 = self.r2_score(energy, _energy)
        print("The results for energy: ")
        print("    Energy R2     {:8.6f}".format(E_r2))
        print("    Energy MAE    {:8.6f}".format(E_mae))
        print("    Energy RMSE   {:8.6f}".format(E_mse))
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
        

    def mean_absolute_error(self, true, predicted):
        """ Calculate mean absolute error of energy or force. """
        return sum(abs(true-predicted)/len(true))


    def mean_squared_error(self, true, predicted):
        """ Calculate mean square error of energy or force. """
        return np.sqrt(sum((true-predicted) ** 2 /len(true)))


    def r2_score(self, true, predicted):
        """ calculate the r square of energy or force. """
        t_bar = sum(true)/len(true)
        square_error = sum((true-predicted) ** 2)
        true_variance = sum((true-t_bar) ** 2)
        return 1 - square_error / true_variance


    def dump_evaluate(self, predicted, true, filename):
        """ Dump the evaluate results to text files. """
        absolute_diff = np.abs(np.subtract(predicted, true))
        combine = np.vstack((predicted, true, absolute_diff)).T
        np.savetxt(self.path+filename, combine, header='Predicted True Diff', fmt='%.7e')


    def log_marginal_likelihood(self):
        K = self.get_covariance_matrix(self.x, self.x, self.dxdr, self.dxdr)
        self.L = torch.cholesky(K + self.noise * torch.eye(len(K)), upper=False)
        self.alpha = torch.cholesky_solve(self.y, self.L)

        # log marginal likelihood
        MLL = -0.5 * torch.einsum("ik,ik", self.y, self.alpha)
        MLL -= torch.log(torch.diag(self.L)).sum()
        MLL -= self.L.shape[0] / 2 * math.log(2 * math.pi)
        
        mu = K.T @ self.alpha
        error = torch.abs(mu - self.y)
        E_mae = error[:self.no_of_structures].mean()
        F_mae = error[self.no_of_structures:].mean()

        return -MLL, E_mae, F_mae


    def get_covariance_matrix(self, X1, X2, dX1, dX2):
        m1, m2 = len(X1), len(X2)
        n1, n2 = 0, 0
        models = self.models['model']

        for ele in self.elements:
            if dX1[0]:
                n1 = sum(dx1[ele].shape[1]*dx1[ele].shape[3] for dx1 in dX1)
            if dX2[0]:
                n2 = sum(dx2[ele].shape[1]*dx2[ele].shape[3] for dx2 in dX2)
        out = torch.zeros((m1+n1, m2+n2))
        
        # This is not necessary true for multi=species needs to fix this in the near future.
        ki, ni = m1, 0
        for ele in self.elements:
            for i, (x1, dx1) in enumerate(zip(X1, dX1)):
                kj = m2
                if dx1:
                    ni = dx1[ele].shape[1]*dx1[ele].shape[3]
                
                for j, (x2, dx2) in enumerate(zip(X2, dX2)):
                    if dx2:
                        nj = dx2[ele].shape[1]*dx2[ele].shape[3]

                    # Covariance between E_i and E_j
                    out[i, j], K = self.get_kee(models, x1[ele], x2[ele])

                    if dx1 and dx2:
                        # Covariance betweeen F_i and F_j
                        out[ki:ki+ni, kj:kj+nj] = self.get_kff(models, x1[ele], x2[ele], dx1[ele], dx2[ele], K, self.force_coefficient, [ni, nj])
                        
                    if dx1:
                        # Covariance between F_i and E_j
                        out[ki:ki+ni, j] = self.get_kfe(models, x1[ele], x2[ele], dx1[ele], K, self.force_coefficient)
                        
                    if dx2:
                        # Covariance between E_i and F_j
                        out[i, kj:kj+nj] = self.get_kef(models, x1[ele], x2[ele], dx2[ele], K, self.force_coefficient)
                        kj += nj

                if dx1:   
                    ki += ni

        return out


    def get_kee(self, models, x1, x2):
        K = models.covariance(x1, x2)
        kee = K.mean()
        return kee, K
        

    def get_kef(self, models, x1, x2, dx2, K, fc):
        grad_x2 = models.dk_dx2(K, x1, x2)
        kef = torch.einsum("ijk, jlkm->lm", grad_x2, dx2) / len(x1)
        return fc * kef.view(-1)
        

    def get_kfe(self, models, x1, x2, dx1, K, fc):
        grad_x1 = models.dk_dx1(K, x1, x2)
        kfe = torch.einsum("ijk, ilkm->lm", grad_x1, dx1) / len(x2)
        return fc * kfe.view(-1)
 

    def get_kff(self, models, x1, x2, dx1, dx2, K, fc, shape):
        grad_x1x2 = models.d2k_dx1dx2(K, x1, x2)
        dx1_grad_x1x2 = torch.einsum("ijkl, ihkm->jlhm", dx1, grad_x1x2)
        kff = torch.einsum("jlhm, hnmp -> jlnp", dx1_grad_x1x2, dx2)
        return fc ** 2 * kff.reshape(shape[0], shape[1])


    def preprocess(self, descriptors, features):
        self.no_of_descriptors, self.no_of_structures = len(descriptors[0]['x'][0]), len(descriptors)

        # Generate and plot descriptor range.
        self.drange = self.get_descriptors_range(descriptors)
        #self.plot_hist(descriptors, figname=self.path+"histogram.png", figsize=(12,24))

        # Transform descriptors into normalized descriptors
        descriptors = self.normalized(descriptors, self.drange, self.unit)
        
        # Parsing descriptor
        self.x = [{} for _ in range(len(descriptors['x']))]
        if self.force_coefficient:
            self.dxdr = [{} for _ in range(len(descriptors['dxdr']))]
        else:
            self.dxdr = [None]*len(self.x)
        
        n_atoms = np.zeros(self.no_of_structures, dtype=int)
        for i in range(self.no_of_structures):
            for element in self.elements:
                self.x[i][element] = torch.Tensor(descriptors['x'][i][element], device=self.device)
                self.x[i][element].requires_grad = True
                n_atoms[i] += len(descriptors['x'][i][element])
                if self.force_coefficient:
                    self.dxdr[i][element] = torch.DoubleTensor(descriptors['dxdr'][i][element], device=self.device)
                    self.dxdr[i][element].requires_grad = True
        del(descriptors)

        # Parse Energy & Forces
        energy = [struct["energy"] for struct in features] / n_atoms
        force = [np.array(struct["force"]) for struct in features]
        if self.force_coefficient:
            y = np.expand_dims(np.hstack([energy[:], np.vstack(force[:]).ravel()]), axis=1)
        else:
            y = np.expand_dims(energy, axis=1)
        self.y = torch.DoubleTensor(y).to(self.device)

        # Perform SOFTMAX for structure importance here! For near future development.
        # Perform DataLoader from PyTorch if necessary here.
    

    def normalized(self, descriptors, drange, unit, norm=[0., 1.]):
        """ Normalizing the descriptors to the range of [0., 1.] based on the
        min and max value of the descriptors in the crystal structures. 
        
        Parameters
        ----------
        descriptors: dict
            The atom-centered descriptors.
        drange:
            The range of the descriptors for each element species.
        unit: str
            The unit of energy ('eV' or 'Ha').
            
        Returns
        -------
        dict
            The normalized descriptors.
        
        """
        d = {}
        d['no_of_structures'] = len(descriptors)
        d['no_of_descriptors'] = len(descriptors[0]['x'][0])
        
        d['x'] = {}
        if 'dxdr' in descriptors[0]:
            d['dxdr'] = {}
    
        # Normalize each structure.
        for i in range(d['no_of_structures']):
            d['x'][i] = {}
            if 'dxdr' in descriptors[0]:
                d['dxdr'][i] = {}
            
            no_of_center_atom = {}
            count = {}
            
            for element in self.elements:
                no_of_center_atom[element] = 0
                no_of_neighbors = 0
                count[element] = 0
                
            for e in descriptors[i]['elements']:
                no_of_center_atom[e] += 1
                no_of_neighbors += 1
                
            for element in self.elements:
                i_size = no_of_center_atom[element]
                j_size = no_of_neighbors
                d['x'][i][element] = np.zeros((i_size, d['no_of_descriptors']))
                if 'dxdr' in descriptors[0]:
                    d['dxdr'][i][element] = np.zeros((i_size, j_size, 
                                                    d['no_of_descriptors'], 3))
            
            for m in range(len(descriptors[i]['x'])):
                _des = descriptors[i]['x'][m]
                element = descriptors[i]['elements'][m]
                _drange = drange[element]
                scale = (norm[1] - norm[0]) / (_drange[:, 1] - _drange[:, 0])
                des = norm[0] + scale * (_des - _drange[:, 0])
                d['x'][i][element][count[element]] = des
               
                if 'dxdr' in descriptors[i].keys():
                    for n in range(len(descriptors[i]['dxdr'][m])):
                        for p in range(3):
                            index = count[element]
                            _desp = descriptors[i]['dxdr'][m, n, :, p]
                            if unit == 'eV':
                                desp = scale * _desp
                            elif unit == 'Ha':
                                desp = 0.529177 * scale * _desp # to 1/Bohr
                            d['dxdr'][i][element][index, n, :, p] = desp

                count[element] += 1


        return d


    def get_descriptors_range(self, descriptors):
        """ Calculate the range (min and max values) of the descriptors 
        corresponding to all of the crystal structures.
        
        Parameters
        ----------
        descriptors: dict of dicts
            Atom-centered descriptors.
            
        Returns
        -------
        dict
            The range of the descriptors for each element species.
        """
        _DRANGE = {}
        no_of_structures = len(descriptors)
        for i in range(no_of_structures):
            for j, descriptor in enumerate(descriptors[i]['x']):
                element = descriptors[i]['elements'][j]
                if element not in _DRANGE.keys():
                    _DRANGE[element] = np.asarray([np.asarray([__, __]) \
                                      for __ in descriptor])
                else:
                    assert len(_DRANGE[element]) == len(descriptor)
                    for j, des in enumerate(descriptor):
                        if des < _DRANGE[element][j][0]:
                            _DRANGE[element][j][0] = des
                        elif des > _DRANGE[element][j][1]:
                            _DRANGE[element][j][1] = des
        
        return _DRANGE


    
class RBFKernel():
    def __init__(self, d, device='cpu'):
        super().__init__()
        self.device = device
        self.sigmaF = torch.DoubleTensor([1.])
        self.sigmaF.requires_grad_()

        # Init the shapes of the RBF kernel as normal distribution
        self.sigmaL = torch.empty([1, d], device=self.device).uniform_() * 10
        self.sigmaL.requires_grad_()


    def covariance(self, x1, x2):
        # x1: m x d, x2: n x d, K: m x n
        _x1 = x1.clone()
        _x2 = x2.clone()
        _x1 /= self.sigmaL ** 2
        _x2 /= self.sigmaL ** 2
        D = torch.sum(_x1*_x1, axis=1, keepdims=True) + torch.sum(_x2*_x2, axis=1) - 2*_x1@_x2.T
        return self.sigmaF ** 2 * torch.exp(-0.5 * D)


    def dk_dx1(self, K, x1, x2):
        # grad_x1: m x n x d
        return K[:, :, None] * (-(x1[:, None, :] - x2) / self.sigmaL ** 2)


    def dk_dx2(self, K, x1, x2):
        # grad_x2: m x n x d
        return K[:, :, None] * ((x1[:, None, :] - x2) / self.sigmaL ** 2)


    def d2k_dx1dx2(self, K, x1, x2):
        # grad_x1x2: m x n x d x d
        d = x2.shape[-1]
        M = (x1[:, None, :] - x2) / self.sigmaL ** 2
        Q = torch.eye(d, device=self.device) / self.sigmaL ** 2 - M[:, :, :, None] * M[:, :, None, :]
        return K[:, :, None, None] * Q


    def parameters(self):
        return [self.sigmaL, self.sigmaF]
