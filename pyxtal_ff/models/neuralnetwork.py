#!/usr/bin/env  python
# encoding: utf-8
import os
import gc
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
import torch.nn.functional as F
torch.set_default_tensor_type(torch.DoubleTensor)

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mticker
plt.style.use("ggplot")

from pyxtal_ff.models.optimizers.regressor import Regressor


class NeuralNetwork():
    """ Atom-centered Neural Network model. The inputs are atom-centered 
    descriptors: BehlerParrinello or Bispectrum. The forward propagation of 
    the Neural Network predicts energy per atom, and the derivative of the 
    forward propagation predicts force.

    A machine learning interatomic potential can developed by optimizing the 
    weights of the Neural Network for a given system.
    
    Parameters
    ----------
    elements: list
         A list of atomic species in the crystal system.
    hiddenlayers: list or dict
        [3, 3] contains 2 layers with 3 nodes each. Each atomic species in the 
        crystal system is assigned with its own neural network architecture.
    activation: str
        The activation function for the neural network model.
        Options: tanh, sigmoid, and linear.
    random_seed: int
        Random seed for generating random initial random weights.
    batch_size: int
        Determine the number of structures in a batch per optimization step.
    epoch: int
        A measure of the number of times all of the training vectors 
        are used once to update the weights.
    device: str
        The device used to train: 'cpu' or 'cuda'.
    force_coefficient: float
        This parameter is used as the penalty parameter to scale 
        the force contribution relative to the energy.
    stress_coefficient: float 
        This parameter is used as the balance parameter scaling
        the stress contribution relative to the energy.
    stress_group: list of strings
        Only the intended group will be considered in stress training,
        i.e. ['Elastic'].
    alpha: float
        L2 penalty (regularization) parameter.
    softmax_beta: float
        The parameters used for Softmax Energy Penalty function.
    unit: str
        The unit of energy ('eV' or 'Ha').
    logging: ?
        ???
    restart: str
        Continuing Neural Network training from where it was left off.
    path: str
        A path to the directory where everything is saved.
    """
    def __init__(self, elements, hiddenlayers, activation, random_seed, 
                 batch_size, epoch, device, alpha, softmax_beta, unit, 
                 force_coefficient, stress_coefficient, stress_group,
                 logging, restart, path):
        
        self.elements = sorted(elements)
        
        # Adding the output layer to the hiddenlayers
        self._hiddenlayers = hiddenlayers
        if isinstance(hiddenlayers, list):
            hl = {}
            for element in self.elements:
                hl[element] = hiddenlayers + [1]
            self.hiddenlayers = hl
        elif isinstance(hiddenlayers, dict):
            for key, value in hiddenlayers.items():
                hiddenlayers[key] = value + [1]
            self.hiddenlayers = hl
        else:
            msg = f"Don't recognize {type(hiddenlayers)}. " +\
                  f"Please refer to documentations!"
            raise TypeError(msg)

        # Set-up activation
        self.activation = {}
        activation_modes = ['Tanh', 'Sigmoid', 'Linear', 'ReLU',
                            'PReLU', 'ReLU6', 'RReLU', 'SELU', 'CELU',
                            'Softplus', 'Softshrink', 'Softsign', 'Tanhshrink',
                            'Softmin', 'Softmax', 'LogSoftmax', 'LogSigmoid',
                            'LeakyReLU', 'Hardtanh', 'Hardshrink', 'ELU',]

        if isinstance(activation, str):
            for e in self.elements:
                self.activation[e] = [activation] * \
                                     len(self.hiddenlayers[e])
        elif isinstance(activation, list):
            for element in self.elements:
                self.activation[element] = activation
        else:
            # Users construct their own activations.
            self.activation = activation
        
        # Check if each of the activation functions is implemented.
        for e in self.elements:
            for act in self.activation[e]:
                if act not in activation_modes:
                    msg = f"{act} is not implemented. " +\
                          f"Please choose from {activation_modes}."
                    raise NotImplementedError(msg)
            assert len(self.activation[e]) == len(self.hiddenlayers[e]),\
            "The length of the activation function is inconsistent "+\
            "with the length of the hidden layers."
        
        if random_seed:
            torch.manual_seed(random_seed)
        
        if batch_size == None:
            self.shuffle = False
        else:
            self.shuffle = True
        self.batch_size = batch_size

        self.epoch = epoch
        self.device = device
        self.alpha = alpha  # l2 regulization
        self.softmax_beta = softmax_beta
        self.force_coefficient = force_coefficient
        self.stress_coefficient = stress_coefficient
        self.stress_group = stress_group

        # Set-up unit
        unit_options = ['eV', 'Ha']
        if unit not in unit_options:
            msg = f"{unit} is not implemented. " +\
                  f"Please choose from {unit_options}."
            raise NotImplementedError(msg)
        self.unit = unit
        
        self.logger = logging
        self.restart = restart
        self.path = path

        self.drange = None


    def train(self, TrainDescriptors, TrainFeatures, optimizer):
        """ Training of Neural Network Potential. """
        
        # If batch_size is None and optimizer is Adam or SGD, 
        # then batch_size equals total structures.
        if optimizer['method'] in ['sgd', 'SGD', 'Adam', 'adam', 'ADAM']:
            if self.batch_size == None:
                self.batch_size = len(TrainDescriptors)

        # Preprocess descriptors and features.
        self.preprocess(TrainDescriptors, TrainFeatures)

        # Calculate total number of parameters.
        self.total_parameters = 0
        for element in self.elements:
            for i, hl in enumerate(self.hiddenlayers[element]):
                if i == 0:
                    self.total_parameters += (self.no_of_descriptors+1)*hl
                else:
                    self.total_parameters += (self.hiddenlayers[element][i-1]+1)*hl
        
        if self.restart is None:
            # Creating Neural Network architectures.
            self.models = {}
            for element in self.elements: # Number of models depend on species
                m = 'nn.Sequential('
                for i, act in enumerate(self.activation[element]):
                    if i == 0:
                        m += f'nn.Linear({self.no_of_descriptors}, \
                               {self.hiddenlayers[element][i]}), '
                    else:
                        m += f'nn.Linear({self.hiddenlayers[element][i-1]}, \
                               {self.hiddenlayers[element][i]}), '
                                   
                    if act == 'Linear':
                        continue
                    else:
                        m += f'nn.{act}(), '
                m += f')'

                self.models[element] = eval(m).double().to(self.device)

            self.regressor = Regressor(optimizer['method'], optimizer['parameters'])
            self.optimizer = self.regressor.regress(models=self.models)

        else:
            # Look for previously saved models and continue optimizing from the last checkpoint.
            self.load_checkpoint(filename=self.restart, 
                                 method=optimizer['method'], args=optimizer['parameters'])
                
        print(f"No of structures  : {self.no_of_structures}")
        print(f"No of descriptors : {self.no_of_descriptors}")
        print(f"No of parameters  : {self.total_parameters}")
        print(f"Optimizer         : {optimizer['method']}")
        print(f"Force_coefficient : {self.force_coefficient}")
        if self.stress_coefficient:
            print(f"Stress_coefficient : {self.stress_coefficient}\n")

        # Run Neural Network Potential Training
        t0 = time.time()
        for epoch in range(self.epoch):
            if optimizer['method'] in ['lbfgs', 'LBFGS', 'lbfgsb']:
                print("Initial state : ")
                def closure(): # LBFGS gets loss and its gradient here.
                    train_loss, E_mae, F_mae, S_mae = self.calculate_loss(self.models, self.data)
                    print("    Loss: {:10.6f}     Energy MAE: {:10.4f}     Force MAE: {:10.4f}     Stress MAE: {:10.4f}".\
                            format(train_loss, E_mae, F_mae, S_mae))
                    self.optimizer.zero_grad()
                    train_loss.backward()
                    return train_loss
                self.optimizer.step(closure)

            elif optimizer['method'] in ['sgd', 'SGD', 'Adam', 'adam', 'ADAM']:
                if epoch == 0:
                    print("Initial state : ")
                    train_loss, E_mae, F_mae, S_mae = 0., 0., 0., 0
                    total = 0
                    for batch in self.data:
                        total += len(batch)
                        tl, Emae, Fmae, Smae = self.calculate_loss(self.models, batch)
                        train_loss += tl * len(batch)
                        E_mae += Emae * len(batch)
                        F_mae += Fmae * len(batch)
                        S_mae += Smae * len(batch)
                    train_loss /= total
                    E_mae /= total
                    F_mae /= total
                    S_mae /= total
                    print("    Loss: {:10.6f}     Energy MAE: {:10.4f}     Force MAE: {:10.4f}     Stress MAE: {:10.4f}".\
                            format(train_loss, E_mae, F_mae, S_mae))

                print("\nIteration {:4d}: ".format(epoch+1))
                for batch in self.data:
                    train_loss, E_mae, F_mae, S_mae = self.calculate_loss(self.models, batch)
                    self.optimizer.zero_grad()
                    train_loss.backward()
                    self.optimizer.step()
                    print("    Loss: {:10.6f}     Energy MAE: {:10.4f}     Force MAE: {:10.4f}     Stress MAE: {:10.4f}".\
                            format(train_loss, E_mae, F_mae, S_mae))
                                        
        t1 = time.time()
        print("\nThe training time: {:.2f} s".format(t1-t0))
        

    def evaluate(self, TestDescriptors, TestFeatures, figname):
        """ Evaluating the train or test data set based on trained Neural Network model. 
        Evaluate will only be performed in cpu mode. """
        
        # Normalized data set based on the training data set (drange).
        TestDescriptors = self.normalize(TestDescriptors, self.drange, self.unit)
        no_of_structures = TestDescriptors['no_of_structures']

        # Parse descriptors and Features
        energy, force, stress = [], [], []
        X = [{} for _ in range(len(TestDescriptors['x']))]
        if self.force_coefficient:
            DXDR = [{} for _ in range(len(TestDescriptors['dxdr']))]
        else:
            DXDR = [None]*len(X)
        if self.stress_coefficient:
            RDXDR = [{} for _ in range(len(TestDescriptors['rdxdr']))]
        else:
            RDXDR = [None]*len(X)

        for i in range(no_of_structures):
            for element in self.elements:
                X[i][element] = torch.DoubleTensor(TestDescriptors['x'][i][element])
                if self.force_coefficient:
                    DXDR[i][element] = torch.DoubleTensor(TestDescriptors['dxdr'][i][element])
                if self.stress_coefficient and (TestFeatures[i]['group'] in self.stress_group):
                    RDXDR[i][element] = torch.DoubleTensor(TestDescriptors['rdxdr'][i][element])
            
            energy.append(TestFeatures[i]['energy']/len(TestFeatures[i]['force']))
            force.append(np.ravel(TestFeatures[i]['force']))
            if self.stress_coefficient and (TestFeatures[i]['group'] in self.stress_group):
                stress.append(np.array(TestFeatures[i]['stress']))
                    
        # Switch models device to cpu if training is done in cuda.
        models = {}
        for element in self.elements:
            if next(self.models[element].parameters()).is_cuda:
                models[element] = self.models[element].cpu()
            else:
                models[element] = self.models[element]

        # Predicting the data set.
        _energy, _force, _stress = [], [], [] # Predicted energy and forces
        test_data = zip(X, DXDR, RDXDR)

        for i, (x, dxdr, rdxdr) in enumerate(test_data):
            dedx = None
            n_atoms = sum(len(value) for value in x.values())
            _Energy = 0.
            _Force = torch.zeros([n_atoms, 3], dtype=torch.float64)
            if self.stress_coefficient and (TestFeatures[i]['group'] in self.stress_group):
                _Stress = torch.zeros([6], dtype=torch.float64)

            for element, model in models.items():
                if x[element].nelement() > 0:
                    _x = x[element].requires_grad_()
                    _e = model(_x).sum()
                    _Energy += _e
                    if self.force_coefficient:
                        dedx = torch.autograd.grad(_e, _x)[0]
                        _Force += -1 * torch.einsum("ik, ijkl->jl", dedx, dxdr[element])
                    if self.stress_coefficient and (TestFeatures[i]['group'] in self.stress_group):
                        if self.force_coefficient is None:
                            dedx = torch.autograd.grad(_e, _x)[0]
                        _Stress += -1 * torch.einsum("ik, ikl->l", dedx, rdxdr[element])
            
            if self.stress_coefficient and (TestFeatures[i]['group'] in self.stress_group):
                _stress.append(np.ravel(_Stress))
            _force.append(np.ravel(_Force.numpy()))
            _energy.append(_Energy.item()/n_atoms)
        
        energy = np.array(energy)
        _energy = np.array(_energy)
        force = np.array([x for i in force for x in i])
        _force = np.array([x for i in _force for x in i])
        if self.stress_coefficient:
            stress = np.ravel(stress)
            _stress = np.ravel(_stress)
        
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
            # Calculate the statistical metrics for forces.
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
            length = 'A'
            if self.unit == 'Ha':
                length == 'Bohr'
            stress_str = 'Stress: r2({:.4f}), MAE({:.3f} {}/{})'. \
                        format(S_r2, S_mae, self.unit, length)
            plt.title(stress_str)
            plt.scatter(stress, _stress, s=5, label='Stress')
            plt.legend(loc=2)
            plt.xlabel('True ({}/{}^3)'.format(self.unit, length))
            plt.ylabel('Prediction ({}/{}^3)'.format(self.unit, length))
            plt.tight_layout()
            plt.savefig(self.path+'Stress_'+figname)
            plt.close()
            print("The stress figure is exported to: {:s}".format(self.path+'Stress_'+figname))
            print("\n")
        else:
            S_mae, S_mse, S_r2 = None, None, None

        
        return (E_mae, E_mse, E_r2, F_mae, F_mse, F_r2, S_mae, S_mse, S_r2)


    def preprocess(self, descriptors, features):
        """ Preprocess the descriptors and features to a convenient format
        for training Neural Network model. """
        self.no_of_descriptors = len(descriptors[0]['x'][0]) 
        self.no_of_structures = len(descriptors)

        if self.stress_coefficient and (self.stress_group is None):
            sg = []
            for i in range(len(features)):
                if features[i]['group'] not in sg:
                    sg.append(features[i]['group'])
            self.stress_group = sg

        # Generate and plot descriptor range.
        if self.drange is None:
            self.drange = self.get_descriptors_range(descriptors)
            #self.plot_hist(descriptors, figname=self.path+"histogram.png", figsize=(12, 24))

        # Transform descriptors into normalized descriptors
        descriptors = self.normalize(descriptors, self.drange, self.unit)
        
        # Convert to Torch
        x = [{} for _ in range(len(descriptors['x']))]
        if self.force_coefficient:
            dxdr = [{} for _ in range(len(descriptors['dxdr']))]
        else:
            dxdr = [None]*len(x)
        if self.stress_coefficient:
            rdxdr = [{} for _ in range(len(descriptors['rdxdr']))]
        else:
            rdxdr = [None]*len(x)
        
        energy, force, stress = [], [], []
        for i in range(self.no_of_structures):
            for element in self.elements:
                x[i][element] = torch.DoubleTensor(descriptors['x'][i][element]).to(self.device)
                if self.force_coefficient:
                    dxdr[i][element] = torch.DoubleTensor(descriptors['dxdr'][i][element]).to(self.device)
                if self.stress_coefficient and features[i]['group'] in self.stress_group:
                    rdxdr[i][element] = torch.DoubleTensor(descriptors['rdxdr'][i][element]).to(self.device)
            
            # Parse energy, forces, and stress
            energy.append(features[i]['energy'])
            force.append(torch.DoubleTensor(features[i]['force']).to(self.device))
            if self.stress_coefficient and features[i]['group'] in self.stress_group:
                s = features[i]['stress']
                stress.append((torch.DoubleTensor(s)).to(self.device))
            else:
                stress.append(None)
        energy = torch.DoubleTensor(energy).to(self.device)
        
        del(descriptors)
        gc.collect() # Flush memory

        # Emphazising the importance of lower Energy mode.
        softmax = self._SOFTMAX(energy, x, beta=self.softmax_beta)
        
        # Compile x, dxdr, energy, force, and softmax into PyTorch DataLoader.
        self.data = data.DataLoader(Dataset(x, dxdr, rdxdr, energy, force, stress, softmax),
                                    batch_size=self.batch_size,
                                    shuffle=self.shuffle,
                                    collate_fn=self.collate_fn)

        del(x)
        del(dxdr)
        del(energy)
        del(force)
        del(softmax)
        gc.collect() # Flush memory


    def calculate_loss(self, models, batch):
        """ Calculate the total loss and MAE for energy and forces
        for a batch of structures per one optimization step. """ 

        energy_loss, force_loss, stress_loss = 0., 0., 0.
        energy_mae, force_mae, stress_mae = 0., 0., 0.
        all_atoms = 0
        s_count = 0
        
        for x, dxdr, rdxdr, energy, force, stress, sf in batch:
            n_atoms = sum(len(value) for value in x.values())
            all_atoms += n_atoms
            _Energy = 0  # Predicted total energy for a structure
            _force = torch.zeros([n_atoms, 3], dtype=torch.float64, device=self.device)
            if stress is not None:
                _stress = torch.zeros([6], dtype=torch.float64, device=self.device)
            
            dedx, sdedx = {}, {}
            for element, model in models.items():
                if x[element].nelement() > 0:
                    _x = x[element].requires_grad_()
                    _energy = model(_x).sum() # total energy for each specie
                    _Energy += _energy

                    if self.force_coefficient:
                        dedx[element] = torch.autograd.grad(_energy, _x, create_graph=True)[0]
                        _force += -1 * torch.einsum("ik, ijkl -> jl", 
                                                dedx[element], dxdr[element]) # [natoms, 3]

                    if stress is not None:
                        if self.force_coefficient is None:
                            dedx[element] = torch.autograd.grad(_energy, _x, create_graph=True)[0]
                        _stress += -1 * torch.einsum("ik, ikl->l", dedx[element], rdxdr[element]) # [6]
            
            energy_loss += sf.item()*((_Energy - energy) / n_atoms) ** 2
            energy_mae  += sf.item()*F.l1_loss(_Energy / n_atoms, energy / n_atoms)

            if self.force_coefficient:
                force_loss += sf.item()*self.force_coefficient * ((_force - force) ** 2).sum()
                force_mae  += sf.item()*F.l1_loss(_force, force) * n_atoms

            if stress is not None:
                stress_loss += sf.item()*self.stress_coefficient * ((_stress - stress) ** 2).sum()
                stress_mae += sf.item()*F.l1_loss(_stress, stress) * 6
                s_count += 6

        energy_loss = energy_loss / (2. * len(batch))
        energy_mae /= len(batch)

        if self.force_coefficient:
            force_loss = force_loss / (2. * all_atoms)
            force_mae /= all_atoms
            if self.stress_coefficient:
                stress_loss = stress_loss / (2. * s_count)
                stress_mae /= s_count
                loss = energy_loss + force_loss + stress_loss
            else:
                loss = energy_loss + force_loss

        else:
            if self.stress_coefficient:
                loss = energy_loss + stress_loss
            else:
                loss = energy_loss

        # Add regularization to the total loss.
        if self.alpha: 
            reg = 0.
            for element, model in models.items():
                for name, params in model.named_parameters():
                    if 'weight' in name:
                        reg += self.alpha * params.pow(2).sum()
            loss += reg
        
        return loss, energy_mae, force_mae, stress_mae


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


    def calculate_properties(self, descriptor, bforce=True, bstress=False):
        """ A routine to compute energy, forces, and stress.
        
        Parameters:
        -----------
        descriptor: list
            list of x, dxdr, and rdxdr.
        energy, force, stress: bool
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
        energy, force, stress = 0., np.zeros([no_of_atoms, 3]), np.zeros([6])
        
        d = self.normalize([descriptor], self.drange, unit=self.unit)
        x = d['x'][0]
        if bforce:
            dxdr = d['dxdr'][0]
        if bstress:
            rdxdr = d['rdxdr'][0]
        
        for element, model in self.models.items():
            if len(x[element]) > 0:
                _x = torch.DoubleTensor(x[element]).requires_grad_()
                if bforce:
                    _dxdr = torch.DoubleTensor(dxdr[element])
                if bstress:
                    _rdxdr = torch.DoubleTensor(rdxdr[element])
                _e = model(_x).sum()
                energy += _e
                
                if bforce:
                    dedx = torch.autograd.grad(_e, _x)[0]
                    force += -torch.einsum("ik, ijkl->jl", dedx, _dxdr).numpy()

                if bstress:
                    if bforce == False:
                        dedx = torch.autograd.grad(_e, _x)[0]
                    stress += -torch.einsum("ik, ikl->l", dedx, _rdxdr).numpy()

        return energy/no_of_atoms, force, stress


    def save_checkpoint(self, des_info, filename=None):
        """ Save PyTorch Neural Network models at a checkpoint. """
        _filename = self.path

        if filename:
            _filename += filename
        else:
            if isinstance(self._hiddenlayers, list):
                _hl = "-".join(str(x) for x in self._hiddenlayers)
                _filename += _hl + '-checkpoint.pth'
            else:
                count = 0
                for i in range(len(self.elements)):
                    _filename += "-".join(str(x) \
                        for x in self._hiddenlayers[self.elements[i]])
                    if count < len(self.elements)-1:
                        _filename += "_"
                _filename += '-checkpoint.pth'

        checkpoint = {'models': self.models,
                      'algorithm': 'NN',
                      'elements': self.elements,
                      'optimizer': self.optimizer.state_dict(),
                      'drange': self.drange,
                      'unit': self.unit,
                      'force_coefficient': self.force_coefficient,
                      'alpha': self.alpha,
                      'softmax_beta': self.softmax_beta,
                      'batch_size': self.batch_size,
                      'des_info': des_info}
                      
        torch.save(checkpoint, _filename)
        print("The Neural Network Potential is exported to {:s}".format(_filename))
        print("\n")


    def load_checkpoint(self, filename=None, method=None, args=None):
        """ Load PyTorch Neural Network models at previously saved checkpoint. """
        checkpoint = torch.load(filename)

        # Inconsistent algorithm.
        if checkpoint['algorithm'] != 'NN':
            msg = "The loaded algorithm is not Neural Network."
            raise NotImplementedError(msg)

        # Check the consistency with the system of elements
        msg = f"The system, {self.elements}, are not consistent with "\
                    +"the loaded system, {checkpoint['elements']}."

        if len(self.elements) != len(checkpoint['elements']):
            raise ValueError(msg)
        
        for i in range(len(self.elements)):
            if self.elements[i] != checkpoint['elements'][i]:
                raise ValueError(msg)

        self.models = checkpoint['models']

        if method:
            # Set-up optimizer for optimizing NN weights.
            self.regressor = Regressor(method, args)
            self.optimizer = self.regressor.regress(models=self.models)

            # If different optimizer is used in loading, start the opt. from scratch.
            # Else load the optimizer state.
            pg = checkpoint['optimizer']['param_groups'][0].keys()
            if method in ['lbfgs', 'LBFGS', 'lbfgsb'] and 'line_search_fn' in pg:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                for key, value in args.items():
                    if key == 'max_eval':
                        if args[key] == None:
                            self.optimizer.param_groups[0][key] = 15000
                        else:
                            self.optimizer.param_groups[0][key] = args[key]
                    else:
                        self.optimizer.param_groups[0][key] = args[key]

            elif method in ['sgd', 'SGD'] and 'nesterov' in pg:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                for key, value in args.items():
                    self.optimizer.param_groups[0][key] = args[key]

            elif method in ['adam', 'ADAM', 'Adam'] and 'amsgrad' in pg:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                for key, value in args.items():
                    self.optimizer.param_groups[0][key] = args[key]

        else: # For predict()
            self.drange = checkpoint['drange']
            self.unit = checkpoint['unit']

        return checkpoint['des_info']
    
    
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
            The ranges of the descriptors for each chemical specie.
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
    
    
    def normalize(self, descriptors, drange, unit, norm=[0., 1.]):
        """ Normalizing the descriptors to the range of [0., 1.] based on the
        min and max value of the entire descriptors.

        Example:
        X.shape == [60, 10]; len(self.elements) == 2
        X_norm -> {'element1': [40, 10], 'element2': [20, 10]}
        
        Parameters
        ----------
        descriptors: dict
            The atom-centered descriptors.
        drange:
            The range of the descriptors for each element species.
        unit: str
            The unit of energy ('eV' or 'Ha').
        norm: tuple of floats.
            The lower and upper bounds of the normalization.
            
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
        if 'rdxdr' in descriptors[0]:
            d['rdxdr'] = {}
        
        # Normalize each structure.
        for i in range(d['no_of_structures']):
            d['x'][i] = {}
            if 'dxdr' in descriptors[0]:
                d['dxdr'][i] = {}
            if 'rdxdr' in descriptors[0]:
                d['rdxdr'][i] = {}
            
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
                d['x'][i][element] = np.zeros([i_size, d['no_of_descriptors']])
                #if 'dxdr' in descriptors[0]:
                if descriptors[i]['dxdr'] is not None:
                    d['dxdr'][i][element] = np.zeros([i_size, j_size, d['no_of_descriptors'], 3])
                #else:
                #    d['dxdr'][i][element] = None
                if 'rdxdr' in descriptors[i] and descriptors[i]['rdxdr'] is not None:
                    d['rdxdr'][i][element] = np.zeros([i_size, d['no_of_descriptors'], 6])
                #else:
                #    d['rdxdr'][i][element] = None
            
            for m in range(len(descriptors[i]['x'])):
                _des = descriptors[i]['x'][m]
                element = descriptors[i]['elements'][m]
                _drange = drange[element]
                scale = (norm[1] - norm[0]) / (_drange[:, 1] - _drange[:, 0])
                des = norm[0] + scale * (_des - _drange[:, 0])
                d['x'][i][element][count[element]] = des
                
                #if 'dxdr' in descriptors[i].keys():
                if descriptors[i]['dxdr'] is not None:
                    if unit == 'eV':
                        desp = np.einsum('j, ijk->ijk', scale, descriptors[i]['dxdr'][m])
                    elif unit == 'Ha':
                        desp = 0.529177 * np.einsum('j, ijk->ijk', scale, descriptors[i]['dxdr'][m])
                    d['dxdr'][i][element][count[element]] = desp
                
                #if descriptors[i]['rdxdr'] is not None:
                if 'rdxdr' in descriptors[i] and descriptors[i]['rdxdr'] is not None:

                    dess = np.einsum('k, kl->kl', scale, descriptors[i]['rdxdr'][m])
                    if unit == 'eV':
                        pass
                    elif unit == 'Ha':
                        dess *= 0.529177
                    d['rdxdr'][i][element][count[element]] = dess
                
                count[element] += 1

        return d


    def _SOFTMAX(self, energy, x, beta=-1):
        """ Assign the weight to each sample based on the softmax function. """

        # Length of smax is equal to the number of samples.
        smax = np.ones(len(energy))

        if beta is not None:
            smax = np.zeros(len(energy))
            no_of_atoms = []
            for i in range(len(x)):
                natoms = 0
                for key in x[i].keys():
                    natoms += len(x[i][key])
                no_of_atoms.append(natoms)

            epa = np.asarray(energy) / np.asarray(no_of_atoms)

            smax += np.exp(beta*epa) / sum(np.exp(beta*epa))
            smax *= len(energy)

        return smax


    def plot_hist(self, descriptors, figname=None, figsize=(12, 16)):
        """ Plot the histogram of descriptors. """
        flatten_array = {}
        for e in self.elements: 
            flatten_array[e] = []
            
        no_of_descriptors = descriptors[0]['x'].shape[1]
        for i in range(len(descriptors)):
            x = descriptors[i]['x']
            symbols = descriptors[i]['elements']
            for e in self.elements:
                ids = []
                for id in range(len(symbols)):
                    if e == symbols[id]:
                        ids.append(id)
                if flatten_array[e] == []:
                    flatten_array[e] = x[ids, :]
                else:
                    flatten_array[e] = np.vstack( (flatten_array[e], x[ids, :]) )

        # Plotting
        fig = plt.figure(figsize=figsize)
        fig.suptitle('The distribution of descriptors after normalization', 
                     fontsize=22)
        gs = GridSpec(no_of_descriptors, len(self.elements))
        for ie, e in enumerate(self.elements):
            if self.drange is not None:
                print('\nDescriptors range for {:s} from the training set {:d}'. format(e, len(self.drange[e])))
                max_x = self.drange[e][:,1]
                min_x = self.drange[e][:,0]
            else:
                print('\nDescriptors range for {:s} from the provided data {:d}'. format(e, len(flatten_array[e])))
                max_x = np.max(flatten_array[e], axis=0)
                min_x = np.min(flatten_array[e], axis=0)

            flatten_array[e] -= min_x
            flatten_array[e] /= (max_x - min_x)

            for ix in range(len(max_x)):
               print('{:12.6f} {:12.6f}'.format(min_x[ix], max_x[ix]))
               tmp = flatten_array[e][:,ix]
               ids = np.where((tmp<-1e-2) | (tmp>1))[0]
               if len(ids) > 0:
                   print('Warning: {:d} numbers are outside the range after normalization'.format(len(ids)))
                   print('-------', ids, tmp[ids], '---------')

            for ix in range(no_of_descriptors-1,-1,-1):
                label = "{:s}{:d}: {:8.4f} {:8.4f}".format(e, ix, min_x[ix], max_x[ix])
                if ix == no_of_descriptors-1:
                    ax0 = fig.add_subplot(gs[ix,ie])
                    ax0.hist(flatten_array[e][:,ix], bins=100, label=label)
                    ax0.legend(loc=1)
                    ax0.yaxis.set_major_formatter(mticker.NullFormatter())
                    ax0.set_xlim([0,1])
                else:
                    ax = fig.add_subplot(gs[ix,ie], sharex=ax0)
                    ax.hist(flatten_array[e][:,ix], bins=100, label=label)
                    ax.legend(loc=1)
                    ax.yaxis.set_major_formatter(mticker.NullFormatter())
                    plt.setp(ax.get_xticklabels(), visible=False)
            print("\n")
        plt.subplots_adjust(hspace=.0)
        #plt.tight_layout()
        plt.savefig(figname)
        plt.close()


    def collate_fn(self, batch):
        """ Return user-defined batch. """
        return batch


class Dataset(data.Dataset):
    """ Defined a Dataset class based on PyTorch Dataset. 

    Tutorial:
    https://pytorch.org/tutorials/beginner/data_loading_tutorial.html.
    """
    def __init__(self, x, dxdr, rdxdr, energy, force, stress, softmax):
        self.x = x
        self.dxdr = dxdr
        self.energy = energy
        self.force = force
        self.softmax = softmax
        self.rdxdr = rdxdr
        self.stress = stress


    def __len__(self):
        return len(self.x)


    def __getitem__(self, index):
        x = self.x[index]
        dxdr = self.dxdr[index]
        rdxdr = self.rdxdr[index]
        energy = self.energy[index]
        force = self.force[index]
        sf = self.softmax[index]
        stress = self.stress[index]

        return x, dxdr, rdxdr, energy, force, stress, sf
