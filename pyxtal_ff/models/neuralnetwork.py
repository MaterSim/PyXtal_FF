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
        This parameter is used in the penalty function to scale the force
        contribution relative to the energy.
    alpha: float
        L2 penalty (regularization) parameter.
        softmax_beta:
        The parameters for Softmax Energy Penalty function.
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
                 batch_size, epoch, device, force_coefficient, alpha, 
                 softmax_beta, unit, logging, restart, path):
        
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
        

    def train(self, TrainDescriptors, TrainFeatures, optimizer):
        """ Training of Neural Network Potential. """
        
        # If batch_size is None and optimizer is Adam or SGD, 
        # then batch_size equals total structures.
        if optimizer['method'] in ['sgd', 'SGD', 'Adam', 'adam', 'ADAM']:
            if self.batch_size == None:
                self.batch_size = len(TrainDescriptors)

        # Preprocess descriptors and features.
        self.preprocess(TrainDescriptors, TrainFeatures)

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

        # Set-up optimizer for optimizing NeuralNetwork weights.
        self.regressor = Regressor(optimizer['method'], optimizer['parameters'])
        self.optimizer = self.regressor.regress(models=self.models)
        
        # Look for previously saved models and continue optimizing from the last checkpoint.
        if self.restart:
            self.load_checkpoint(filename=self.restart, 
                                 method=optimizer['method'], args=optimizer['parameters'])
        
        print("==================================== Training ====================================")
        print("\n")
        print(f"Optimizer         : {optimizer['method']}")
        print(f"Force_coefficient : {self.force_coefficient}\n")

        # Run Neural Network Potential Training
        t0 = time.time()
        for epoch in range(self.epoch):
            if optimizer['method'] in ['lbfgs', 'LBFGS', 'lbfgsb']:
                print("Initial state : ")
                def closure(): # LBFGS gets loss and its gradient here.
                    train_loss, E_mae, F_mae = self.calculate_loss(self.models, self.data)
                    print("    Loss: {:10.6f}     Energy MAE: {:10.4f}     Force MAE: {:10.4f}".\
                            format(train_loss, E_mae, F_mae))
                    self.optimizer.zero_grad()
                    train_loss.backward()
                    return train_loss
                self.optimizer.step(closure)

            elif optimizer['method'] in ['sgd', 'SGD', 'Adam', 'adam', 'ADAM']:
                if epoch != 0:
                    print("\nIteration {:4d}: ".format(epoch+1))
                    for batch in self.data:
                        train_loss, E_mae, F_mae = self.calculate_loss(self.models, batch)
                        self.optimizer.zero_grad()
                        train_loss.backward()
                        self.optimizer.step()
                        print("    Loss: {:10.6f}     Energy MAE: {:10.4f}     Force MAE: {:10.4f}".\
                                format(train_loss, E_mae, F_mae))
                else:
                    print("Initial state : ")
                    train_loss, E_mae, F_mae = 0., 0., 0.
                    for batch in self.data:
                        tl, Emae, Fmae = self.calculate_loss(self.models, batch)
                        train_loss += tl
                        E_mae += Emae
                        F_mae += Fmae
                    print("    Loss: {:10.6f}     Energy MAE: {:10.4f}     Force MAE: {:10.4f}".\
                            format(train_loss, E_mae, F_mae))
                    
        t1 = time.time()
        print("\nThe training time: {:.2f} s".format(t1-t0))
        
        self.save_checkpoint()
        print("\n============================== Training is Completed =============================\n")

            
    def evaluate(self, TestDescriptors, TestFeatures, figname):
        """ Evaluating the train or test data set based on trained Neural Network model. 
        Evaluate will only be performed in cpu mode. """
        
        # If-else for consistent separations in printing.
        if figname[:-4] == 'Train':
            print(f"============================= Evaluating {figname[:-4]}ing Set ============================\n")
        else:
            print("============================= Evaluating Testing Set =============================\n")
        
        # Normalized data set based on the training data set (drange).
        TestDescriptors = self.normalized(TestDescriptors, self.drange, self.unit)
        no_of_structures = TestDescriptors['no_of_structures']

        # Parse descriptors and Features
        energy, force = [], []
        X = [{} for _ in range(len(TestDescriptors['x']))]
        if self.force_coefficient:
            DXDR = [{} for _ in range(len(TestDescriptors['dxdr']))]
        else:
            DXDR = [None]*len(X)

        for i in range(no_of_structures):
            energy.append(TestFeatures[i]['energy']/len(TestFeatures[i]['force']))
            force.append(np.ravel(TestFeatures[i]['force']))
            for element in self.elements:
                X[i][element] = torch.DoubleTensor(TestDescriptors['x'][i][element])
                if self.force_coefficient:
                    DXDR[i][element] = torch.DoubleTensor(TestDescriptors['dxdr'][i][element])
        
        # Switch models device to cpu if training is done in cuda.
        models = {}
        for element in self.elements:
            if next(self.models[element].parameters()).is_cuda:
                models[element] = self.models[element].cpu()
            else:
                models[element] = self.models[element]
        
        # Predicting the data set.
        _energy, _force = [], [] # Predicted energy and forces
        test_data = zip(X, DXDR)
        for x, dxdr in test_data:
            n_atoms = sum(len(value) for value in x.values())
            _Energy = 0.
            _Force = torch.zeros([n_atoms, 3], dtype=torch.float64)

            for element, model in models.items():
                if x[element].nelement() > 0:
                    _x = x[element].requires_grad_()
                    _e = model(_x).sum()
                    _Energy += _e
                    if self.force_coefficient:
                        dedx = torch.autograd.grad(_e, _x)[0]
                        _Force += -torch.einsum("ik, ijkl -> jl", dedx, dxdr[element])
            _force.append(np.ravel(_Force.numpy()))
            _energy.append(_Energy.item()/n_atoms)
        
        energy = np.array(energy)
        _energy = np.array(_energy)
        force = np.array([x for i in force for x in i])
        _force = np.array([x for i in _force for x in i])
        
        # Dump the true and predicted values into text file.
        self.dump_evaluate(_energy, energy, filename=figname[:-4]+'Energy.txt')
        if self.force_coefficient:
            self.dump_evaluate(_force, force, filename=figname[:-4]+'Force.txt')

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

        print("============================= Evaluation is Completed ============================")
        print("\n")
        
        return (E_mae, E_mse, E_r2, F_mae, F_mse, F_r2)


    def preprocess(self, descriptors, features):
        """ Preprocess the descriptors and features to a convenient format
        for training Neural Network model. """
        self.no_of_descriptors = len(descriptors[0]['x'][0]) 
        self.no_of_structures = len(descriptors)

        # Generate and plot descriptor range.
        self.drange = self.get_descriptors_range(descriptors)
        self.plot_hist(descriptors, figname=self.path+"histogram.png", figsize=(12, 24))

        # Transform descriptors into normalized descriptors
        descriptors = self.normalized(descriptors, self.drange, self.unit)
        
        # Parsing descriptors in Torch mode
        x = [{} for _ in range(len(descriptors['x']))]
        if self.force_coefficient:
            dxdr = [{} for _ in range(len(descriptors['dxdr']))]
        else:
            dxdr = [None]*len(x)

        for i in range(self.no_of_structures):
            for element in self.elements:
                x[i][element] = torch.DoubleTensor(descriptors['x'][i][element], ).to(self.device)
                if self.force_coefficient:
                    dxdr[i][element] = torch.DoubleTensor(descriptors['dxdr'][i][element]).to(self.device)
        
        del(descriptors)
        gc.collect() # Flush memory

        # Parse Energy & Forces
        energy = []
        force = []
        for i in range(self.no_of_structures):
            energy.append(features[i]['energy'])
            force.append(torch.DoubleTensor(features[i]['force']).to(self.device))
        energy = torch.DoubleTensor(energy).to(self.device)

        # Emphazising the importance of lower Energy mode.
        softmax = self._SOFTMAX(energy, x, beta=self.softmax_beta)
        
        # Compile x, dxdr, energy, force, and softmax into PyTorch DataLoader.
        self.data = data.DataLoader(Dataset(x, dxdr, energy, force, softmax),
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

        energy_loss, force_loss = 0., 0.
        energy_mae, force_mae = 0., 0.
        all_atoms = 0

        for x, dxdr, energy, force, sf in batch:
            n_atoms = sum(len(value) for value in x.values())
            all_atoms += n_atoms
            _Energy = 0  # Predicted total energy for a structure
            _force = torch.zeros([n_atoms, 3], dtype=torch.float64, device=self.device)
            dedx = {}
            for element, model in models.items():
                if x[element].nelement() > 0:
                    _x = x[element].requires_grad_()
                    _energy = model(_x).sum() # total energy for each specie
                    _Energy += _energy

                    if self.force_coefficient:
                        dedx[element] = torch.autograd.grad(_energy, _x, create_graph=True)[0]
                        _force += -torch.einsum("ik, ijkl -> jl", 
                                                dedx[element], dxdr[element]) # [natoms, 3]
            
            energy_loss += sf.item()*((_Energy - energy) / n_atoms) ** 2
            energy_mae  += sf.item()*F.l1_loss(_Energy / n_atoms, energy / n_atoms)

            if self.force_coefficient:
                force_loss += sf.item()*self.force_coefficient * ((_force - force) ** 2).sum()
                force_mae  += sf.item()*F.l1_loss(_force, force) * n_atoms

        energy_loss = energy_loss / (2. * len(batch))
        energy_mae /= len(batch)

        if self.force_coefficient:
            force_loss = force_loss / (2. * all_atoms)
            force_mae /= all_atoms
            loss = energy_loss + force_loss
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
        
        return loss, energy_mae, force_mae


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

        
    def save_checkpoint(self, filename=None):
        """ Save PyTorch Neural Network models at a checkpoint. """
        self.filename = self.path

        if filename:
            self.filename += filename
        else:
            if isinstance(self._hiddenlayers, list):
                _hl = "-".join(str(x) for x in self._hiddenlayers)
                self.filename += _hl + '-checkpoint.pth'
            else:
                count = 0
                for i in range(len(self.elements)):
                    self.filename += "-".join(str(x) \
                        for x in self._hiddenlayers[self.elements[i]])
                    if count < len(self.elements)-1:
                        self.filename += "_"
                self.filename += '-checkpoint.pth'

        checkpoint = {'model': [mode for mode in self.models.values()],
                      'state_dict': [sd.state_dict() for sd in self.models.values()],
                      'optimizer': self.optimizer.state_dict()}
        
        torch.save(checkpoint, self.filename)
        print("The NNP is exported to {:s}".format(self.filename))


    def load_checkpoint(self, filename, method, args):
        """ Load PyTorch Neural Network models at previously saved checkpoint. """
        checkpoint = torch.load(filename)

        for i, element in enumerate(self.elements):
            self.models[element].load_state_dict(checkpoint['state_dict'][i])
            for parameter in self.models[element].parameters():
                parameter.requires_grad = True
        
        # If different optimizer is used in loading, start the opt. from the beginning.
        # Else load the optimizer state.
        if method in ['lbfgs', 'LBFGS', 'lbfgsb']:
            if 'line_search_fn' in checkpoint['optimizer']['param_groups'][0].keys():
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                for key, value in args.items():
                    if key == 'max_eval':
                        if args[key] == None:
                            self.optimizer.param_groups[0][key] = 15000
                        else:
                            self.optimizer.param_groups[0][key] = args[key]
                    else:
                        self.optimizer.param_groups[0][key] = args[key]

        elif method in ['sgd', 'SGD']:
            if 'nesterov' in checkpoint['optimizer']['param_groups'][0].keys():
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                for key, value in args.items():
                    self.optimizer.param_groups[0][key] = args[key]

        elif method in ['adam', 'ADAM', 'Adam']:
            if 'amsgrad' in checkpoint['optimizer']['param_groups'][0].keys():
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                for key, value in args.items():
                    self.optimizer.param_groups[0][key] = args[key]
                    
    
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
    
    
    def normalized(self, descriptors, drange, unit, norm=[0., 1.]):
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
    def __init__(self, x, dxdr, energy, force, softmax):
        self.x = x
        self.dxdr = dxdr
        self.energy = energy
        self.force = force
        self.softmax = softmax


    def __len__(self):
        return len(self.x)


    def __getitem__(self, index):
        x = self.x[index]
        dxdr = self.dxdr[index]
        energy = self.energy[index]
        force = self.force[index]
        sf = self.softmax[index]

        return x, dxdr, energy, force, sf
