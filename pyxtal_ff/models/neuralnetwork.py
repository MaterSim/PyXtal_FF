#!/usr/bin/env  python
# encoding: utf-8
import os
import gc
import time
import shelve
import numpy as np
from copy import deepcopy
from random import randint

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
from pyxtal_ff.utilities.elements import Element

eV2GPa = 160.21766

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
    restart: str
        Continuing Neural Network training from where it was left off.
    path: str
        A path to the directory where everything is saved.
    memory: str
        There are two options: 'in' or 'out'. 'in' will use load all
        descriptors to memory as 'out' will call from disk as needed.
    """
    def __init__(self, elements, hiddenlayers, activation, random_seed, 
                 batch_size, epoch, device, alpha, softmax_beta, unit, 
                 force_coefficient, stress_coefficient, stress_group,
                 restart, path, memory):
        
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
        
        self.restart = restart
        self.path = path
        self.memory = memory

        self.drange = None


    def train(self, TrainData, optimizer):
        """ Training of Neural Network Potential. """
        # If batch_size is None and optimizer is Adam or SGD, 
        # then batch_size equals total structures.
        if optimizer['method'] in ['sgd', 'SGD', 'Adam', 'adam', 'ADAM']:
            if self.batch_size == None:
                db = shelve.open(self.path+TrainData)
                self.batch_size = len(db.keys())
                db.close()

        self.preprocess(TrainData)
        
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
                
        print(f"No of structures   : {self.no_of_structures}")
        print(f"No of descriptors  : {self.no_of_descriptors}")
        print(f"No of parameters   : {self.total_parameters}")
        print(f"No of epochs       : {self.epoch}")
        print(f"Optimizer          : {optimizer['method']}")
        print(f"Force_coefficient  : {self.force_coefficient}")
        print(f"Stress_coefficient : {self.stress_coefficient}")
        print(f"Batch_size         : {self.batch_size}\n")

        # Run Neural Network Potential Training
        t0 = time.time()
        for epoch in range(self.epoch):
            print("\nEpoch {:4d}: ".format(epoch+1))
            if optimizer['method'] in ['lbfgs', 'LBFGS', 'lbfgsb']:
                if self.batch_size is None: #Full batch
                    print("Initial state : ")
                    def closure(): # LBFGS gets loss and its gradient here.
                        train_loss, E_mae, F_mae, S_mae = self.calculate_loss(self.models, self.data)
                        print("    Loss: {:10.6f}     Energy MAE: {:10.4f}     Force MAE: {:10.4f}     Stress MAE: {:10.4f}".\
                                format(train_loss, E_mae, F_mae, S_mae))
                        self.optimizer.zero_grad()
                        train_loss.backward()
                        return train_loss
                    self.optimizer.step(closure)
                else: #mini batch #10 batches, id [0, 9], id=0, 1, 2, 3, 4, self.data[4]
                    for i, batch in enumerate(self.data):
                        if i == 0:
                            def closure(): # LBFGS gets loss and its gradient here.
                                train_loss, E_mae, F_mae, S_mae = self.calculate_loss(self.models, batch)
                                print("    Loss: {:10.6f}     Energy MAE: {:10.4f}     Force MAE: {:10.4f}     Stress MAE: {:10.4f}".\
                                    format(train_loss, E_mae, F_mae, S_mae))
                                self.optimizer.zero_grad()
                                train_loss.backward() 
                                return train_loss
                            self.optimizer.step(closure)
                            break



            elif optimizer['method'] in ['sgd', 'SGD', 'Adam', 'adam', 'ADAM']:
                for batch in self.data:
                    train_loss, E_mae, F_mae, S_mae = self.calculate_loss(self.models, batch)
                    self.optimizer.zero_grad()
                    train_loss.backward()
                    self.optimizer.step()
                    print("    Loss: {:10.6f}     Energy MAE: {:10.4f}     Force MAE: {:10.4f}     Stress MAE: {:10.4f}".\
                            format(train_loss, E_mae, F_mae, S_mae))
                                        
        t1 = time.time()

        self.data.dataset.close()
        print("\nThe training time: {:.2f} s".format(t1-t0))
        

    def evaluate(self, data, figname):
        """ Evaluating the train or test data set based on trained Neural Network model. 
        Evaluate will only be performed in cpu mode. """
        self.normalize(data, self.drange, self.unit)

        # Switch models device to cpu if training is done in cuda.
        models = {}
        for element in self.elements:
            if next(self.models[element].parameters()).is_cuda:
                models[element] = self.models[element].cpu()
            else:
                models[element] = self.models[element]

        db = shelve.open(self.path+data+'_norm')
        db2 = shelve.open(self.path+data)

        # Predicting the data set
        _energy, _force, _stress = [], [], [] # Predicted energy and forces
        energy, force, stress = [], [] ,[]
        formulas = []
        for item, value in db.items():
            data2 = db2[item]

            dedx = None
            n_atoms = sum(len(value) for value in value['x'].values())
            _Energy = 0.
            _Force = torch.zeros([n_atoms, 3], dtype=torch.float64)
            if self.stress_coefficient and (data2['group'] in self.stress_group):
                _Stress = torch.zeros([6], dtype=torch.float64)

            formula = ""
            for element, model in models.items():
                formula += element
                formula += str(len(value['x'][element]))
                formula += " "

                if value['x'][element].nelement() > 0:
                    _x = value['x'][element].requires_grad_()
                    _e = model(_x).sum()
                    _Energy += _e
                    if self.force_coefficient:
                        dedx = torch.autograd.grad(_e, _x)[0]
                        _dxdr = value['dxdr'][element]

                        tmp = np.zeros([len(value['x'][element]), n_atoms, value['x'][element].shape[1], 3])
                        for _m in range(n_atoms):
                            rows = np.where(value['seq'][element][:,1]==_m)[0]
                            tmp[value['seq'][element][rows, 0], _m, :, :] += _dxdr[rows, :, :]
                        _Force -= torch.einsum("ik, ijkl->jl", dedx, torch.from_numpy(tmp)) 

                    if self.stress_coefficient and (data2['group'] in self.stress_group):
                        if self.force_coefficient is None:
                            dedx = torch.autograd.grad(_e, _x)[0]
                        _rdxdr = value['rdxdr'][element]
                        _Stress += -1 * torch.einsum("ik, ikl->l", dedx, _rdxdr)
            
            formulas.append(formula)
            energy.append(data2["energy"]/len(data2["elements"]))
            force.append(np.ravel(data2["force"]))
            
            _energy.append(_Energy.item()/n_atoms)
            _force.append(np.ravel(_Force.numpy()))

            if self.stress_coefficient and (data2['group'] in self.stress_group):
                _stress.append(np.ravel(_Stress*eV2GPa)) #transform to GPa
                stress.append(np.array(data2['stress']))

                # print outliers for debug
                a = np.ravel(_Stress*eV2GPa)
                b = np.array(data2['stress'])
                #if self.mean_absolute_error(a[2:], b[2:]) > 1.5:
                #print('{:12s} {:6.3f} {:6.3f} {:6.3f} {:6.3f} {:6.3f} {:6.3f}'.format(formula, a[0], a[1], a[2], a[3], a[4], a[5]))
                #print('{:12s} {:6.3f} {:6.3f} {:6.3f} {:6.3f} {:6.3f} {:6.3f}'.format('DFT:',  b[0], b[1], b[2], b[3], b[4], b[5]))
                #import sys; sys.exit()


        db.close()
        db2.close()

        energy = np.array(energy)
        _energy = np.array(_energy)
        if self.force_coefficient:
            force = np.array([x for i in force for x in i])
            _force = np.array([x for i in _force for x in i])
        if self.stress_coefficient:
            stress = np.ravel(stress)
            _stress = np.ravel(_stress)
        
        # Dump the true and predicted values into text file.
        self.dump_evaluate(_energy, energy, filename=figname[:-4]+'Energy.txt')
        self.dump_formulas(formulas, filename=figname[:-4]+'formulas.txt')
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
        energy_str = 'Energy: r2({:.4f}), MAE({:.4f} {}/atom), {:d} data'. \
                     format(E_r2, E_mae, self.unit, len(energy))
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
            force_str = 'Force: r2({:.4f}), MAE({:.3f} {}/{}), {:d} data'. \
                        format(F_r2, F_mae, self.unit, length, len(force))
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
            stress_str = 'Stress: r2({:.4f}), MAE({:.3f} GPa), {:d} data'. \
                        format(S_r2, S_mae, len(stress))
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


    def preprocess(self, TrainData):
        """ Preprocess TrainData to a convenient format for Neural Network training. """

        if os.path.exists(self.path+"drange.npy"):
            self.drange = np.load(self.path+"drange.npy", allow_pickle=True)[0]
        else:
            self.drange = self.get_descriptors_range(TrainData)
        #self.plot_hist(descriptors, figname=self.path+"histogram.png", figsize=(12, 24))

        self.normalize(TrainData, self.drange, self.unit)

        self.get_stress_group(TrainData)

        self.softmax = self._SOFTMAX(TrainData, beta=self.softmax_beta)

        self.data = data.DataLoader(Dataset(self.path+TrainData, self.softmax,
                                            self.device, self.memory, 
                                            self.force_coefficient, 
                                            self.stress_coefficient),
                                    batch_size=self.batch_size,
                                    shuffle=self.shuffle,
                                    collate_fn=self.collate_fn,)

        gc.collect()
       

    def calculate_loss(self, models, batch):
        """ Calculate the total loss and MAE for energy and forces
        for a batch of structures per one optimization step. """ 

        energy_loss, force_loss, stress_loss = 0., 0., 0.
        energy_mae, force_mae, stress_mae = 0., 0., 0.
        all_atoms = 0
        s_count = 0

        for x, dxdr, seq, rdxdr, energy, force, stress, sf, group in batch:
            n_atoms = sum(len(value) for value in x.values())
            all_atoms += n_atoms
            _Energy = 0  # Predicted total energy for a structure
            _force = torch.zeros([n_atoms, 3], dtype=torch.float64, device=self.device)
            if self.stress_coefficient and (group in self.stress_group):
                _stress = torch.zeros([6], dtype=torch.float64, device=self.device)
            
            dedx, sdedx = {}, {}
            for element, model in models.items():
                if x[element].nelement() > 0:
                    _x = x[element].requires_grad_()
                    _energy = model(_x).sum() # total energy for each specie
                    _Energy += _energy

                    if self.force_coefficient:
                        dedx[element] = torch.autograd.grad(_energy, _x, create_graph=True)[0]
                        tmp = np.zeros([len(x[element]), n_atoms, x[element].shape[1], 3])
                        for _m in range(n_atoms):
                            rows = np.where(seq[element][:,1]==_m)[0]
                            tmp[seq[element][rows, 0], _m, :, :] += dxdr[element][rows, :, :]
                        _force -= torch.einsum("ik, ijkl->jl", dedx[element], torch.from_numpy(tmp)) 

                    if self.stress_coefficient and (group in self.stress_group):
                        if self.force_coefficient is None:
                            dedx[element] = torch.autograd.grad(_energy, _x, create_graph=True)[0]
                        _stress += -1 * torch.einsum("ik, ikl->l", dedx[element], rdxdr[element]) # [6]
            
            energy_loss += sf.item()*((_Energy - energy) / n_atoms) ** 2
            energy_mae  += sf.item()*F.l1_loss(_Energy / n_atoms, energy / n_atoms)

            if self.force_coefficient:
                force_loss += sf.item()*self.force_coefficient * ((_force - force) ** 2).sum()
                force_mae  += sf.item()*F.l1_loss(_force, force) * n_atoms

            if self.stress_coefficient and (group in self.stress_group):
                _stress *= eV2GPa

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
        reg = 0.
        if self.alpha: 
            for element, model in models.items():
                for name, params in model.named_parameters():
                    if 'weight' in name:
                        reg += self.alpha * params.pow(2).sum()
            loss += reg

        print("eng_loss: {:10.6f}     force_loss: {:10.6f}   stress_loss: {:10.6f}  regularization: {:10.6f}".format(\
                    energy_loss, force_loss, stress_loss, reg))

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


    def dump_formulas(self, formulas, filename):
        """ Dump the evaluate results to text files. """
        with open(self.path+filename, 'w') as f:
            for formula in formulas:
                f.write(formula+'\n')

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
        no_of_descriptors = descriptor['x'].shape[1]
        energies, force, stress = np.zeros([no_of_atoms]), np.zeros([no_of_atoms, 3]), np.zeros([6])
        
        # Normalizing
        d = {'x': {}, 'dxdr': {}, 'seq': {}, 'rdxdr': {}}
        for element in self.elements:
            _drange = self.drange[element]
            scale = (1 - 0) / (_drange[:, 1] - _drange[:, 0])

            e = np.where(np.array(descriptor['elements'])==element)[0]
            ee0 = np.where(descriptor['seq'][:,0]==e[0])[0][0]
            ee1 = np.where(descriptor['seq'][:,0]==e[-1])[0][-1]

            i_size = list(descriptor['elements']).count(element) #number of atoms in type i
            m_size = ee1-ee0+1# number of pairs

            d['x'][element] = torch.zeros([i_size, no_of_descriptors])
            d['dxdr'][element] = torch.zeros([m_size, no_of_descriptors, 3], dtype=torch.float64) ####
            d['rdxdr'][element] = torch.zeros([i_size, no_of_descriptors, 6])

            if e.size > 0:
                des = 0 + np.einsum('j,ij->ij', scale, (descriptor['x'][e[0]:e[-1]+1] - np.expand_dims(_drange[:, 0], 0)))
                desp = np.einsum('j,ijk->ijk', scale, descriptor['dxdr'][ee0:ee1+1])
                dess = np.einsum('j,ijk->ijk', scale, descriptor['rdxdr'][e[0]:e[-1]+1])
                
                d['x'][element] += torch.from_numpy(des)
                d['seq'][element] = deepcopy(descriptor['seq'][ee0:ee1+1])

                if self.unit == 'eV':  
                    d['dxdr'][element] += torch.from_numpy(desp)
                    d['rdxdr'][element] += torch.from_numpy(dess)
                else:
                    d['dxdr'][element] += torch.from_numpy(0.529177 * desp)
                    d['rdxdr'][element] += torch.from_numpy(0.529177**3*desp)
            d['seq'][element][:, 0] -= min(d['seq'][element][:, 0]) #adjust the initial position

        x = d['x']
        if bforce:
            dxdr = d['dxdr']
        if bstress:
            rdxdr = d['rdxdr']
        
        count = 0
        for element, model in self.models.items():
            if element in x.keys():
                _x = x[element].requires_grad_()
                if bforce:
                    _dxdr = dxdr[element]
                if bstress:
                    _rdxdr = rdxdr[element]
                _e = model(_x).sum()
                #energy += _e.detach().numpy() 
                energies[count:count+len(_x)] = model(_x).detach().numpy().reshape([len(_x)])
                #print(element, model(_x).detach().numpy())
                count += len(_x)

                
                if bforce:
                    dedx = torch.autograd.grad(_e, _x)[0]
                    shp = dedx.shape
                    #force += -torch.einsum("ik, ijkl->jl", dedx, _dxdr).numpy()
                    if no_of_atoms > 400: 
                        for _m in range(no_of_atoms):
                            ids = np.where(d['seq'][element][:,1]==_m)[0]
                            tmp = _dxdr[ids, :, :]
                            _dedx = dedx[d['seq'][element][ids][:,0]]
                            force[_m, :] -= torch.einsum("ij, ijk->k", _dedx, tmp).numpy()
                    else:
                        tmp = torch.zeros([len(d['x'][element]), no_of_atoms, d['x'][element].shape[1], 3])
                        for _m in range(no_of_atoms):
                            ids = np.where(d['seq'][element][:,1]==_m)[0]
                            tmp[d['seq'][element][ids, 0], _m, :, :] += _dxdr[ids, :, :]
                        force -= torch.einsum("ik, ijkl->jl", dedx, tmp).numpy() 

                if bstress:
                    if bforce == False:
                        dedx = torch.autograd.grad(_e, _x)[0]
                    stress += -torch.einsum("ik, ikl->l", dedx, _rdxdr).numpy()
        return energies, force, stress*eV2GPa


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
        if des_info['type'] in ['SNAP', 'snap', 'SO3', 'SOAP']:
            self.save_weights_to_txt(des_info)
        print("The Neural Network Potential is exported to {:s}".format(_filename))
        print("\n")


    def save_weights_to_txt(self, des_info):
        """ Saving the model weights to txt file. """
        with open(self.path+"NN_weights.txt", "w") as f:
            f.write("# Neural networks weights generated in PyXtal_FF \n")
            f.write("# total_species nparams \n")
            f.write(f"{len(self.elements)} ")
            for e, element in enumerate(self.elements):
                model = self.models[element]
                count = 0
                for parameters in model.parameters():
                    if len(parameters.shape) == 1:
                        count += len(parameters)
                    else:
                        for param in parameters:
                            count += len(param)
            f.write(f"{count} \n\n")
            
            for e, element in enumerate(self.elements):
                if e == 0:
                    f.write("# NET ndescriptors nlayers activation_func for layer 1, number of nodes for layer 1, ..., ..., ...., \n")
                    f.write(f"NET {self.no_of_descriptors} {len(self.hiddenlayers[element])} ")
                model = self.models[element]

                _PARAMETERS = []
                for parameters in model.parameters():
                    _PARAMETERS.append(parameters)
                    
                _parameters = []
                for i, hl in enumerate(self.hiddenlayers[element]):
                    if e == 0:
                        f.write(f"{self.activation[element][i]} ".lower())
                        f.write(f"{hl} ")
                    _parameters.append(torch.cat([_PARAMETERS[2*i+1][:,None], _PARAMETERS[2*i]], dim=1))

                f.write("\n\n")

                drange = self.drange[element]
                dmins = drange[:,0]
                dmaxs = drange[:,1] - drange[:,0]
                str_min, str_max = "", ""
                
                cnt = 1
                for dmin, dmax in zip(dmins, dmaxs):
                    str_min += "{:20.13f} ".format(dmin)
                    str_max += "{:20.13f} ".format(dmax)
                    if cnt%5==0 or cnt==len(dmins):
                        str_min += "\n"
                        str_max += "\n"
                    cnt += 1

                f.write("# Scale0 \n")
                f.write(str_min)
                f.write("\n")
                f.write("# Scale1 \n")
                f.write(str_max)
                f.write("\n")

                cnt = 1
                f.write("# Coefficients ({:s})\n".format(element))
                for parameter in _parameters:
                    for param in parameter:
                        for p in param:
                            if cnt%5==0:
                                f.write("{:18.13f} \n".format(p))
                            else:
                                f.write("{:18.13f} ".format(p))
                            cnt += 1
                
                if (cnt-1)%5==0:
                    f.write("\n")
                else:
                    f.write("\n\n")

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

        self.models = checkpoint['models']

        if method:
            if len(self.elements) != len(checkpoint['elements']):
                raise ValueError(msg)
            
            for i in range(len(self.elements)):
                if self.elements[i] != checkpoint['elements'][i]:
                    raise ValueError(msg)

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
    
    
    def get_descriptors_range(self, data):
        """ Calculate the range (min and max values) of the descriptors 
        corresponding to all of the crystal structures.
        
        Parameters
        ----------
        data: dict
            data contains atom-centered descriptors.
            
        Returns
        -------
        dict
            The ranges of the descriptors for each chemical specie.
        """
        if not os.path.exists(self.path+'drange.npy'):
            _DRANGE = {}
            db = shelve.open(self.path+data)
            no_of_structures = len(list(db.keys()))

            for i in range(no_of_structures):
                dic = db[str(i)]
                for j, descriptor in enumerate(dic['x']):
                    element = dic['elements'][j]
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
            db.close()
            np.save(self.path+'drange.npy', [_DRANGE])

        else:
            _DRANGE = np.load(self.path+'drange.npy', allow_pickle=True)[0]

        return _DRANGE

    
    def normalize(self, data, drange, unit, norm=[0., 1.]):
        """ Normalizing the descriptors to the range of [0., 1.] based on the
        min and max value of the entire descriptors.

        Example:
        X.shape == [60, 10]; len(self.elements) == 2
        X_norm -> {'element1': [40, 10], 'element2': [20, 10]}
        
        Parameters
        ----------
        data: str
            The directory path to the database
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
        db1 = shelve.open(self.path+data)
        self.no_of_structures = len(list(db1.keys()))
        self.no_of_descriptors = db1['0']['x'].shape[1]
        doit = True if not os.path.exists(self.path+data+'_norm.dat') else False
        
        if doit:
            db2 = shelve.open(self.path+data+'_norm')

            for i in range(self.no_of_structures):
                d = {'x': {}, 'dxdr': {}, 'seq': {}, 'rdxdr': {}}
                descriptor = db1[str(i)]

                for element in self.elements:
                    _drange = drange[element]
                    scale = (norm[1] - norm[0]) / (_drange[:, 1] - _drange[:, 0])
                    e = np.where(np.array(descriptor['elements'])==element)[0]
                    try:
                        ee0 = np.where(descriptor['seq'][:,0]==e[0])[0][0]
                        ee1 = np.where(descriptor['seq'][:,0]==e[-1])[0][-1]
                    except:
                        ee0, ee1 = 0, 1
                    
                    i_size = list(descriptor['elements']).count(element) #number of atoms in type i
                    m_size = ee1-ee0+1# number of pairs

                    d['x'][element] = torch.zeros([i_size, self.no_of_descriptors], dtype=torch.float64)
                    #d['dxdr'][element] = torch.zeros([m_size, self.no_of_descriptors, 3], dtype=torch.float64) ####
                    d['dxdr'][element] = np.zeros([m_size, self.no_of_descriptors, 3]) ####
                    d['rdxdr'][element] = torch.zeros([i_size, self.no_of_descriptors, 6], dtype=torch.float64)
                    d['seq'][element] = None

                    if e.size > 0:
                        des = norm[0] + np.einsum('j,ij->ij', scale, (descriptor['x'][e[0]:e[-1]+1] - np.expand_dims(_drange[:, 0], 0)))
                        dess = np.einsum('j,ijk->ijk', scale, descriptor['rdxdr'][e[0]:e[-1]+1])
                        desp = np.einsum('j,ijk->ijk', scale, descriptor['dxdr'][ee0:ee1+1])

                        d['x'][element] += torch.from_numpy(des)
                        d['seq'][element] = deepcopy(descriptor['seq'][ee0:ee1+1])
                        #d['seq'][element] = torch.from_numpy(descriptor['seq'][ee0:ee1+1])

                        if unit == 'eV':
                            #d['dxdr'][element] += torch.from_numpy(desp)
                            d['dxdr'][element] += desp
                            d['rdxdr'][element] += torch.from_numpy(dess)
                        else:
                            d['dxdr'][element] += torch.from_numpy(0.529177 * desp)
                            d['rdxdr'][element] += torch.from_numpy(0.529177 * dess)
                    #d['seq'][element][:, 0] -= torch.min(d['seq'][element][:, 0]) #adjust the initial position
                        d['seq'][element][:, 0] -= np.min(d['seq'][element][:, 0]) #adjust the initial position
                
                db2[str(i)] = d
                
            db1.close()
            db2.close()


    def get_stress_group(self, data):
        """ Get every kind of stress groups if None is defined by the user. """
        if self.stress_coefficient and (self.stress_group is None):
            db = shelve.open(self.path+data)
            sg = []
            for i in range(self.no_of_structures):
                if db[str(i)]['group'] not in sg:
                    sg.append(db[str(i)]['group'])
            self.stress_group = sg


    def _SOFTMAX(self, data, beta=-1):
        """ Assign the weight to each sample based on the softmax function. """
        db = shelve.open(self.path+data)

        # Length of smax is equal to the number of samples.
        smax = torch.ones(len(db.keys()))

        if beta is not None:
            epa = []
            for i in range(self.no_of_structures):
                _epa = db[str(i)]['energy'][0]/len(db[str(i)]['elements'])
                epa.append(_epa)
            epa = torch.DoubleTensor(epa)

            smax = torch.exp(beta*epa) / sum(torch.exp(beta*epa))
            smax *= len(smax)    # what's this for?

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
    def __init__(self, data, softmax, device, memory, fc, sc):
        self.softmax = softmax
        self.device = device
        self.memory = memory
        self.fc, self.sc = fc, sc
        
        db1 = shelve.open(data)
        self.db2 = shelve.open(data+'_norm')
        self.length = len(list(db1.keys()))

        if memory == 'in':
            self.x, self.dxdr, self.seq, self.rdxdr = [], [], [], []
            self.energy, self.force, self.stress, self.group = [], [], [], []

            for i in range(self.length):
                data1 = db1[str(i)]
                data2 = self.db2[str(i)]
                self.energy.append(float(data1['energy']))
                self.group.append(data1['group'])

                if self.fc:
                    self.force.append(torch.DoubleTensor(data1['force']).to(self.device))
                else:
                    self.force.append(None)

                if self.sc and data1['stress'] is not None:
                    self.stress.append(torch.from_numpy(data1['stress']).to(self.device))
                else:
                    self.stress.append(None)
                
                x, dx, seq, rdx = {}, {}, {}, {}
                for k in data2['x'].keys(): #["Si", "O"]
                    x[k] = data2['x'][k].to(self.device)
                    #dx[k] = data2['dxdr'][k].to(self.device) if self.fc else None
                    dx[k] = data2['dxdr'][k] if self.fc else None
                    seq[k] = data2['seq'][k] if self.fc else None
                    #seq[k] = data2['seq'][k].to(self.device) if self.fc else None
                    rdx[k] = data2['rdxdr'][k].to(self.device) if (self.sc and data1['stress'] is not None) else None

                self.x.append(x)
                if self.fc:
                    self.dxdr.append(dx)
                    self.seq.append(seq)
                else:
                    self.dxdr.append(None)
                    self.seq.append(None)
                if self.sc and data1['stress'] is not None:
                    self.rdxdr.append(rdx)
                else:
                    self.rdxdr.append(None)

            self.energy = torch.DoubleTensor(self.energy).to(self.device)

        else:
            self.x, self.dxdr, self.seq, self.rdxdr = None, None, None, None
            self.energy, self.force, self.stress, self.group = [], [], [], []

            for i in range(len(list(db1.keys()))):
                data1 = db1[str(i)]
                self.energy.append(float(data1['energy']))
                self.group.append(data1['group'])
                
                if self.fc:
                    self.force.append(torch.DoubleTensor(data1['force']).to(self.device))
                else:
                    self.force.append(None)

                if self.sc and data1['stress'] is not None:
                    self.stress.append(torch.DoubleTensor(data1['stress']).to(self.device))
                else:
                    self.stress.append(None)

            self.energy = torch.DoubleTensor(self.energy).to(self.device)

        db1.close()
        

    def __len__(self):
        return self.length


    def __getitem__(self, index):
        energy = self.energy[index]
        force = self.force[index]
        stress = self.stress[index]
        
        sf = self.softmax[index]
        group = self.group[index]
        
        if self.memory == 'in':
            x = self.x[index]
            dxdr = self.dxdr[index]
            seq = self.seq[index]
            rdxdr = self.rdxdr[index]

        else:
            data = self.db2[str(index)]
            x, dxdr, seq, rdxdr = data['x'], data['dxdr'], data['seq'], data['rdxdr']
            if self.device == 'cuda':
                for k in x.keys():
                    x[k] = x[k].to(self.device)
                    #dxdr[k] = dxdr[k].to(self.device)
                    rdxdr[k] = rdxdr[k].to(self.device)

        return x, dxdr, seq, rdxdr, energy, force, stress, sf, group


    def close(self,):
        self.db2.close()
