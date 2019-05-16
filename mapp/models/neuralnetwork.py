import os
from collections import OrderedDict
import copy

import numpy as np
from ase.calculators.calculator import Parameters

from .model import calculate_descriptor_range
from ..utilities.lregression import Regressor

class NeuralNetwork:
    """This class implements a Neural Network model.

    Parameters
    ----------
    hiddenlayers: tuple
        (3,3) means 2 layers with 3 nodes each.
    activation: str
        The activation function for the neural network.
    feature_mode: bool
        The user input feature. 
        If False, ASE will provide calculator to calculate the energies 
        and forces.
    """
    def __init__(self, hiddenlayers=(3, 3), activation='tanh', elements=['Pt', 'Cu'], feature_mode=False, weights=None):
        p = self.parameters = Parameters()
        
        act_mode = ['tanh', 'sigmoid', 'linear']
        if activation not in act_mode:
            msg = f"{activation} is not implemented here. Please choose from {act_mode}."
            raise NotImplementedError(msg)

        p.hiddenlayers = hiddenlayers
        p.activation = activation
        p.elements = elements
        p.feature_mode = feature_mode
        p.weights = weights


    def fit(self, images, adescriptors, feature=None):
        """
        Fit the model parameters here.

        images: List
            List of ASE atomic objects for neural network training.
        adescriptor: array
            All descriptors of images.
        """
        p = self.parameters

        if p.feature_mode:
            if feature is None:
                msg = f"You must input the feature if the feature_mode is {p.feature_mode}"
            else:
                p.feature = feature
        
        # Convert hiddenlayers to dictionary:
        if isinstance(p.hiddenlayers, (tuple, list)):
            hiddenlayers = {}
            for e in p.elements:
                hiddenlayers[e] = p.hiddenlayers
            p.hiddenlayers = hiddenlayers

        p.images = images
        p.adescriptors = adescriptors
        #p.descriptor_shape = (len(adescriptor[0]), len(adescriptor[0][0][1]))

        p.desrange = calculate_descriptor_range(images, adescriptors)
        p.scalings = self.activation_scaling(images, p.activation, p.desrange.keys())

        if p.weights == None:
            p.weights = self.get_random_weights(p.hiddenlayers, p.descriptor_shape,)

        self.regressor = Regressor()
        self.result = self.regressor.regress(model=self) # return [parameters, loss]

    def calculate_loss(self, parametervector, lossprime=True):
        self.vector = parametervector
        p = self.parameters
        
        energyloss = 0.
        denergyloss_dparameters = np.zeros((len(parametervector),))
        images = p.images
        adescriptors = p.adescriptors

        for i, image in enumerate(images):
            no_of_atoms = len(image)

            true_energy = image.get_potential_energy(apply_constraint=False) # Need to make this provided by the users
            nn_energy = self.calculate_nnEnergy(adescriptors[i])
            residual = nn_energy - true_energy
            energyloss += (residual / no_of_atoms) ** 2

            if lossprime:
                dnnenergy_dparameters = self.calculate_dnnEnergy_dParameters(adescriptors[i])
                denergyloss_dparameters += 2. * residual * dnnenergy_dparameters / no_of_atoms ** 2.


        loss = energyloss # can include weights in this loss too
        dloss_dparameters = denergyloss_dparameters

        self.loss = nn_energy - true_energy

        return loss, dloss_dparameters

    def activation_scaling(self, images, activation, elements):
        """
        To scale the range of activation to the range of actual energies.

        Parameters
        ----------
        images: list
            ASE atom objects.
        activation: str
            The type of activation function.
        elements: list
            List of atomic symbols in str.

        Returns
        -------
        dict
            The scalings parameters, i.e. slope and intercept.
        """
        n_images = len(images)
    
        # Max and min of true energies.
        max_E = max(image.get_potential_energy(apply_constraint=False) for image in images)
        min_E = min(image.get_potential_energy(apply_constraint=False) for image in images)

        for _ in range(n_images):
            image = images[_]
            n_atoms = len(image)
            if image.get_potential_energy(apply_constraint=False) == max_E:
                n_atoms_of_max_E = n_atoms
            if image.get_potential_energy(apply_constraint=False) == min_E:
                n_atoms_of_min_E = n_atoms

        max_E_per_atom = max_E / n_atoms_of_max_E
        min_E_per_atom = min_E / n_atoms_of_min_E

        scaling = {}

        for element in elements:
            scaling[element] = {}
            if activation == 'tanh':
                scaling[element]['intercept'] = (max_E_per_atom + min_E_per_atom) / 2.
                scaling[element]['slope'] = (max_E_per_atom - min_E_per_atom) / 2.
            elif activation == 'sigmoid':
                scaling[element]['intercept'] = min_E_per_atom
                scaling[element]['slope'] = (max_E_per_atom - min_E_per_atom)
            elif activation == 'linear':
                scaling[element]['intercept'] = (max_E_per_atom + min_E_per_atom) / 2.
                scaling[element]['slope'] = (10. ** (-10.)) * (max_E_per_atom - min_E_per_atom) / 2.

        return scaling


    def get_random_weights(self, hiddenlayers, descriptor_shape):
        """(CP)
        Generating random weights for the neural network architecture.

        Returns
        -------
        dict
            Randomly-generated weights.
        """
        rs = np.random.RandomState(seed=13)
        weights = {}
        nn_structure = {}

        elements = hiddenlayers.keys()
        for element in sorted(elements):
            weights[element] = {}
            nn_structure[element] = [descriptor_shape[1]] + [l for l in hiddenlayers[element]] + [1]

            epsilon = np.sqrt(6. / (nn_structure[element][0] +
                                    nn_structure[element][1]))
            normalized_arg_range = 2. * epsilon
            weights[element][1] = (rs.random_sample(
                (descriptor_shape[1] + 1, nn_structure[element][1])) *
                normalized_arg_range - normalized_arg_range / 2.)
            len_of_hiddenlayers = len(list(nn_structure[element])) - 3
            for layer in range(len_of_hiddenlayers):
                epsilon = np.sqrt(6. / (nn_structure[element][layer + 1] +
                                        nn_structure[element][layer + 2]))
                normalized_arg_range = 2. * epsilon
                weights[element][layer + 2] = rs.random_sample(
                    (nn_structure[element][layer + 1] + 1,
                     nn_structure[element][layer + 2])) * \
                    normalized_arg_range - normalized_arg_range / 2.

            epsilon = np.sqrt(6. / (nn_structure[element][-2] +
                                    nn_structure[element][-1]))
            normalized_arg_range = 2. * epsilon
            weights[element][len(list(nn_structure[element])) - 1] = \
                rs.random_sample((nn_structure[element][-2] + 1, 1)) \
                * normalized_arg_range - normalized_arg_range / 2.

            if False:  # This seemed to be setting all biases to zero?
                len_of_weight = len(weights[element])
                for _ in range(len_of_weight):  # biases
                    size = weights[element][_ + 1][-1].size
                    for __ in range(size):
                        weights[element][_ + 1][-1][__] = 0.

        return weights


    def calculate_nnEnergy(self, descriptors):
        """
        Calculate the predicted energy with neural network.

        Parameters
        ----------
        descriptor: list
            The list of descriptor per structure.
        """
        p = self.parameters
        self.nnEnergies = []
        Energy = 0.

        for i, (element, des) in enumerate(descriptors):
            scaling = p.scalings[element]
            weights = p.weights[element]
            hl = p.hiddenlayers[element]
            desrange = p.desrange[element]
            activation = p.activation

            nnEnergies = self.forward(hl, des, weights, desrange, activation)
            nnEnergy = scaling['slope'] * float(nnEnergies[len(nnEnergies)-1]) + scaling['intercept']
            
            self.nnEnergies.append(nnEnergy)
            Energy += nnEnergy

        return Energy


    def calculate_nnForce(self, descriptors, output):
        """
        Calculate the predicted force with neural network.
        """
        p = self.parameters
        self.nnForces = []
        Force = 0.

        

        return Force


    def calculate_dnnEnergy_dParameters(self, descriptors):
        """
        I still have no clue what this function does.
        """
        p = self.parameters
        dE_dP = None
        
        for i, (element, des) in enumerate(descriptors):
            scaling = p.scalings[element]
            W = self.weights_wo_bias(p.weights)
            W = W[element]
            dnnEnergy_dParameters = np.zeros(self.ravel.count)
            
            dnnEnergy_dWeights, dnnEnergy_dScalings = self.ravel.to_dicts(dnnEnergy_dParameters)
            outputs = self.forward(p.hiddenlayers[element], des, p.weights[element], p.desrange[element])
            
            D, delta, ohat = self.get_D_delta_ohat(outputs, W, residual=1)
            dnnEnergy_dScalings[element]['intercept'] = 1.
            dnnEnergy_dScalings[element]['slope'] = float(outputs[len(outputs)-1])
            
            for j in range(1, len(outputs)):
                dnnEnergy_dWeights[element][j] = float(scaling['slope']) * np.dot(np.matrix(ohat[j-1]).T, np.matrix(delta[j]).T)                       

            dnnEnergy_dParameters = self.ravel.to_vector(dnnEnergy_dWeights, dnnEnergy_dScalings)

            if dE_dP is None:
                dE_dP = dnnEnergy_dParameters
            else: 
                dE_dP += dnnEnergy_dParameters
        
        return dE_dP
            

    def forward(self, hiddenlayers, descriptors, weight, desrange, activation='tanh'):
        """
        This function is the neural network architecture. The input is given as 
        the descriptor, and the output is calculated for the corresponding energy about a 
        specific atom. The sum of these energies is the total energy of the 
        crystal.

        Parameters
        ----------
        hiddenlayers: tuple
            (3,3) means 2 layers with 3 nodes each.
        descriptor: array
            The descriptor.
        weight: dict
            The Neural Network weights.
        desrange: array
            The range of the descriptor. This is used for scaling.
        activation:
            The activation function.

        Returns
        -------
        dict
            Outputs of neural network nodes.
        """
        layer = 0
        fingerprint = descriptors
        len_fp = len(fingerprint)
        for _ in range(len_fp):
            if (desrange[_][1] - desrange[_][0] > (10.**(-8.))):
                fingerprint[_] = -1.0 + 2.0 * ((fingerprint[_] - desrange[_][0]) /
                        (desrange[_][1] - desrange[_][0]))

        out = {}
        lin = {}

        temp = np.zeros((1, len_fp+1))
        temp[0, len_fp] = 1.0
        for _ in range(len_fp):
            temp[0, _] = fingerprint[_]
        out[0] = temp

        # Neural network architecture
        for i, hl in enumerate(hiddenlayers):
            layer += 1
            lin[i+1] = np.dot(temp, weight[i+1])
            if activation == 'tanh':
                out[i+1] = np.tanh(lin[i+1])
            elif activation == 'sigmoid':
                out[i+1] = 1. / (1. + np.exp(-lin[i+1]))
            elif activation == 'linear':
                out[i+1] = lin[i+1]
            temp = np.zeros((1, hl+1))
            temp[0, hl] = 1.0
            for _ in range(hl):
                temp[0, _] = out[i+1][0][_]

        # The output (i.e. energies)
        lin[layer+1] = np.dot(temp, weight[layer+1])
        if activation == 'tanh':
            out[layer+1] = np.tanh(lin[layer+1])
        elif activation == 'sigmoid':
            out[layer+1] = 1. / (1. + np.exp(-lin[layer+1]))
        elif activation == 'linear':
            out[layer+1] = lin[layer+1]

        return out

    def forwardPrime(hiddenlayers, descriptors, weight, desrange, outputs, activation='tanh',):
        
        layers = len(hiddenlayers)
        fingerprint = copy.deepcopy(descriptors)
        len_fp = len(fingerprint)
        
        for _ in range(len_fp):
            if (desrange[_][1] - desrange[_][0] > (10.**(-8.))):
                fingerprint[_] =  2.0 * (fingerprint[_] /
                        (desrange[_][1] - desrange[_][0]))

        dnodes_dr = {}

        dnodes_dr[0] = np.asarray(fingerprint)

        for layer in range(1, layers+1):
            temp = np.dot(dnodes_dr[layer-1], np.asarray(weight[layer])[:-1,:])
            
            if activation == 'tanh':
                dnodes_dr[layer] = temp * \
                                   (1. - outputs[layer][0] * outputs[layer][0])
            elif activation == 'sigmoid':
                dnodes_dr[layer] = temp * (outputs[layer][0] * (1. - outputs[layer][0]))
            elif activation == 'linear':
                dnodes_dr[layer] = temp

        return dnodes_dr

    def weights_wo_bias(self, weights):
        """
        Return weights without the bias.
        """
        W = {}
        for k in weights.keys():
            W[k] = {}
            w = weights[k]
            for i in range(len(w)):
                W[k][i+1] = w[i+1][:-1]
        return W

    @property
    def vector(self):
        """Access to get or set the model parameters (weights, scaling for
        each network) as a single vector, useful in particular for
        regression.

        Parameters
        ----------
        vector : list
            Parameters of the regression model in the form of a list.
        """
        if self.parameters['weights'] is None:
            return None
        p = self.parameters
        if not hasattr(self, 'ravel'):
            self.ravel = Raveler(p.weights, p.scalings)
        return self.ravel.to_vector(weights=p.weights, scalings=p.scalings)

    @vector.setter
    def vector(self, vector):
        p = self.parameters
        if not hasattr(self, 'ravel'):
            self.ravel = Raveler(p.weights, p.scalings)
        weights, scalings = self.ravel.to_dicts(vector)
        p['weights'] = weights
        p['scalings'] = scalings

    
    def get_D_delta_ohat(self, outputs, W, residual):
        """
        Calculate and store delta, ohat, and D.
        These parameters are needed in calculating the derivative of the output with respect to the weights.

        output: dict
            Outputs of the neural network nodes.
        W: dict
            The weights of the neural network without the bias.
        residual: float
            True energy minus the neural network predicted energy
        """
        p = self.parameters
        activation = p.activation

        N = len(outputs)

        # Delete the bias for outputs[0]
        o = outputs[0].tolist()
        o[0].pop(-1)
        outputs[0] = np.asarray(o)

        D = {}
        for i in range(N):
            n = np.size(outputs[i])
            D[i] = np.zeros((n, n))
            for j in range(n):
                if activation == 'linear':
                    D[i][j,j] = 1.
                elif activation == 'sigmoid':
                    D[i][j,j] = float(outputs[i][0,j]) * \
                            float(1. - outputs[i][0,j])
                elif activation == 'tanh':
                    D[i][j,j] = float(1. - outputs[i][0,j] * outputs[i][0,j])

        delta = {}
        delta[N-1] = D[N-1] # Missing the (o_j - t_j)

        for i in range(N-2, 0, -1):
            delta[i] = np.dot(D[i], np.dot(W[i+1], delta[i+1]))

        ohat = {}
        for i in range(1, N):
            n = np.size(outputs[i-1])
            ohat[i-1] = np.zeros((1,n+1))
            for j in range(n):
                ohat[i-1][0,j] = outputs[i-1][0,j]
            ohat[i-1][0,n] = 1.

        return D, delta, ohat
    

class Raveler:
    """(CP) Class to ravel and unravel variable values into a single vector.

    This is used for feeding into the optimizer. Feed in a list of dictionaries
    to initialize the shape of the transformation. Note no data is saved in the
    class; each time it is used it is passed either the dictionaries or vector.
    The dictionaries for initialization should be two levels deep.

    weights, scalings are the variables to ravel and unravel
    """
    # why would scalings need to be raveled?
    def __init__(self, weights, scalings):

        self.count = 0
        self.weightskeys = []
        self.scalingskeys = []
        for key1 in sorted(weights.keys()):  # element
            for key2 in sorted(weights[key1].keys()):  # layer
                value = weights[key1][key2]
                self.weightskeys.append({'key1': key1,
                                         'key2': key2,
                                         'shape': np.array(value).shape,
                                         'size': np.array(value).size})
                self.count += np.array(weights[key1][key2]).size

        for key1 in sorted(scalings.keys()):  # element
            for key2 in sorted(scalings[key1].keys()):  # slope / intercept
                self.scalingskeys.append({'key1': key1,
                                          'key2': key2})
                self.count += 1

        self.vector = np.zeros(self.count)

    def to_vector(self, weights, scalings):
        """Puts the weights and scalings embedded dictionaries into a single
        vector and returns it. The dictionaries need to have the identical
        structure to those it was initialized with."""
        vector = np.zeros(self.count)
        count = 0
        for k in self.weightskeys:
            lweights = np.array(weights[k['key1']][k['key2']]).ravel()
            vector[count:(count + lweights.size)] = lweights
            count += lweights.size
        for k in self.scalingskeys:
            vector[count] = scalings[k['key1']][k['key2']]
            count += 1

        return vector

    def to_dicts(self, vector):
        """Puts the vector back into weights and scalings dictionaries of the
        form initialized. vector must have same length as the output of
        unravel."""

        assert len(vector) == self.count
        count = 0
        weights = OrderedDict()
        scalings = OrderedDict()

        for k in self.weightskeys:
            if k['key1'] not in weights.keys():
                weights[k['key1']] = OrderedDict()
            matrix = vector[count:count + k['size']]
            matrix = matrix.flatten()
            matrix = np.matrix(matrix.reshape(k['shape']))
            weights[k['key1']][k['key2']] = matrix.tolist()
            count += k['size']
        for k in self.scalingskeys:
            if k['key1'] not in scalings.keys():
                scalings[k['key1']] = OrderedDict()
            scalings[k['key1']][k['key2']] = vector[count]
            count += 1
        return weights, scalings



