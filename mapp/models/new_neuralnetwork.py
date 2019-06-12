import os
from copy import deepcopy
from collections import OrderedDict

import numpy as np

from ase.calculators.calculator import Parameters

from ..utilities.lregression import Regressor

class NeuralNetwork:
    """Atom-centered neural network model. The inputs of this model have
    to be atom-centered descriptors to predict the total energy and forces
    of a crystal structure. By optimizing the weights of neural network based
    on a given system, a machine learning interatomic potential is developed.

    Parameters
    ----------
    elements: list
        A list of atomic species in the crystal system.
    input_type: str
        The chosen descriptor for the inputs to the neural network.
        Options: BehlerParrinello and Bispectrum.
    hiddenlayers: list or dict
        [3, 3] contains 2 layers with 3 nodes each. hiddenlayers can also be 
        passed as a dict. Each atomic species in the crystal system is 
        assigned with a neural network architecture. 
        For example, {'Na': [3, 3], 'Cl': [2, 2]} has two neural network 
        architectures for NaCl system.
    activation: str
        The activation function for the neural network model.
        Options: tanh, sigmoid, and linear.
    weights: OrderedDict
        Predefined user input weights. Must be in the form of OrderedDict.
        For example, the weights for a [3, 3] neural network with 2 input
        values:
            w = OrderedDict([(1, [[1, 2, 3],
                                  [1, 2, 3],
                                  [1, 2, 3]]),
                             (2, [[1, 2 ,3], 
                                  [1, 2, 3],
                                  [1, 2, 3],
                                  [1, 2, 3]]),
                             (3, [[1], [2], [3], [4]])])
        Every last array of in the dict represents the bias, and layer 3
        represents the 1 output node.
    seed: int
        Random seed for generating random initial random weights.
    force_coeff: float
        This parameter is used in the penalty function to scale the force
        contribution relative to the energy.
    """
    def __init__(self, elements, input_type='BehlerParrinello', 
                 hiddenlayers=[3, 3], activation='tanh', weights=None, 
                 seed=13, force_coefficient=0.03):
        p = self.parameters = Parameters()

        p.elements = elements
        
        # Format hiddenlayers and add the last output layer
        if isinstance(hiddenlayers, list):
            hl = {}
            for elem in p.elements:
                hl[elem] = hiddenlayers + [1]
            p.hiddenlayers = hl
        elif isinstance(hiddenlayers, dict):
            for key, value in hiddenlayers.items():
                hiddenlayers[key] = value + [1]
            p.hiddenlayers = hiddenlayers
        else:
            msg = f"Don't recognize {type(hiddenlayers)}. " +\
                  f"Please refer to documentations!"
            raise TypeError(msg)
        
        # Activation options
        activation_modes = ['tanh', 'sigmoid', 'linear']
        if activation not in activation_modes:
            msg = f"{activation} is not implemented. " +\
                    f"Please choose from {activation_modes}."
            raise NotImplementedError(msg)
        p.activation = activation
        
        # weights
        p.weights = weights
        p.seed = seed

        p.force_coefficient = force_coefficient


    def fit(self, descriptors, features, images=None,):
        """Fitting of the neural network model.
        
        Parameters
        ----------
        descriptors: dict of dicts
            The atom-centered descriptors. The descriptors of 2 crystal 
            structures should be presented in this form:
                {1: {'G': [('Na', [1, 2, 3, 4]), ('Cl', [3, 2, 1, 0]), ...],
                     'Gprime': [ ... ]},
                 2: {'G': [('Na', [2, 2, 2, 2]), ('Cl', [3, 3, 3, 3]), ...],
                     'Gprime': [ ... ]}}
        features: dict of dicts
            energies and forces of crystal structures. The features of 2 
            crystal structures should be presented in this form:
                {1: {'energy': -10.0, 'forces': [ ... ]},
                 2: {'energy': -20.0, 'forces': [ ... ]}}
        images: dict
            ASE atomic objects. Provide the original images of ASE structures 
            if the features are not defined.
        """
        p = self.parameters
        
        p.descriptors = descriptors
        p.features = features
        p.training_images = images
        
        p.no_of_structures = len(descriptors)                   # number of columns
        p.no_of_adescriptor = len(descriptors[0]['G'][0][1])    # number of rows

        # Calculate the range of the descriptors and scaling the energy.
        p.drange = NeuralNetwork.descriptor_range(p.no_of_structures, p.descriptors)
        energies = [features[i]['energy'] for i in range(p.no_of_structures)]
        min_E = min(energies)
        max_E = max(energies)
        p.scalings = NeuralNetwork.scalings(p.activation, [min_E, max_E], p.drange.keys())

        # Generate random weight if None is given.
        if p.weights == None:
            p.weights = self.random_weights(p.hiddenlayers, p.no_of_adescriptor, p.seed)
        
        self.regressor = Regressor()
        self.result = self.regressor.regress(model=self)


    def calculate_loss(self, parameters, lossprime=True):
        """Get loss and its derivative for Scipy optimization."""
        self.vector = parameters
        p = self.parameters
        descriptors = p.descriptors

        energyloss = 0.
        forceloss = 0.
        dLossdParameters = np.zeros((len(self.vector),)) # The derivative of energy loss w.r.t. the parameters.

        for i in range(p.no_of_structures):
            no_of_atoms = len(descriptors[i]['G'])

            true_energy = p.features[i]['energy']
            nnEnergy = self.calculate_nnEnergy(descriptors[i])
            energyloss += (nnEnergy - true_energy) ** 2.

            if lossprime:
                dnnEnergydParameters = self.calculate_dnnEnergydParameters(descriptors[i])
                dLossdParameters += 2. * (nnEnergy - true_energy) * dnnEnergydParameters * p.energy_coefficient
            
            true_forces = p.features[i]['force']
            nnForces = self.calculate_nnForces(descriptors[i])
            forceresidual = (nnForces - true_forces)
            
            forceloss += np.sum(forceresidual ** 2.)
            
            if lossprime:
                dnnForcesdParameters = self.calculate_dnnForcesdParameters(descriptors[i])
                temp = 0.
                for i in range(no_of_atoms):
                    for l in range(3):
                        temp += (nnForces[i][l] - true_forces[i][l]) * dnnForcesdParameters[(i, l)]
                dLossdParameters += p.force_coefficient * 2. * temp
                dforceloss = 2. * temp

        loss = p.energy_coefficient * energyloss + p.force_coefficient * forceloss
        
        return loss, dLossdParameters


    def calculate_nnEnergy(self, descriptor):
        """Predicting energy with neural network.

        Parameters
        ----------
        descriptor: dict
            The atom-centered descriptor of a crystal structure.
        
        Returns
        -------
        energy: float
            The predicted energy.
        """
        p = self.parameters
        
        nodes = []
        energy = 0.

        for i, (element, des) in enumerate(descriptor['G']):
            scalings = p.scalings[element]
            weight = p.weights[element]
            hl = p.hiddenlayers[element]
            drange = p.drange[element]
            activation = p.activation

            nodes = self.forward(hl, des, weight, drange, activation)
            energy += scalings['slope'] * nodes[len(nodes)-1][0] + scalings['intercept']

        return energy


    def calculate_nnForces(self, descriptor):
        """Calculate the predicted forces based on the derivative of 
        neural network and the derivative of the atom-centered descriptor.
        
        Parameters
        ----------
        descriptor: dict
            The atom-centered descriptor of a crystal structure.

        Returns
        -------
        forces: array
            The predicted forces.
        """
        p = self.parameters

        _descriptor = descriptor['G']
        _descriptorPrime = descriptor['Gprime']

        forces = np.zeros((len(_descriptor),3))
        
        for tup, desp in _descriptorPrime:
            index, symbol, nindex, nsymbol, direction = tup
            des = _descriptor[nindex][1]
            
            scaling = p.scalings[nsymbol]
            hiddenlayers = p.hiddenlayers[nsymbol]
            weight = p.weights[nsymbol]
            w = self.weights_wo_bias(p.weights)[nsymbol]
            drange = p.drange[nsymbol]
            activation = p.activation
            
            output = self.forward(hiddenlayers, des, weight, drange, activation)
            outputPrime = self.forwardPrime(output, hiddenlayers, desp, w, drange, activation)
            
            force = -(scaling['slope'] * outputPrime[len(outputPrime)-1][0])
            forces[index][direction] += force        # I think this should be nindex instead of index
        
        return forces


    def calculate_dnnEnergydParameters(self, descriptor):
        """Calculate the derivative of the energy with respect to 
        the parameters (i.e. weights and scalings).

        Parameters
        ----------
        descriptor: list
             The atom-centered descriptor of a crystal structure.
        
        Returns
        -------
        list
            The derivative of energy with respect to the parameters.
        """
        p = self.parameters
        
        dE_dP = None

        w = self.weights_wo_bias(p.weights)

        for i, (element, des) in enumerate(descriptor['G']):
            scalings = p.scalings[element]
            weight = p.weights[element]
            hl = p.hiddenlayers[element]
            drange = p.drange[element]
            activation = p.activation

            _w = w[element]

            dnnEnergydParameters = np.zeros(self.ravel.count)       # might not need this!
            dnnEnergydWeights, dnnEnergydScalings = self.ravel.to_dicts(dnnEnergydParameters)
            output = self.forward(hl, des, weight, drange, activation)

            D, delta, ohat = self.backprop(output, _w, residual=1.)
            
            dnnEnergydScalings[element]['intercept'] = 1.
            dnnEnergydScalings[element]['slope'] = output[len(output)-1]

            for j in range(1, len(output)):
                dnnEnergydWeights[element][j] = scalings['slope']* np.dot(np.matrix(ohat[j-1]).T, np.matrix(delta[j]).T) # rewrite this?

            dnnEnergydParameters = self.ravel.to_vector(dnnEnergydWeights, dnnEnergydScalings)

            if dE_dP is None:
                dE_dP = dnnEnergydParameters
            else:
                dE_dP += dnnEnergydParameters

        return dE_dP


    def calculate_dnnForcesdParameters(self, descriptor):
        """Calculate the derivative of the force with respect to 
        the parameters.
        
        Parameters
        ----------
        descriptor: list
             The atom-centered descriptor of a crystal structure.

        Returns
        -------
        dict
            The derivative of the forces w.r.t. the parameters
        """
        p = self.parameters

        _descriptor = descriptor['G']
        _descriptorPrime = descriptor['Gprime']
        
        dF_dP = {(i, j): None 
                 for i in range(len(_descriptor))
                 for j in range(3)}

        for tup, desp in _descriptorPrime:
            index, symbol, nindex, nsymbol, direction = tup
            des = _descriptor[nindex][1]

            scaling = p.scalings[nsymbol]
            hiddenlayers = p.hiddenlayers[nsymbol]
            weight = p.weights[nsymbol]
            w = self.weights_wo_bias(p.weights)[nsymbol]
            drange = p.drange[nsymbol]
            activation = p.activation

            dnnForcesdParameters = np.zeros(self.ravel.count)
            dnnForcesdWeights, dnnForcesdScalings = self.ravel.to_dicts(dnnForcesdParameters)
            
            output = self.forward(hiddenlayers, des, weight, drange, activation)
            D, delta, ohat = self.backprop(output, w, residual=1.)
            outputPrime = self.forwardPrime(output, hiddenlayers, desp, w, drange, activation)
            
            N = len(output)

            # maybe put this in backpropPrime
            Dprime = {}
            for i in range(1, N):
                n = len(output[i])
                Dprime[i] = np.zeros((n, n))

                for j in range(n):
                    if activation == 'tanh':
                        Dprime[i][j, j] = -2. * output[i][j] * outputPrime[i][j]
                    elif activation == 'linear':
                        Dprime[i][j, j] = 0.
                    elif activation == 'sigmoid':
                        Dprime[i][j, j] = outputPrime[i][j] - 2. * output[i][j] * outputPrime[i][j]

            deltaPrime = {}
            deltaPrime[N-1] = Dprime[N-1] 

            temp1 = {}
            temp2 = {}
            for i in range(N-2, 0, -1):
                temp1[i] = np.dot(w[i+1], delta[i+1])
                temp2[i] = np.dot(w[i+1], deltaPrime[i+1])
                deltaPrime[i] = np.dot(Dprime[i], temp1[i]) + np.dot(D[i], temp2[i])

            ohatPrime = {}
            doutputPrimedWeights = {}
            for i in range(1, N):
                ohatPrime[i-1] = [None] * (1 + len(outputPrime[i-1]))
                n = len(outputPrime[i-1])
                for j in range(n):
                    ohatPrime[i-1][j] = outputPrime[i-1][j]
                ohatPrime[i-1][-1] = 0.
                doutputPrimedWeights[i] = np.dot(np.matrix(ohatPrime[i-1]).T,
                                                 np.matrix(delta[i]).T) + \
                                          np.dot(np.matrix(ohat[i-1]).T,
                                                 np.matrix(deltaPrime[i]).T)

            for i in range(1, N):
                dnnForcesdWeights[nsymbol][i] = scaling['slope'] * doutputPrimedWeights[i]
            dnnForcesdScalings[nsymbol]['slope'] = outputPrime[N-1][0]

            dnnForcesdParameters = self.ravel.to_vector(dnnForcesdWeights, dnnForcesdScalings)

            dnnForcesdParameters *= -1.

            if dF_dP[(index, direction)] is None:
                dF_dP[(index, direction)] = dnnForcesdParameters
            else:
                dF_dP[(index, direction)] += dnnForcesdParameters

        return dF_dP


    def forward(self, hiddenlayers, descriptor, weight, drange, activation):
        """The feedforward neural network function.
        
        Parameters
        ----------
        hiddenlayers: list
            The hiddenlayers nodes of the neural network.
        descriptor: list
            The atom-centered descriptor of the corresponding element.
        weight: dict
            The neural network weights.
        drange: array
            The range values of the descriptor.
        activation:
            The activation function for the neural network model.

        Returns
        -------
        dict
            The output of the neural network nodes.
        """
        no_of_adescriptor = len(descriptor)
        _hiddenlayers = [no_of_adescriptor] + deepcopy(hiddenlayers)
        _descriptor = deepcopy(descriptor)

        for i in range(no_of_adescriptor):
            diff = drange[i][1] - drange[i][0]
            if (diff > (10.**(-8.))):
                _descriptor[i] = -1. + 2. * (_descriptor[i] - drange[i][0]) /\
                                 diff
        
        output = {}
        output_b = {}
        
        output[0] = np.asarray(_descriptor)
        output_b[0] = np.asarray(_descriptor + [1.])

        # Feedforward neural network
        for i in range(len(_hiddenlayers)-1):
            term = np.dot(output_b[i], weight[i+1])
            if activation == 'tanh':
                activate = np.tanh(term)
            elif activation == 'sigmoid':
                activate = 1. / (1. + np.exp(-term))
            elif activation == 'linear':
                activate = term
                
            output[i+1] = activate
            output_b[i+1] = np.append(activate, [1.])

        return output


    def forwardPrime(self, output, hiddenlayers, descriptorPrime, weight, drange, activation,):
        """The derivative of the feedforward w.r.t. the atom-centered descriptor.

        Parameters
        ----------
        output: dict
            The output of the neural network nodes.
        hiddenlayers: list
            The hiddenlayers nodes of the neural network.
        descriptorPrime: list
            The derivative of the atom-centered descriptor of 
            the corresponding element.
        weight: dict
            The neural network weights without the bias.
        drange: array
            The range values of the descriptor.
        activation:
            The activation function for the neural network model.

        Returns
        -------
        dict
            The derivative of the output of the neural network nodes.
        """
        no_of_adescriptorPrime = len(descriptorPrime)
        layers = len(hiddenlayers)
        _dPrime = deepcopy(descriptorPrime)

        for _ in range(no_of_adescriptorPrime):
            if (drange[_][1] - drange[_][0] > (10.**(-8.))):
                _dPrime[_] =  2.0 * (_dPrime[_] / 
                            (drange[_][1] - drange[_][0]))

        outputPrime = {}

        outputPrime[0] = np.asarray(_dPrime)

        for layer in range(1, layers+1):
            term = np.dot(outputPrime[layer-1], np.asarray(weight[layer]))

            if activation == 'tanh':
                outputPrime[layer] = term * (1. - output[layer] * output[layer])
            elif activation == 'sigmoid':
                outputPrime[layer] = term * (output[layer] * (1. - output[layer]))
            elif activation == 'linear':
                outputPrime[layer] = temp

        return outputPrime


    def backprop(self, output, w, residual):
        """The backpropagation method to get the derivative of the output
        with respect to the adjustable parameters.

        Parameters
        ----------
        output: dict
            The output of the feedforward neural network.
        w: dict
            The neural network weight without the bias.
        residual: float
            The predicted energy minus the true energy.

        Returns
        -------
        documentation here
        """
        p = self.parameters
        activation = p.activation

        N = len(output)

        D = {}
        for i in range(N):
            n = np.size(output[i])
            D[i] = np.zeros((n,n))
            for j in range(n):
                if activation == 'linear':
                    D[i][j,j] = 1.
                elif activation == 'sigmoid':
                    D[i][j,j] = output[i][j] * (1. - outpui][j])
                elif activation == 'tanh':
                    D[i][j,j] = (1. - output[i][j] * output[i][j])

        delta = {}
        delta[N-1] = D[N-1] # missing the residual term (o_j - t_j)

        for i in range(N-2, 0, -1):
            delta[i] = np.dot(D[i], np.dot(w[i+1], delta[i+1]))
        
        # I don't think I need ohat
        ohat = {}
        for i in range(1, N):
            n = np.size(output[i-1])
            ohat[i-1] = np.zeros((n+1))
            for j in range(n):
                ohat[i-1][j] = output[i-1][j]
            ohat[i-1][n] = 1.

        return D, delta, ohat


    def random_weights(self, hiddenlayers, no_of_adescriptor, seed=None):
        """Generating random initial weights for the neural network.

        Parameters
        ----------
        hiddenlayers: list
            The hiddenlayers nodes of the neural network.
        no_of_adescriptor: int
            The length of a descriptor.
        seed: int
            The seed for Numpy random generator.

        Returns
        -------
        dict
            Randomly-generated weights.
        """
        weights = {}
        nnArchitecture = {}

        r = np.random.RandomState(seed=seed)

        elements = hiddenlayers.keys()
        for element in sorted(elements):
            weights[element] = {}
            nnArchitecture[element] = [no_of_adescriptor] + hiddenlayers[element]
            n = len(nnArchitecture[element])

            for layer in range(n-1):
                epsilon = np.sqrt(6. / (nnArchitecture[element][layer] +
                                        nnArchitecture[element][layer+1]))
                norm_epsilon = 2. * epsilon
                weights[element][layer+1] = r.random_sample(
                        (nnArchitecture[element][layer]+1,
                         nnArchitecture[element][layer+1])) * \
                                 norm_epsilon - epsilon
        
        return weights


    def weights_wo_bias(self, weights):
        """Return the weights without bias."""
        w = {}
        for key in weights.keys():
            w[key] = {}
            for i in range(len(weights[key])):
                    w[key][i+1] = weights[key][i+1][:-1]
        return w


    @staticmethod
    def descriptor_range(no_of_structures, descriptors):
        """Calculate the range (min and max values) of the descriptors 
        corresponding to all of the crystal structures.
        
        Parameters
        ----------
        no_of_structures: int
            The number of structures.
        descriptors: dict of dicts
            Atom-centered descriptors.
            
        Returns
        -------
        dict
            The range of the descriptors for each element species.
        """
        drange = {}

        for i in range(no_of_structures):
            for _ in descriptors[i]['G']:
                element = _[0]
                descriptor = _[1]
                if element not in drange.keys():
                    drange[element] = [[__, __] for __ in descriptor]
                else:
                    assert len(drange[element]) == len(descriptor)
                    for j, des in enumerate(descriptor):
                        if des < drange[element][j][0]:
                            drange[element][j][0] = des
                        elif des > drange[element][j][1]:
                            drange[element][j][1] = des

        return drange


    @staticmethod
    def scalings(activation, energies, elements):
        """To scale the range of activation to the range of actual energies.

        Parameters
        ----------
        activation: str
            The activation function for the neural network model.
        energies: list
            The min and max value of the energies.

        Returns
        -------
        dict
            The scalings parameters, i.e. slope and intercept.
        """
        # Max and min of true energies.
        min_E = energies[0] 
        max_E = energies[1] 

        scalings = {}

        for element in elements:
            scalings[element] = {}
            if activation == 'tanh':
                scalings[element]['intercept'] = (max_E + min_E) / 2.
                scalings[element]['slope'] = (max_E - min_E) / 2.
            elif activation == 'sigmoid':
                scalings[element]['intercept'] = min_E
                scalings[element]['slope'] = (max_E - min_E)
            elif activation == 'linear':
                scalings[element]['intercept'] = (max_E + min_E) / 2.
                scalings[element]['slope'] = (10. ** (-10.)) * (max_E - min_E) / 2.

        return scalings


    @property
    def vector(self):
        """(CP) Access to get or set the model parameters (weights, 
        scaling for each network) as a single vector, useful in particular 
        for regression.

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


########################### AUX function ######################################


class Raveler:
    """(CP) Class to ravel and unravel variable values into a single vector.

    This is used for feeding into the optimizer. Feed in a list of dictionaries
    to initialize the shape of the transformation. Note no data is saved in the
    class; each time it is used it is passed either the dictionaries or vector.
    The dictionaries for initialization should be two levels deep.

    weights, scalings are the variables to ravel and unravel
    """
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
