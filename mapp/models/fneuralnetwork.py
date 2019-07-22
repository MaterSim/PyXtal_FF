from copy import deepcopy
from collections import OrderedDict
import inspect

import numpy as np
import numba
import json

from .model import Model
from ..utilities.regression import lRegressor as Regressor
from ..utilities.neighborhood import Element


class fNeuralNetwork(Model):
    """Atom-centered neural network model. The inputs are atom-centered 
    descriptors: BehlerParrinello or Bispectrum. The forward function of 
    the Neural Network predicts energy per atom, and the derivative of the 
    forward function predicts the force acting on that particular atom. 
    A machine learning interatomic potential is developed by optimizing the 
    weights of the Neural Network baed on a given system.
    
    """
    def __init__(self, elements, 
                 hiddenlayers=[3, 3],
                 activation='tanh',
                 weights=None,
                 random_seed=1001,
                 force_coefficient=0.03):
        self.elements = elements

        # Format hiddenlayers and add the last output layer.
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

        # Activation options:
        activation_modes = ['tanh', 'sigmoid', 'linear']
        if activation == 'tanh':
            self.activation = 1
        elif activation == 'sigmoid':
            self.activation = 2
        elif activation == 'linear':
            self.activation == 3
        else:
            msg = f"{activation} is not implemented. " +\
                    f"Please choose one of these: {activation_modes}."
            raise NotImplementedError(msg)
            
        self.weights = weights
        self.seed = random_seed
        self.force_coefficient = force_coefficient
        
    def train(self, structures=None, descriptors=None, 
              features=None, save=True):
        """Training the Neural Network model."""
        if structures == None:
            if isinstance(descriptors, dict):
                self.descriptors = descriptors
            elif descriptors == None:
                msg = "Structures and descriptors can't be both None."
                raise TypeError(msg)
            else:
                msg = f"Invalid type of descriptors: {type(descriptors)}." +\
                      "Please refer to documentations!"
                raise NotImplementedError(msg)
        else:
            self.descriptor_model = descriptors
            self.descriptors = self._convert(structures, 
                                             self.descriptor_model)
                    
        # Test the length of features and descriptors
        assert len(self.descriptors) == len(features), \
        "Inbalance length of features to descriptors."
        self.features = features
        
        # Create save to json format
        if isinstance(descriptors, dict) and save:
            msg = "Don't need to save the saved descriptors. "+\
                    "Please input save to False."
            raise TypeError(msg)
        if save:
            with open(self.descriptor_model._type+'.json', 'w') as f:
                json.dump(self.descriptors, f, indent=2)


        
        self.no_of_structures = len(self.descriptors)                     # number of columns
        self.no_of_descriptors = len(self.descriptors["0"]['G'][0][1])    # number of rows
        
        # Calculate the range of the descriptors and scaling the energy.
        self.drange = self.descriptor_range(self.no_of_structures,
                                            self.descriptors)
        
        # Parse features
        energies = []
        self.forces = numba.typed.Dict.empty(
                        key_type=numba.types.int64,
                        value_type=numba.types.float64[:, :])
        for i in range(self.no_of_structures):
            energies.append(self.features[i]['energy'])
            self.forces[i] = np.asarray(self.features[i]['forces'])
        self.energies = np.asarray(energies)
            
        min_E = min(self.energies)           # Perhaps these 2 should be user-
        max_E = max(self.energies)           # defined parameters
        self.scalings = self.scalings(self.activation, [min_E, max_E],
                                      self.drange.keys())
        
        # Generate random weight if None is given.
        self.weights = self.random_weights(self.hiddenlayers, 
                                           self.no_of_descriptors, 
                                           self.seed)
        
        # Parse important information to be read by Numba.
        # 1. hiddenlayers
        self.nb_hiddenlayers = numba.typed.Dict.empty(
                                key_type=numba.types.int64,
                                value_type=numba.types.int64[:])
        
        for i, (k, v) in enumerate(self.hiddenlayers.items()):
            arr = [self.no_of_descriptors] + v
            self.nb_hiddenlayers[i] = np.asarray(arr)

        # 2. descriptors
        self.nb_descriptors = numba.typed.Dict.empty(
                                key_type=numba.types.int64,
                                value_type=numba.types.float64[:, :])
        
        nb_descriptor_elements = []
        for i in range(self.no_of_structures):
            nd = []
            nde = []
            for j in range(len(self.descriptors[str(i)]['G'])):
                nd.append(np.asarray(self.descriptors[str(i)]['G'][j][1]))
                nde.append(self.descriptors[str(i)]['G'][j][0])
            self.nb_descriptors[i] = np.asarray(nd)
            nb_descriptor_elements.append(nde)
            
        self.nb_descriptor_elements = numba.typed.Dict.empty(
                                        key_type=numba.types.int64,
                                        value_type=numba.types.int64[:])
        nb_descriptor_elements = _convert_element(nb_descriptor_elements,
                                                  self.elements)
        for i in range(self.no_of_structures):
            self.nb_descriptor_elements[i] = nb_descriptor_elements[i]
                    
        # 3. drange
        nb_drange = []
        for k, v in self.drange.items():
            nb_drange.append(v)
        self.nb_drange = np.asarray(nb_drange)

        # 4. species
        self.nb_elements = [i for i in range(len(self.elements))]
        
        # 5. descriptors prime
        self.nb_dprime_tuple = numba.typed.Dict.empty(
                                key_type=numba.types.int64,
                                value_type=numba.types.int64[:, :])
        self.nb_dprime = numba.typed.Dict.empty(
                            key_type=numba.types.int64,
                            value_type=numba.types.float64[:, :])
        
        for i in range(self.no_of_structures):
            ndt = []
            ndp = []
            for x, y in descriptors[str(i)]['Gprime']:
                ndt.append(_convert_tuple(x, self.elements))
                ndp.append(y)
            self.nb_dprime_tuple[i] = np.asarray(ndt)
            self.nb_dprime[i] = np.asarray(ndp)
                
        # Run Neural Network
        self.regressor = Regressor()
        self.result = self.regressor.regress(model=self)
        
        
    def predict(self,):
        pass
        
    
    def get_loss(self, parameters, lossprime=True):
        """Get the loss and the derivative of the loss with respect to 
        the parameters. The parameters are weights and scaling factors 
        and to be optimized by Scipy optimizer.
        
        This error function is consistent with:
            Behler, J. Int. J. Quantum Chem. 2015, 115, 1032– 1050.
        
        Parameters
        ----------
        parameters: array
            The adjustable parameters to be optimized.
        lossprime: bool
            If True, calculate the dLossdParameters.
        
        Returns
        -------
        loss: float
            The value of the loss function.
        dLossdParameters:
            The derivative of the lossfunction with respect tot he parameters
        """
        self.vector = parameters
        
        nb_weights = numba.typed.Dict.empty(
                        key_type=numba.types.int64,
                        value_type=numba.types.float64[:, :])
        nb_weights_nb = numba.typed.Dict.empty(
                            key_type=numba.types.int64,
                            value_type=numba.types.float64[:, :])
        
        # 1. Parse weights
        for i, (k1, v1) in enumerate(self.weights.items()):
            for k2, v2 in v1.items():
                k2 = k2*10
                index = k2+i
                nb_weights[index] = np.asarray(v2)
                nb_weights_nb[index] = np.asarray(v2[:-1])
                
                
        # 2. Parse Scalings
        scalings = []
        for element in self.elements:
            scalings.append([self.scalings[element]['intercept'], 
                             self.scalings[element]['slope']])
        nb_scalings = np.asarray(scalings)

        dLossdParameters = np.zeros((len(self.vector),))

        # perhaps parse scalings here
        loss, dLossdParameters = calculate_loss(self.no_of_structures,
                                                self.nb_hiddenlayers,
                                                self.nb_descriptors,
                                                self.nb_descriptor_elements,
                                                self.nb_dprime,
                                                self.nb_dprime_tuple,
                                                self.nb_drange,
                                                nb_weights,
                                                nb_weights_nb,
                                                nb_scalings,
                                                self.activation,
                                                self.energies,
                                                self.forces,
                                                self.ravel.count,
                                                self.force_coefficient,
                                                lossprime)

        print(loss)
        print(dLossdParameters)
        
        return loss, dLossdParameters
    
    
    def descriptor_range(self, no_of_structures, descriptors):
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
            for _ in descriptors[str(i)]['G']:
                element = _[0]
                descriptor = _[1]
                if element not in drange.keys():
                    drange[element] = np.asarray([np.asarray([__, __]) \
                          for __ in descriptor])
                else:
                    assert len(drange[element]) == len(descriptor)
                    for j, des in enumerate(descriptor):
                        if des < drange[element][j][0]:
                            drange[element][j][0] = des
                        elif des > drange[element][j][1]:
                            drange[element][j][1] = des

        return drange
    
    
    def scalings(self, activation, energies, elements):
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
            if activation == 1:
                scalings[element]['intercept'] = (max_E + min_E) / 2.
                scalings[element]['slope'] = (max_E - min_E) / 2.
            elif activation == 2:
                scalings[element]['intercept'] = min_E
                scalings[element]['slope'] = (max_E - min_E)
            elif activation == 3:
                scalings[element]['intercept'] = (max_E + min_E) / 2.
                scalings[element]['slope'] = (10. ** (-10.)) * \
                                                (max_E - min_E) / 2.

        return scalings
    
    
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
        if self.weights is None:
            return None
        if not hasattr(self, 'ravel'):
            self.ravel = Raveler(self.weights, self.scalings)
            
        return self.ravel.to_vector(self.weights, self.scalings)


    @vector.setter
    def vector(self, vector):
        if not hasattr(self, 'ravel'):
            self.ravel = Raveler(self.weights, self.scalings)
        weights, scalings = self.ravel.to_dicts(vector)
        self.weights = weights
        self.scalings = scalings
    

@numba.njit(cache=True, nogil=True, fastmath=True)
def calculate_loss(no_of_structures,
                   hiddenlayers,
                   descriptors,
                   descriptor_elements,
                   dprime,
                   dprime_tuple,
                   drange,
                   weights,
                   weights_nb,
                   scalings,
                   activation,
                   energies,
                   forces,
                   ravel_count,
                   force_coefficient,
                   lossprime):
    
    energyloss = 0.
    forceloss = 0.
    dLossdParameters = np.zeros((ravel_count,))


    for i in range(no_of_structures):
        no_of_atoms = len(descriptors[i])
        
        true_energy = energies[i]
        nnEnergy = calculate_nnEnergy(descriptors[i], 
                                      descriptor_elements[i],
                                      weights,
                                      scalings,
                                      hiddenlayers,
                                      drange,
                                      activation)
        residual =  nnEnergy - true_energy
        energyloss += residual ** 2.
        
        if lossprime:
            dnnEnergydParameters = calculate_dnnEnergydParameters(
                                    descriptors[i],
                                    descriptor_elements[i],
                                    weights,
                                    weights_nb,
                                    scalings,
                                    hiddenlayers,
                                    drange,
                                    activation,
                                    ravel_count)
            dLossdParameters += 2. * residual * dnnEnergydParameters
        
        true_forces = forces[i]
        nnForces = calculate_nnForces(descriptors[i],
                                      dprime[i],
                                      dprime_tuple[i],
                                      weights,
                                      weights_nb,
                                      scalings,
                                      hiddenlayers,
                                      drange,
                                      activation)
        forceresidual = (nnForces - true_forces)
        forceloss += np.sum(forceresidual ** 2.) / 3. / no_of_atoms

        if lossprime:
            dnnForcesdParameters = calculate_dnnForcesdParameters(descriptors[i],
                                                                  dprime[i],
                                                                  dprime_tuple[i],
                                                                  weights,
                                                                  weights_nb,
                                                                  scalings,
                                                                  hiddenlayers,
                                                                  drange,
                                                                  activation,
                                                                  ravel_count)

            temp = np.zeros(ravel_count)
            for _ in range(no_of_atoms):
                for l in range(3):
                    temp += (nnForces[_][l] - true_forces[_][l]) * dnnForcesdParameters[(_, l)]
            dLossdParameters += force_coefficient * 2. * temp / 3. / no_of_atoms

    loss = (energyloss + force_coefficient * forceloss) / no_of_structures
    dLossdParameters /= no_of_structures

    return loss, dLossdParameters


@numba.njit(cache=True, nogil=True, fastmath=True)
def calculate_dnnEnergydParameters(descriptors,
                                   descriptor_elements,
                                   weights,
                                   weights_nb,
                                   scalings,
                                   hiddenlayers,
                                   drange,
                                   activation,
                                   ravel_count):
    
    dE_dP = np.zeros(ravel_count)
    
    for i in range(len(descriptor_elements)):
        element = descriptor_elements[i]
        _scalings = scalings[element]
        _hiddenlayers = hiddenlayers[element]
        _drange = drange[element]
        
        dnnEnergydParameters = np.zeros(ravel_count)
        
        dnnEnergydWeights = dict()
        for k, v in weights.items():
            dnnEnergydWeights[k] = np.zeros(np.shape(v))
            
        nodes = forward(element, _hiddenlayers, descriptors[i],
                        weights, _drange, activation)
        
        D, delta, ohat = backprop(element, nodes, weights_nb, activation)
        
        count = element * (ravel_count - scalings.size)
        for j in range(1, len(nodes)):
            ind = j*10+element
            _ohat = np.reshape(ohat[j-1], (-1, 1))
            _result = _scalings[1] * np.dot(_ohat, delta[j].T)
            _size = _result.size
            dnnEnergydParameters[count:(count+_size)] += _result.ravel()
            count += _size

        dnnEnergydScalings = np.zeros(np.shape(scalings))
        dnnEnergydScalings[element][0] = 1.
        dnnEnergydScalings[element][1] = nodes[len(nodes) - 1][0]
        _scaling_size = dnnEnergydScalings.size
        dnnEnergydParameters[-_scaling_size:] += dnnEnergydScalings.ravel()
    
        dE_dP += dnnEnergydParameters

    return dE_dP


@numba.njit(cache=True, nogil=True, fastmath=True)
def calculate_dnnForcesdParameters(descriptors,
                                   dprime,
                                   dprime_tuple,
                                   weights,
                                   weights_nb,
                                   scalings,
                                   hiddenlayers,
                                   drange,
                                   activation,
                                   ravel_count):
    dF_dP = dict()

    for i in range(len(descriptors)):
        for j in range(3):
            dF_dP[(i, j)] = np.zeros(ravel_count)

    for i in range(len(dprime)):
        index, symbol, nindex, nsymbol, direction = dprime_tuple[i]
        _descriptor = descriptors[nindex]

        _scalings = scalings[nsymbol]
        _hiddenlayers = hiddenlayers[nsymbol]
        _drange = drange[nsymbol]
        
        dnnForcesdParameters = np.zeros(ravel_count)

        dnnForcesdWeights = dict()
        for k, v in weights.items():
            dnnForcesdWeights[k] = np.zeros(np.shape(v))

        nodes = forward(nsymbol, _hiddenlayers, _descriptor, weights, _drange, activation)
        D, delta, ohat = backprop(nsymbol, nodes, weights_nb, activation)
        nodesPrime = forwardPrime(nsymbol, nodes, _hiddenlayers, dprime[i], weights_nb, _drange, activation)
        dnodesPrimedWeights = backpropPrime(nsymbol, nodes, nodesPrime, D, delta, ohat, weights_nb, activation)
        
        N = len(nodes)
        count = nsymbol * (ravel_count - scalings.size)
        for j in range(1, N):
            _size = dnodesPrimedWeights[j].size
            _result = _scalings[1] * dnodesPrimedWeights[j]
            dnnForcesdParameters[count:(count+_size)] = _result.ravel()
            count += _size

        dnnForcesdScalings = np.zeros(np.shape(scalings))
        dnnForcesdScalings[nsymbol][1] = nodesPrime[N-1][0]
        _scaling_size = dnnForcesdScalings.size
        dnnForcesdParameters[-_scaling_size:] += dnnForcesdScalings.ravel()

        dnnForcesdParameters *= -1

        dF_dP[(index, direction)] += dnnForcesdParameters

    return dF_dP

@numba.njit(cache=True, nogil=True, fastmath=True)
def backpropPrime(nsymbol, output, outputPrime, D, delta, ohat, weights_nb, activation):
    N = len(output)

    Dprime = dict()
    for i in range(1, N):
        n = len(output[i])
        Dprime[i] = np.zeros((n, n))

        for j in range(n):
            if activation == 1:
                Dprime[i][j, j] = -2. * output[i][j] * outputPrime[i][j]
            elif activation == 2:
                Dprime[i][j, j] = outputPrime[i][j] - 2. * output[i][j] * outputPrime[i][j]
            elif activation == 3:
                Dprime[i][j, j] = 0.

    deltaPrime = dict()
    deltaPrime[N-1] = Dprime[N-1]

    temp1 = dict()
    temp2 = dict()
    for i in range(N-2, 0, -1):
        index = (i+1)*10+nsymbol
        temp1[i] = np.dot(weights_nb[index], delta[i+1])
        temp2[i] = np.dot(weights_nb[index], deltaPrime[i+1])
        deltaPrime[i] = np.dot(Dprime[i], temp1[i]) + np.dot(D[i], temp2[i])

    ohatPrime = dict()
    doutputPrimedWeights = dict()

    for i in range(1, N):
        n = len(outputPrime[i-1])
        ohatPrime[i-1] = np.zeros((n+1))
        n = len(outputPrime[i-1])
        for j in range(n):
            ohatPrime[i-1][j] = outputPrime[i-1][j]
        ohatPrime[i-1][-1] = 0.
        _ohatPrime = np.reshape(ohatPrime[i-1], (-1, 1))
        _ohat = np.reshape(ohat[i-1], (-1, 1))
        doutputPrimedWeights[i] = np.dot(_ohatPrime,
                                         delta[i].T) +\
                                  np.dot(_ohat,
                                         deltaPrime[i].T)

    return doutputPrimedWeights
       
@numba.njit(cache=True, nogil=True, fastmath=True)
def calculate_nnEnergy(descriptors,
                       descriptor_elements,
                       weights,
                       scalings,
                       hiddenlayers,
                       drange,
                       activation):
    
    energy = 0.

    for i in range(len(descriptor_elements)):
        element = descriptor_elements[i]
        _scalings = scalings[element]
        _hiddenlayers = hiddenlayers[element]
        _drange = drange[element]
            
        nodes = forward(element, _hiddenlayers, descriptors[i],
                        weights, _drange, activation)
        energy += _scalings[1] * nodes[len(nodes)-1][0] + _scalings[0]
    
    return energy

@numba.njit(cache=True, nogil=True, fastmath=True)
def calculate_nnForces(descriptors,
                       dprime,
                       dprime_tuple,
                       weights,
                       weights_nb,
                       scalings,
                       hiddenlayers,
                       drange,
                       activation):
    
    forces = np.zeros((len(descriptors), 3))
    
    for i in range(len(dprime)):
        index, symbol, nindex, nsymbol, direction = dprime_tuple[i]
        _descriptor = descriptors[nindex]
        
        _scalings = scalings[nsymbol]
        _hiddenlayers = hiddenlayers[nsymbol]
        _drange = drange[nsymbol]

        nodes = forward(nsymbol, _hiddenlayers, _descriptor, weights, _drange, activation)
        nodesPrime = forwardPrime(nsymbol, nodes, _hiddenlayers, dprime[i], weights_nb, _drange, activation)
        
        force = -(_scalings[1] * nodesPrime[len(nodesPrime)-1][0])
        forces[index][direction] += force

    return forces

@numba.njit(cache=True, nogil=True, fastmath=True)
def forwardPrime(element, output, hiddenlayers, descriptorPrime, weights_nb, drange, activation):
    no_of_adescriptorPrime = len(descriptorPrime)
    
    layers = len(hiddenlayers)

    _dPrime = np.zeros((no_of_adescriptorPrime)) + descriptorPrime
    
    for _ in range(no_of_adescriptorPrime):
        if ((drange[_][1] - drange[_][0]) > (10 ** (-8.))):
            _dPrime[_] = 2. * (_dPrime[_] / 
                        (drange[_][1] - drange[_][0]))

    outputPrime = dict()

    outputPrime[0] = _dPrime
    
    for layer in range(1, layers):
        index = layer*10 + element    
        term = np.dot(outputPrime[layer-1],weights_nb[index])
        if activation == 1:
            outputPrime[layer] = term * (1. - output[layer] * output[layer])
        elif activation == 2:
            outputPrime[layer] = term * (output[layer] * (1. - output[layer]))
        elif activation == 3:
            outputPrime[layer] = term

    return outputPrime


#@numba.njit(cache=True, nogil=True, fastmath=True)
#def calculate_dnnEnergydParameters(descriptors,
#                                   descriptor_elements,
#                                   weights,
#                                   weights_nb,
#                                   scalings,
#                                   hiddenlayers,
#                                   drange,
#                                   activation,
#                                   ravel_count):
#    
#    dE_dP = np.zeros(ravel_count)
#    
#    for i in range(len(descriptor_elements)):
#        element = descriptor_elements[i]
#        _scalings = scalings[element]
#        _hiddenlayers = hiddenlayers[element]
#        _drange = drange[element]
#        
#        dnnEnergydParameters = np.zeros(ravel_count)
#        
#        dnnEnergydWeights = dict()
#        for k, v in weights.items():
#            dnnEnergydWeights[k] = np.zeros(np.shape(v))
#            
#        nodes = forward(element, _hiddenlayers, descriptors[i],
#                        weights, _drange, activation)
#        
#        D, delta, ohat = backprop(element, nodes, weights_nb, activation)
#        
#        count = element * (ravel_count - scalings.size)
#        for j in range(1, len(nodes)):
#            ind = j*10+element
#            _ohat = np.reshape(ohat[j-1], (-1, 1))
#            _result = _scalings[1] * np.dot(_ohat, delta[j].T)
#            _size = _result.size
#            dnnEnergydParameters[count:(count+_size)] += _result.ravel()
#            count += _size
#
#        dnnEnergydScalings = np.zeros(np.shape(scalings))
#        dnnEnergydScalings[element][0] = 1.
#        dnnEnergydScalings[element][1] = nodes[len(nodes) - 1][0]
#        _scaling_size = dnnEnergydScalings.size
#        dnnEnergydParameters[-_scaling_size:] += dnnEnergydScalings.ravel()
#    
#        dE_dP += dnnEnergydParameters
#
#    return dE_dP
    
@numba.njit(cache=True, nogil=True, fastmath=True)
def backprop(element, output, weights_nb, activation):
    
    N = len(output)
    
    D = dict()
    for i in range(N):
        n = output[i].size
        D[i] = np.zeros((n, n))
        for j in range(n):
            if activation == 1:
                D[i][j,j] = (1. - output[i][j] * output[i][j])
            elif activation == 2:
                D[i][j,j] = output[i][j] * (1. - output[i][j])
            elif activation == 3:
                D[i][j,j] = 1.

    delta = dict()
    delta[N-1] = D[N-1]

    for i in range(N-2, 0, -1):
        ind = (i+1)*10+element
        delta[i] = np.dot(D[i], np.dot(weights_nb[ind], delta[i+1]))
    
    ohat = dict()
    for i in range(1, N):
        n = output[i-1].size
        ohat[i-1] = np.zeros((n+1))
        for j in range(n):
            ohat[i-1][j] = output[i-1][j]
        ohat[i-1][n] = 1.

    return D , delta, ohat



@numba.njit(cache=True, nogil=True, fastmath=True)
def forward(element, hiddenlayers, descriptors, weight, drange, activation):
    
    no_of_adescriptor = len(descriptors)
    
    _descriptors = np.zeros((no_of_adescriptor)) + descriptors
    
    for i in range(no_of_adescriptor):
        diff = drange[i][1] - drange[i][0]
        if (diff > (10.**(-8.))):
            _descriptors[i] = -1. + 2. * (_descriptors[i] - drange[i][0]) /\
                              diff
    
    output = dict()
    output_b = dict()
    
    output[0] = _descriptors
    output_b[0] = np.concatenate((_descriptors, np.ones((1,))))

    for j in range(len(hiddenlayers)-1):
        ind = (j+1)*10+element
        term = np.dot(output_b[j], weight[ind])
        if activation == 1:
            activate = np.tanh(term)
        elif activation == 2:
            activate = 1. / (1. + np.exp(-term))
        elif activation == 3:
            activate = term
            
        output[j+1] = activate
        output_b[j+1] = np.concatenate((activate, np.ones((1,))))
        
    return output




def _convert_element(descriptor_elements, elements):
    array = []
    
    for i in range(len(descriptor_elements)):
        array1 = []
        for j in range(len(descriptor_elements[i])):
            if descriptor_elements[i][j] in elements:
                index = elements.index(descriptor_elements[i][j])
            array1.append(index)
        array.append(array1)
    
    return np.asarray(array)

def _convert_tuple(tup, elements):
    array = np.zeros((len(tup),), dtype=int)
    
    a, b, c, d, e = tup
    
    index1 = elements.index(b)
    index2 = elements.index(d)
    
    array += np.asarray([a, index1, c, index2, e])

    return array

        
################################ AUX function #################################


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
