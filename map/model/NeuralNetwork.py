import os
from collections import OrderedDict


import numpy as np
from ase.calculators.calculator import Parameters

from model import calculate_descriptor_range

class Neuralnetwork:
    """
    This class implements a feed-forward neural network.

    Parameters
    ----------
    hiddenlayers: tuple
        (3,3) means 2 layers with 3 nodes each.
    activation: str
        The activation function for the neural network.
    feature_mode: bool
        The user input feature. If False, ASE will provide calculator to calculate the energies and forces.
    """
    def __init__(self, hiddenlayers=(3,3), activation='tanh', elements=['Pt', 'Cu'], feature_mode=False):
        p = self.parameters = Parameters()
        
        act_mode = ['tanh', 'sigmoid', 'linear']
        if activation not in act_mode:
            msg = f"{activation} is not implemented here. Please choose from {act_mode}."
            raise NotImplementedError(msg)

        p.hiddenlayers = hiddenlayers
        p.activation = activation
        p.elements = elements
        p.feature_mode = feature_mode
        p.weights = None


    def fit(self, images, descriptor, feature=None):
        """
        Fit the model parameters here.

        images: List
            List of ASE atomic objects for neural network training.
        descriptor: array
            The descriptor that corresponds to the images.
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
        p.descriptor = descriptor
        p.descriptor_shape = (len(descriptor[0]), len(descriptor[0][0][1]))

        p.desrange = calculate_descriptor_range(images, descriptor)
        p.scalings = self.activation_scaling(images, p.activation, p.desrange.keys())
        p.weights = self.get_random_weights(p.hiddenlayers, p.descriptor_shape,)

        self.vec = self.vector

        nnEnergy = self.calculate_nnEnergy(descriptor[0])
        nndEnergy = self.calculate_dnnEnergy_dParameters(descriptor[0])


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


    def calculate_nnEnergy(self, descriptor):
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

        for i, (element, des) in enumerate(descriptor):
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

    def calculate_dnnEnergy_dParameters(self, descriptor):
        """
        I still have no clue what this function does.
        """
        p = self.parameters
        dE_dP = 0.
        
        for i, (element, des) in enumerate(descriptor):
            W = self.weights_wo_bias(p.weights)
            W = W[element]
            dnnEnergy_dParameters = np.zeros(self.ravel.count)
        
            dnnEnergy_dWeights, dnnEnergy_dScalings = self.ravel.to_dicts(dnnEnergy_dParameters)
            outputs = self.forward(p.hiddenlayers[element], des, p.weights[element], p.desrange[element])
            
            print(outputs)

    def forward(self, hiddenlayers, descriptor, weight, desrange, activation='tanh'):
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
        fingerprint = descriptor
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
#        print(f"This is before ravel: {weights}")
        for k in self.weightskeys:
            lweights = np.array(weights[k['key1']][k['key2']]).ravel()
            vector[count:(count + lweights.size)] = lweights
            count += lweights.size
        for k in self.scalingskeys:
            vector[count] = scalings[k['key1']][k['key2']]
            count += 1
#        print(f"This is after ravel: {vector}")
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

##########################################################################################

from ase.calculators.emt import EMT
from ase.build import fcc110
from ase import Atoms, Atom
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units
from ase.md import VelocityVerlet
from ase.constraints import FixAtoms
def generate_data(count):
    atoms = fcc110('Pt', (2,2,2), vacuum=7.)
    adsorbate = Atoms([Atom('Cu', atoms[7].position + (0., 0., 2.5)),
        Atom('Cu', atoms[7].position + (0., 0., 5.))])
    atoms.extend(adsorbate)
    atoms.set_constraint(FixAtoms(indices=[0, 2]))
    atoms.set_calculator(EMT())
    MaxwellBoltzmannDistribution(atoms, 300.*units.kB)
    dyn = VelocityVerlet(atoms, dt=1.*units.fs)
    newatoms = atoms.copy()
    newatoms.set_calculator(EMT())
    newatoms.get_potential_energy()
    images = [newatoms]
    for step in range(count-1):
        dyn.run(50)
        newatoms = atoms.copy()
        newatoms.set_calculator(EMT())
        newatoms.get_potential_energy()
        images.append(newatoms)
    return images

from amp import Amp
from amp.descriptor.gaussian import Gaussian, make_symmetry_functions
from amp.model.neuralnetwork import NeuralNetwork
from amp.model import LossFunction
from amp.utilities import hash_images

label = 'train_test_g5/calc'
train_images = generate_data(2)
elements = ['Pt', 'Cu']
G = make_symmetry_functions(elements=elements, type='G2',
        etas=np.logspace(np.log10(0.05), np.log10(5.),
            num=4))
G = {element: G for element in elements}
calc = Amp(descriptor=Gaussian(Gs=G),
        model=NeuralNetwork(hiddenlayers=(3, 3)),
        label=label,
        cores=1)
loss = LossFunction(convergence={'energy_rmse': 0.02,
    'force_rmse': 0.03})
calc.model.lossfunction = loss

calc.train(images=train_images,)

# Generating the gaussian descriptors
h_images = hash_images(train_images)
d = []
for key in h_images.keys():
    d.append(calc.model.trainingparameters.descriptor.fingerprints[key])


nn = Neuralnetwork()
nn.fit(images=train_images, descriptor=d)
