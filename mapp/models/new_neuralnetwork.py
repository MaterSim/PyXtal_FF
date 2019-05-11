import os
import copy
from collections import OrderedDict

import numpy as np

from ase.calculators.calculator import Parameters

from .model import calculate_descriptor_range
from ..utilities.lregression import Regressor

class NeuralNetwork:
    """This neural network model takes inputs from Behler-Parrinello (BP)
    symmetry functions and predict the total energies and forces of crystal
    systems. BP symmetry functions are atom-centered method.

    Parameters
    ----------
    hiddenlayers: list
        [3, 3] means 2 layers with 3 nodes each.
    activation: str
        The activation function for the neural network model.
    feature_mode: bool
        The user input feature.
        If False, ASE will calculate the total energies and forces of the 
        crystal system.

    """
    def __init__(self, hiddenlayers=[3, 3], activation='tanh', 
                 feature_mode=False, 
