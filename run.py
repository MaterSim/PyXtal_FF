import sys

import numpy as np
np.set_printoptions(threshold=sys.maxsize)

from pymatgen.io.ase import AseAtomsAdaptor

from ase.calculators.emt import EMT
from ase.build import fcc110
from ase import Atoms, Atom
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units
from ase.md import VelocityVerlet
from ase.constraints import FixAtoms

from mapp.descriptors.behlerparrinello import BehlerParrinello
from mapp.models.new_neuralnetwork import nNeuralNetwork as NN


from amp import Amp
from amp.descriptor.gaussian import Gaussian, make_symmetry_functions
from amp.model.neuralnetwork import NeuralNetwork
from amp.model import LossFunction



def generate_data(count):
    """Generates test or training data with a simple MD simulation."""
    atoms = fcc110('Pt', (2, 2, 2), vacuum=7.)
    adsorbate = Atoms([Atom('Cu', atoms[7].position + (0., 0., 2.5)),
                       Atom('Cu', atoms[7].position + (0., 0., 5.))])
    atoms.extend(adsorbate)
    atoms.set_constraint(FixAtoms(indices=[0, 2]))
    atoms.set_calculator(EMT())
    MaxwellBoltzmannDistribution(atoms, 300. * units.kB)
    dyn = VelocityVerlet(atoms, dt=1. * units.fs)
    newatoms = atoms.copy()
    newatoms.set_calculator(EMT())
    newatoms.get_potential_energy()
    images = [newatoms]
    for step in range(count - 1):
        dyn.run(50)
        newatoms = atoms.copy()
        newatoms.set_calculator(EMT())
        newatoms.get_potential_energy()
        images.append(newatoms)
    return images

images = generate_data(3)

structures = []
for image in images:
        structures.append(AseAtomsAdaptor.get_structure(image))

symmetry = {'G2': {'eta': np.logspace(np.log10(0.05),
                                      np.log10(5.), num=4),
                   'Rc': 6.5}}


descriptors = {}
features = {}
for i in range(len(structures)):
    bp = BehlerParrinello(structures[i], symmetry, derivative=True)
    descriptors[i] = bp.Gs
    features[i] = {}
    features[i]['energy'] = (images[i].get_potential_energy(apply_constraint=False)/len(images[i]))
    features[i]['force'] = image.get_forces(apply_constraint=False)

mode = NN(elements=['Pt', 'Cu'], activation='tanh')
mode.fit(descriptors, features)
