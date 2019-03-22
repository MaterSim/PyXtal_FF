# why do we need this?

def activation_scaling(images, activation, elements):
    """
    To scale the range of activation to the range of actual energies.

    images: list
        ASE atom objects.
    activation: str
        The type of activation function.
    elements: list
        List of atomic symbols in str.

    Returns
    -------
    float
        The scalings.
    """

    n_images = len(images)
    
    # Max and min of true energies.
    max_E = max(image.get_potential_energy(apply_constraint=False) for image in images)
    min_E = min(image.get_potential_energy(apply_constraint=False) for image in images)
    #print(f"This is max and min energy: {max_E}, {min_E}")

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

"""
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

t_images = generate_data(2)
sc = activation_scaling(images=t_images, activation='tanh', elements=['Pt', 'Cu'])

print(sc)
"""
