from pprint import pprint
import json
from pymatgen import Structure
from snap import bispectrum

with open("AIMD.json") as f:
    data = json.load(f)


structures = []
energies = []
lattice = data[0]['structure']['lattice']['matrix']

for struc in data[0:1]:
    lat = struc['structure']['lattice']['matrix']
    species = []
    position = []
    for site in struc['structure']['sites']:
        species.append(site['label'])
        position.append(site['xyz'])
    
    structures.append(Structure(lat, species, position))
    energies.append(struc['outputs']['energy'])

profile = dict(Cu=dict(r=0.5, w=1.0))

from pymatgen import Lattice, Structure

s = Structure.from_spacegroup(225, Lattice.cubic(5.69169),
                                      ['Na', 'Cl'],
                                      [[0, 0, 0], [0, 0, 0.5]])
profile = dict(Na=dict(r=0.5, w=0.9), Cl=dict(r=0.5, w=3.0))

L = bispectrum(s, 6, 2, profile, diagonal=3)

#b = bispectrum(structures[0], 5.0, 3, profile, diagonal=3)
