import os
import gc
import time
import shelve
import numpy as np
from ase import Atoms
from copy import deepcopy
from random import sample
from functools import partial
from torch.utils.data import Dataset
from monty.serialization import loadfn
from multiprocessing import Pool, cpu_count
from collections.abc import MutableSequence


class Database():#MutableSequence):
    def __init__(self, name):
        self.name = name
        self.database = shelve.open(self.name)
        
        self.msg = "Index must be an integer in the interval [0,len(self)]"
        self.length = len(list(self.database.keys()))


    def __len__(self):
        return self.length


    def __setitem__(self, index, value):
        if isinstance(index, int) and index >= 0:
            self.database[str(index)] = value
        else:
            raise IndexError(self.msg)


    def __getitem__(self, index):
        if isinstance(index, int) and index >= 0 and index < len(self):
            return self.database[str(index)]
        else:
            raise IndexError(self.msg)


    def __delitem__(self, index):
        if isinstance(index, int) and index >= 0:
            del(self.database[str(index)])
        else:
            raise IndexError(self.msg)


    def insert(self, index, value):
        """ Insert the value to the dictionary at index. """
        if isinstance(index, int) and index >= 0:
            pass
        else:
            raise IndexError(self.msg)
        self[index] = value


    def append(self, value):
        """ Append value to the end of the sequence. """
        self.insert(len(self), value)


    def close(self):
        self.database.close()


    def store(self, structure_file, function, storage):
        """ Map structures to descriptors and store them, including features, to database.
        If compute is False, print pre-computed descriptors message. """
        if storage:
            if os.path.isdir(structure_file) or structure_file.find('json') > 0:
                fmt = 'json'
            elif structure_file.find('xyz') > 0:
                fmt = 'xyz'
            else:
                fmt = 'vasp-out'
        
            # extract the structures and energy, forces, and stress information.
            if fmt == 'json':
                data = parse_json(structure_file)
            elif fmt == 'vasp-out':
                data = parse_OUTCAR_comp(structure_file)
            elif fmt == 'xyz':
                data = parse_xyz(structure_file)
            else:
                raise NotImplementedError('PyXtal_FF supports only json, vasp-out, and xyz formats')
            print("{:d} structures have been loaded.".format(len(data)))

            self.add(function, data)

        else:
            print(f"Features and precomputed descriptors exist: {self.name}.dat\n")


    def add(self, function, data):
        """ Add descriptors for all structures to database. """
        print('Computing the descriptors...')

        _N = deepcopy(function['N'])
        _cpu = deepcopy(function['ncpu'])
        _random = deepcopy(function['random_sample'])
        
        if _cpu == 1:
            N1 = len(data)
            if _N is not None and _N < N1:
                if _random:
                    lists = sample(range(N1), _N)
                else:
                    lists = range(_N)
            else:
                lists = range(N1)
            
            for i, index in enumerate(lists):
                d = self.compute(function, data[index])
                self.append(d)
                self.length += 1
                print('\r{:4d} out of {:4d}'.format(i+1, len(lists)), flush=True, end='')

        else:
            with Pool(_cpu) as p:
                func = partial(self.compute, function)
                for i, d in enumerate(p.imap_unordered(func, data)):
                    self.append(d)
                    self.length += 1
                    print('\r{:4d} out of {:4d}'.format(i+1, len(data)), flush=True, end='')
                p.close()
                p.join()
            
        print(f"\nSaving descriptor-feature data to {self.name}.dat\n")

  
    def compute(self, function, data):
        """ Compute descriptor for one structure to the database. """

        if function['type'] == 'BehlerParrinello':
            from pyxtal_ff.descriptors.behlerparrinello import BehlerParrinello
            d = BehlerParrinello(function['parameters'],
                                 function['Rc'], 
                                 True, True).calculate(data['structure'])
        
        elif function['type'] == 'Bispectrum':
            from pyxtal_ff.descriptors.bispectrum import SO4_Bispectrum
            d = SO4_Bispectrum(function['parameters']['lmax'],
                               function['Rc'],
                               derivative=True,
                               stress=True,
                               normalize_U=function['parameters']['normalize_U']).calculate(data['structure'])
        
        elif function['type'] == 'SOAP':
            from pyxtal_ff.descriptors.SOAP import SOAP
            d = SOAP(function['parameters']['nmax'],
                     function['parameters']['lmax'],
                     function['Rc'],
                     derivative=True,
                     stress=True).calculate(data['structure'])
        else:
            msg = f"{function['type']} is not implemented"
            raise NotImplementedError(msg)
        
        shp = d['rdxdr'].shape
        d['rdxdr'] = np.einsum('ijklm->iklm', d['rdxdr'])\
            .reshape([shp[0], shp[2], shp[3]*shp[4]])[:, :, [0, 4, 8, 1, 2, 5]]

        d['energy'] = np.asarray(data['energy'])
        d['force'] = np.asarray(data['force'])
        if data['stress'] is not None:
            d['stress'] = np.asarray(data['stress'])
        else:
            d['stress'] = data['stress']
        d['group'] = data['group']

        return d


def compute_descriptor(function, structure):
    """ Compute descriptor for one structure. """

    if function['type'] == 'BehlerParrinello':
        from pyxtal_ff.descriptors.behlerparrinello import BehlerParrinello
        d = BehlerParrinello(function['parameters'],
                             function['Rc'], 
                             True, True).calculate(structure)
    elif function['type'] == 'Bispectrum':
        from pyxtal_ff.descriptors.bispectrum import SO4_Bispectrum
        d = SO4_Bispectrum(function['parameters']['lmax'],
                           function['Rc'],
                           derivative=True,
                           stress=True,
                           normalize_U=function['parameters']['normalize_U']).calculate(structure)
    elif function['type'] == 'SOAP':
        from pyxtal_ff.descriptors.SOAP import SOAP
        d = SOAP(function['parameters']['nmax'],
                 function['parameters']['lmax'],
                 function['Rc'],
                 derivative=True,
                 stress=True).calculate(structure)
    else:
        msg = f"{function['type']} is not implemented"
        raise NotImplementedError(msg)
    
    if d['rdxdr'] is not None:
        shp = d['rdxdr'].shape
        d['rdxdr'] = np.einsum('ijklm->iklm', d['rdxdr'])\
            .reshape([shp[0], shp[2], shp[3]*shp[4]])[:, :, [0,4,8,1,2,5]]

    return d


def parse_json(path, N=None, Random=False):
    """ Extract structures/energy/forces/stress information from json file. """
    if os.path.isfile(path):
        structure_dict = loadfn(path)
    elif os.path.isdir(path):
        import glob
        cwd = os.getcwd()
        os.chdir(path)
        files = glob.glob('*.json')
        os.chdir(cwd)
        structure_dict = []
        for file in files:
            fp = os.path.join(path, file)
            structure_dict += loadfn(fp)

    if N is None:
        N = len(structure_dict)
    elif Random and N < len(structure_dict):
        structure_dict = sample(structure_dict, N)

    data = []
    for i, d in enumerate(structure_dict):
        if 'structure' in d:
            structure = Atoms(symbols=d['structure'].atomic_numbers,
                              positions=d['structure'].cart_coords,
                              cell=d['structure'].lattice._matrix, pbc=True)
            v = structure.get_volume()
            if 'data' in d:
                key = 'data'
            else:
                key = 'outputs'
            
            if 'energy_per_atom' in d[key]:
                energy = d[key]['energy_per_atom']*len(structure)
            else:
                energy = d[key]['energy']
            force = d[key]['forces']
            group = d['group']
            if 'virial_stress' in d[key]:
                s = [-1*s*v/1602.1766208 for s in d[key]['virial_stress']]
                stress = [s[0], s[1], s[2], s[3], s[5], s[4]]
            elif 'stress' in d[key]:
                s = [-1*s*v/1602.1766208 for s in d[key]['stress']]
                stress = [s[0], s[1], s[2], s[3], s[5], s[4]]
            else:
                stress = None
            
            data.append({'structure': structure,
                         'energy': energy, 'force': force, 
                         'stress': stress, 'group': group})
        
        else:   # For PyXtal
            structure = Atoms(symbols=d['elements'], scaled_positions=d['coords'], 
                              cell=d['lattice'], pbc=True)
            data.append({'structure': structure,
                         'energy': d['energy'], 'force': d['force'], 
                         'stress': None, 'group': 'random'})
           
        if i == (N-1):
            break

    return data


def create_label(elements, hiddenlayers):
    label = ''
    for e in elements:
        label += e
        label += '-'
    for l in hiddenlayers:
        label += str(l)
        label += '-'
    label += '1'
    return label


def get_descriptors_parameters(symmetry, system):
    from itertools import combinations_with_replacement
    G = []
    if 'G2' in symmetry:
        combo = list(combinations_with_replacement(system, 1))
        if 'Rs' not in symmetry['G2']:
            Rs = [0.]
        else:
            Rs = symmetry['G2']['Rs']
        
        for eta in symmetry['G2']['eta']:
            for rs in Rs:
                for element in combo:
                    g = [element[0], 'nan', rs, eta, 'nan', 'nan']
                    G.append(g)

    if 'G4' in symmetry:
        combo = list(combinations_with_replacement(system, 2))
        for zeta in symmetry['G4']['zeta']:
            for lamBda in symmetry['G4']['lambda']:
                for eta in symmetry['G4']['eta']:
                    for p_ele in combo:
                        g = [p_ele[0], p_ele[1], 'nan', eta, lamBda, zeta]
                        G.append(g)

    return G


def parse_xyz(structure_file, N=1000000):
    """
    Extract structures/enegy/force information of xyz file provided by the Cambridge group
    """
    data = []
    with open(structure_file, 'r') as f:
        lines = f.readlines()
        count = 0
        while True:
            line = lines[count]
            symbols = []
            number = int(line)
            coords = np.zeros([number, 3])
            forces = np.zeros([number, 3])
            infos = lines[count+1].split('=')
            for i, info in enumerate(infos):
                if info.find('energy') == 0:
                    energy = float(infos[i+1].split()[0])
                elif info.find('Lattice') == 8:
                    lat = []
                    tmp = infos[i+1].split()
                    for num in tmp:
                        num = num.replace('"','')
                        if num.replace('.', '', 1).replace('-', '', 1).isdigit():
                            lat.append(float(num))
                    lat = np.array(lat).reshape([3,3])
                    break
                elif info.find('virial') > 0:
                    s = []
                    tmp = infos[i+1].split()
                    for num in tmp:
                        num = num.replace('"','')
                        if num.replace('.', '', 1).replace('-', '', 1).isdigit():
                            s.append(float(num))
                    s = np.array(s).flatten()
                    stress = np.array([s[0], s[4], s[7], s[1], s[2], s[5]])

            for i in range(number):
                infos = lines[count+2+i].split()
                symbols.append(infos[0])
                coords[i, :] = np.array([float(num) for num in infos[5:8]])
                forces[i, :] = np.array([float(num) for num in infos[-3:]])

            structure = Atoms(symbols=symbols,
                              positions=coords,
                              cell=lat, pbc=True)
            data.append({'structure': structure,
                         'energy': energy, 'force': forces,
                         'stress': stress, 'group': 'random'})

            count = count + number + 2
            if count >= len(lines) or len(data) == N:
                break

    return data


def parse_OUTCAR_comp(structure_file, N=1000000):
    """
    Extract structures/enegy/force information from compressed OUTCAR file
    """
    data = []
    with open(structure_file, 'r') as f:
        lines = f.readlines()
        read_symbol = True
        read_composition = False
        read_structure = False
        symbols = []
        count = 0
        while True:
            line = lines[count]
            if read_symbol:
                if line.find('POTCAR') > 0:
                    symbol = line.split()[2]
                    if symbol in symbols:
                        read_symbol = False
                        read_composition = True
                    else:
                        symbols.append(symbol)
            elif read_composition:
                if line.find('ions per type') > 0:
                    symbol_array = []
                    tmp = line.split('=')[1].split()
                    numIons = []
                    for i in range(len(symbols)):
                        for j in range(int(tmp[i])):
                            symbol_array.append(symbols[i])
                    read_composition = False
                    read_structure = True
                    count += 4
                    line_number = len(symbol_array) + 16
            else: #read_structure
                lat_string = lines[count+1:count+4]
                lat = np.zeros([3,3])
                for i, l in enumerate(lines[count+1:count+4]):
                    lat[i,:] = np.fromstring(l, dtype=float, sep=' ')[:3]
                coor = np.zeros([len(symbol_array), 3])
                force = np.zeros([len(symbol_array), 3])
                for i, l in enumerate(lines[count+6:count+6+len(symbol_array)]):
                    array = np.fromstring(l, dtype=float, sep=' ')
                    coor[i,:] = array[:3]
                    force[i,:] = array[3:]
                energy = float(lines[count+line_number-3].split()[-2])
                structure = Atoms(symbols=symbol_array,
                                  positions=coor,
                                  cell=lat, pbc=True)
                Features.append({'structure': structure,
                                 'energy': energy, 'force': force,
                                 'stress': None, 'group': 'random'})
                count += line_number - 1
            count += 1
            if count >= len(lines) or len(data) == N:
                break
    return data
