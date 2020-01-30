import os
import numpy as np
from ase import Atoms
from monty.serialization import loadfn

#@profile
def convert_to_descriptor(structure_file, descriptor_file, function, \
            fmt=None, show_progress=True, ncpu=1):
    """ Obtain training features, structures, and descriptors. """
    if fmt is None:
        if structure_file.find('json')>0:
            fmt = 'json'
        elif structure_file.find('xyz')>0:
            fmt = 'xyz'
        else:
            fmt = 'vasp-out'

    N = function['N']

    # extract the structures and energy/force information
    if fmt == 'json':      
        Structures, Features = parse_json(structure_file, N)
    elif fmt == 'vasp-out':
        Structures, Features = parse_OUTCAR_comp(structure_file, N)
    elif fmt == 'xyz':
        Structures, Features = parse_xyz(structure_file, N)
    else:
        raise NotImplementedError('We support only json and vasp-out format')
    print("{:d} structures have been loaded.".format(len(Structures)))

    # compute the descriptors
    if os.path.exists(descriptor_file):
        descriptors = np.load(descriptor_file, allow_pickle=True) 
        N1 = len(descriptors)
        if N is not None and N < N1:
            Descriptors = descriptors[:N]    
            del descriptors #does not help to free memory at the moment
        else:
            Descriptors = descriptors
        print('Load precomputed descriptors from {:s}, {:d} entries'.format(\
                descriptor_file, len(Descriptors)))
        print("\n")
    else:
        print('Computing the descriptors...')
        if ncpu == 1:
            des = []
            for i in range(len(Structures)):
                des.append(compute_descriptor(function, Structures[i]))
                if show_progress:
                    print('\r{:4d} out of {:4d}'.format(i+1, len(Structures)), flush=True, end='') 
        else:
            from multiprocessing import Pool, cpu_count
            from functools import partial
            with Pool(ncpu) as p:
                func = partial(compute_descriptor, function)
                des = p.map(func, Structures)
                p.close()
                p.join()
        np.save(descriptor_file, des)
        Descriptors = des
        print('\nSaved the descriptors to', descriptor_file)
        print("\n")
    return Features, Descriptors
    
def compute_descriptor(function, structure):
    if function['type'] == 'BehlerParrinello':
        from pyxtal_ff.descriptors.behlerparrinello import BehlerParrinello
        d = BehlerParrinello(function['parameters'], 
                                function['Rc'], 
                                function['derivative']).calculate(structure)
    elif function['type'] == 'Bispectrum':
        from pyxtal_ff.descriptors.bispectrum import SO4_Bispectrum
        d = SO4_Bispectrum(function['parameters']['lmax'],
                              function['Rc'],
                              derivative=function['derivative'],
                           normalize_U=function['parameters']['normalize_U']).calculate(structure)
    else:
        raise NotImplementedError

    return d


def parse_json(structure_file, N=None):
    """
    Extract structures/enegy/force information from json file
    """
    Structures = []
    Features = {}

    structure_dict = loadfn(structure_file)
    if N is None:
        N = len(structure_dict)

    ## Train features
    Features = {}
    Structures = []
    for i, d in enumerate(structure_dict):
        Features[i] = {}
        if 'structure' in d:
            structure = Atoms(symbols=d['structure'].atomic_numbers,
                              positions=d['structure'].cart_coords,
                              cell=d['structure'].lattice._matrix, pbc=True)
            Structures.append(structure)
            Features[i]['energy'] = d['outputs']['energy']
            Features[i]['force'] = d['outputs']['forces']
        else:
            structure = Atoms(symbols=d['elements'], scaled_positions=d['coords'], 
                              cell=d['lattice'], pbc=True)
            Structures.append(structure)
            Features[i]['energy'] = d['energy']
            Features[i]['force'] = d['force']
           
        if i == (N-1):
            break

    return Structures, Features

    
def parse_OUTCAR_comp(structure_file, N=1000000):
    """
    Extract structures/enegy/force information from compressed OUTCAR file
    """
    Structures = []
    Features = {}
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
                        print(symbols)
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
                Structures.append(structure)
                id = len(Structures) - 1
                Features[id] = {}
                Features[id]['energy'] = energy
                Features[id]['force'] = force
                count += line_number - 1
            count += 1
            if count >= len(lines) or len(Structures) == N:
                break
    return Structures, Features

#a short function to create the label for a calculation
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
    Structures = []
    Features = {}
    with open(structure_file, 'r') as f:
        lines = f.readlines()
        count = 0
        while True:
            line = lines[count]
            coords = []
            forces = []
            symbols = []
            number = int(line)
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
            for i in range(number):
                infos = lines[count+2+i].split()
                symbols.append(infos[0])
                coords.append([float(num) for num in infos[5:8]])
                forces.append([float(num) for num in infos[8:]])
                
            Structures.append(Structure(lat, symbols, np.array(coords)))
            id = len(Structures) - 1
            Features[id] = {}
            Features[id]['energy'] = energy
            Features[id]['force'] = np.array(forces)

            count = count + number + 2
            if count >= len(lines) or len(Structures) == N:
                break
    return Structures, Features


