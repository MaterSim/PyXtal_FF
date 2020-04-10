import os
import gc
import numpy as np
from ase import Atoms
from copy import deepcopy
from random import sample
from functools import partial
from monty.serialization import loadfn
from multiprocessing import Pool, cpu_count


def convert_to_descriptor(structure_file, descriptor_file, function,
                          fmt=None, show_progress=True):
    """ Obtain training features, structures, and descriptors. """
    if fmt is None:
        if os.path.isdir(structure_file) or structure_file.find('json') > 0:
            fmt = 'json'
        elif structure_file.find('xyz') > 0:
            fmt = 'xyz'
        else:
            fmt = 'vasp-out'
    
    # extract the structures and energy, forces, and stress information
    if fmt == 'json':
        Structures, Features = parse_json(structure_file)
    elif fmt == 'vasp-out':
        Structures, Features = parse_OUTCAR_comp(structure_file)
    elif fmt == 'xyz':
        Structures, Features = parse_xyz(structure_file)
    else:
        raise NotImplementedError('PyXtal_FF supports only json and vasp-out formats')
    print("{:d} structures have been loaded.".format(len(Structures)))
    
    if os.path.exists(descriptor_file+'x.npy'):
        _N = deepcopy(function['N'])
        _force = deepcopy(function['force'])
        _stress = deepcopy(function['stress'])
        _random = deepcopy(function['random_sample'])

        if _stress:
            if not _force:
                msg = "If stress descriptors are to be calculated, so are force descriptors."
                raise NotImplementedError(msg)
        
        descriptors = np.load(descriptor_file+'x.npy', allow_pickle=True)
        print(f"x has been loaded from {descriptor_file+'x.npy'}")
        
        if _force:
            try:
                dxdr = list(np.load(descriptor_file+'dxdr.npy', allow_pickle=True))
                print(f"dxdr has been loaded from {descriptor_file+'dxdr.npy'}")
                                
                if _stress:
                    try:
                        rdxdr = list(np.load(descriptor_file+'rdxdr.npy', allow_pickle=True))
                        print(f"rdxdr has been loaded from {descriptor_file+'rdxdr.npy'}")
                        print("\n")
                        for i in range(len(descriptors)):
                            descriptors[i]['dxdr'] = dxdr.pop(0)['dxdr']
                            descriptors[i]['rdxdr'] = rdxdr.pop(0)['rdxdr']
                        Features, Descriptors = get_descriptors_and_features(descriptors, Features, _N, _random)

                    except:
                        del(dxdr)
                        gc.collect()
                        print(f"rdxdr didn't load, {descriptor_file+'rdxdr.npy'} doesn't exit")
                        print("The descriptors must be recalculated.\n")
                        Features, Descriptors = compute_descriptors(function, Structures, Features, descriptor_file)
                    
                else:
                    print(f"rdxdr didn't load")
                    print("\n")
                    for i in range(len(descriptors)):
                        descriptors[i]['dxdr'] = dxdr.pop(0)['dxdr']
                    Features, Descriptors = get_descriptors_and_features(descriptors, Features, _N, _random)
            
            except:
                print(f"dxdr didn't load, {descriptor_file+'dxdr.npy'} doesn't exist")
                print("The descriptors must be recalculated.")
                print("\n")
                Features, Descriptors = compute_descriptors(function, Structures, Features, descriptor_file)
        
    else:
        Features, Descriptors = compute_descriptors(function, Structures, Features, descriptor_file)
            
    return Features, Descriptors


def get_descriptors_and_features(descriptors, features, N, rand):
    """ This function is to randomized and/or select a particular
    chunk of the data set. """
    N1 = len(descriptors)
    if N is not None and N < N1:
        if rand:
            lists = sample(range(N1), N)
            descriptors = descriptors[lists]
            features = [features[i] for i in lists]
        else:
            descriptors = descriptors[:N]
            features = features[:N]
        print("{:d} structures have been extracted.".format(N))
    if len(features) != len(descriptors):
        raise ValueError('Number of features/descriptors are inconsistent {:d}/{:d}'.format(len(features), len(descriptors)))

    return features, descriptors


def compute_descriptors(function, structures, features, des_file, show_progress=True):
    """ Computing descriptors for all structures. """
    print('Computing the descriptors...')
    
    _N = deepcopy(function['N'])
    _cpu = deepcopy(function['ncpu'])
    _force = deepcopy(function['force'])
    _stress = deepcopy(function['stress'])
    _compress = deepcopy(function['compress'])
    _random = deepcopy(function['random_sample'])
    
    X, DXDR, RDXDR = [], [], []
    if _cpu == 1:
        
        N1 = len(structures)
        if _N is not None and _N < N1:
            if _random:
                lists = sample(range(N1), _N)
                features = [features[i] for i in lists]
            else:
                lists = range(_N)
                features = features[:_N]
        else:
            lists = range(N1)

        for i, index in enumerate(lists):
            d = compute_descriptor(function, structures[index])
            
            X.append({'x': d['x'], 'elements': d['elements'], 'compressed': d['compressed']})
            DXDR.append({'dxdr': d['dxdr']})
            RDXDR.append({'rdxdr': d['rdxdr']})

            if show_progress:
                print('\r{:4d} out of {:4d}'.format(i+1, len(lists)), flush=True, end='')

    else:
        with Pool(_cpu) as p:
            func = partial(compute_descriptor, function)
            des = p.map(func, structures)
            p.close()
            p.join()

        for d in des:
            X.append({'x': d['x'], 'elements': d['elements'], 'compressed': d['compressed']})
            DXDR.append({'dxdr': d['dxdr']})
            RDXDR.append({'rdxdr': d['rdxdr']})
    np.save(des_file+'x.npy', X)
    print('\nSaving x to', des_file+'x.npy')
    if _force:
        np.save(des_file+'dxdr.npy', DXDR)
        print('Saving dxdr to', des_file+'dxdr.npy')
    if _stress:
        np.save(des_file+'rdxdr.npy', RDXDR)
        print('Saving rdxdr to', des_file+'rdxdr.npy')
    print("\n")

    for i in range(len(X)):
        if _force:
            X[i]['dxdr'] = DXDR.pop(0)['dxdr']
        if _stress:
            X[i]['rdxdr'] = RDXDR.pop(0)['rdxdr']

    descriptors = X

    return features, descriptors


def compute_descriptor(function, structure):
    """ Compute descriptor for one structure. """

    if function['type'] == 'BehlerParrinello':
        from pyxtal_ff.descriptors.behlerparrinello import BehlerParrinello
        d = BehlerParrinello(function['parameters'],
                             function['Rc'], 
                             function['force'],
                             function['stress']).calculate(structure)
    elif function['type'] == 'Bispectrum':
        from pyxtal_ff.descriptors.bispectrum import SO4_Bispectrum
        d = SO4_Bispectrum(function['parameters']['lmax'],
                           function['Rc'],
                           derivative=function['force'],
                           stress=function['stress'],
                           normalize_U=function['parameters']['normalize_U']).calculate(structure)
    elif function['type'] == 'SOAP':
        from pyxtal_ff.descriptors.SOAP import SOAP
        d = SOAP(function['parameters']['nmax'],
                 function['parameters']['lmax'],
                 function['Rc'],
                 derivative=function['force'],
                 stress=function['stress']).calculate(structure)
    else:
        msg = f"{function['type']} is not implemented"
        raise NotImplementedError(msg)
    
    d['compressed'] = False
    
    if d['rdxdr'] is not None:
        shp = d['rdxdr'].shape
        d['rdxdr'] = np.einsum('ijklm->iklm', d['rdxdr'])\
            .reshape([shp[0], shp[2], shp[3]*shp[4]])[:, :, [0,4,8,1,2,5]]

    if function['compress']:
        d = compress_descriptors(d, function["system"])

    return d


def compress_descriptors(descriptor, system):
    """ Compress the x and dxdr of a structure. 
    Initially:
        x.shape = [n, d]
        dxdr.shape = [n, m, d, 3]
        rdxdr.shape = [n, d, 6]
        n = the number of center atoms
        m = the number of atoms in the unit cell
        d = the number of descriptors
        3 = x, y, z directions
    
    Returns
    -------
        X.shape = [e, d]
        DXDR.shape = [e, m, d, 3]
        rdxdr.shape = [e, d, 6]
        e = the number of elemental species.    
    """
    system = sorted(system)
    
    if descriptor['dxdr'] is None:
        msg = "Can't convert dxdr to 3D if derivative is not calculated."
        raise ValueError(msg)

    elements = descriptor['elements']
    x = descriptor['x']
    dxdr = descriptor['dxdr']
    if descriptor['rdxdr'] is not None:
        rdxdr = descriptor['rdxdr']

    X = np.zeros([len(system), len(x[0])])
    DXDR = np.zeros([len(system), len(x), len(x[0]), 3])
    if descriptor['rdxdr'] is not None:
        RDXDR = np.zeros([len(system), len(x[0]), 6])
    else:
        RDXDR = descriptor['rdxdr']
    
    x_temp, dxdr_temp, rdxdr_temp, count = {}, {}, {}, {}
    for ele in system:
        x_temp[ele] = None
        dxdr_temp[ele] = None
        rdxdr_temp[ele] = None
        count[ele] = 0
    
    # Loop over the number of atoms in the structure.
    for e, element in enumerate(elements):
        if x_temp[element] is None:
            x_temp[element] = x[e]
            dxdr_temp[element] = dxdr[e]
            if descriptor['rdxdr'] is not None:
                rdxdr_temp[element] = rdxdr[e] # [d, 6]
        else:
            x_temp[element] += x[e]
            dxdr_temp[element] += dxdr[e]
            if descriptor['rdxdr'] is not None:
                rdxdr_temp[element] += rdxdr[e] # [d, 6]
        count[element] += 1

    for e, element in enumerate(system):
        if count[element] > 0:
            x_temp[element] /= count[element]
            X[e, :] = x_temp[element]
            DXDR[e, :, :, :] = dxdr_temp[element]
            if descriptor['rdxdr'] is not None:
                RDXDR[e, :, :] = rdxdr_temp[element]#.reshape([len(x[0]), 9])

    d = {'x': X, 'dxdr': DXDR, 'rdxdr': RDXDR,
         'elements': elements, 'compressed': True}

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

    ## Train features
    Features = []
    Structures = []
    for i, d in enumerate(structure_dict):
        if 'structure' in d:
            structure = Atoms(symbols=d['structure'].atomic_numbers,
                              positions=d['structure'].cart_coords,
                              cell=d['structure'].lattice._matrix, pbc=True)
            Structures.append(structure)
            v = structure.get_volume()
            try:
                energy = d['data']['energy_per_atom']*len(structure)
                force = d['data']['forces']
                s = [-1*s*v/1602.1766208 for s in d['data']['virial_stress']]
                stress = [s[0], s[1], s[2], s[5], s[4], s[3]]
                group = d['group']
            except:
                energy = d['outputs']['energy']
                force = d['outputs']['forces']
                s = [-1*s*v/1602.1766208 for s in d['outputs']['virial_stress']]
                stress = [s[0], s[1], s[2], s[3], s[5], s[4]]
                group = d['group']

            Features.append({'energy': energy, 'force': force, 'stress': stress, 
                             'group': group})
        
        else:   # For PyXtal
            structure = Atoms(symbols=d['elements'], scaled_positions=d['coords'], 
                              cell=d['lattice'], pbc=True)
            Structures.append(structure)
            Features.append({'energy': d['energy'], 'force': d['force'], 
                             'stress': None, 'group': 'random'})
           
        if i == (N-1):
            break

    return Structures, Features


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
    Features = []
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
            Structures.append(structure)
            Features.append({'energy': energy, 'force': forces, 
                'stress': stress, 'group': 'random'})

            count = count + number + 2
            if count >= len(lines) or len(Structures) == N:
                break
    return Structures, Features


def parse_OUTCAR_comp(structure_file, N=1000000):
    """
    Extract structures/enegy/force information from compressed OUTCAR file
    """
    Structures = []
    Features = []
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
                Structures.append(structure)
                Features.append({'energy': energy, 'force': force, 
                    'stress': None, 'group': 'random'})
                count += line_number - 1
            count += 1
            if count >= len(lines) or len(Structures) == N:
                break
    return Structures, Features
