import os
import gc
import time
import shelve
import numpy as np
from ase import Atoms, units
from copy import deepcopy
from functools import partial
from torch.utils.data import Dataset
from monty.serialization import loadfn
from multiprocessing import Pool, cpu_count
from collections.abc import MutableSequence
from .base_potential import ZBL


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


    def store(self, structure_file, function, storage, ase_db=None):
        """ Map structures to descriptors and store them, including features, to database.
        If compute is False, print pre-computed descriptors message. """
        if function['base_potential']:
            self.base_potential = ZBL(function['base_potential']['inner'],
                                      function['base_potential']['outer'])
        else:
            self.base_potential = None

        if storage:
            if os.path.isdir(structure_file):
                fmt = 'dat'
            elif structure_file.find('json') > 0:
                fmt = 'json'
            elif structure_file.find('xyz') > 0:
                fmt = 'xyz'
            elif structure_file.find('db') > 0:
                fmt = 'db'
            elif structure_file.find('traj') > 0:
                fmt = 'traj'
            else:
                fmt = 'vasp-out'
        
            # extract the structures and energy, forces, and stress information.
            if fmt == 'json':
                data = parse_json(structure_file)
            elif fmt == 'vasp-out':
                data = parse_OUTCAR_comp(structure_file)
            elif fmt == 'xyz':
                data = parse_xyz(structure_file)
            elif fmt == 'db':
                data = parse_ase_db(structure_file)
            elif fmt == 'traj':
                data = parse_traj(structure_file)
            elif fmt == 'dat':
                data = parse_dat(structure_file)
            else:
                raise NotImplementedError('PyXtal_FF supports only json, vasp-out, and xyz formats')
            print("{:d} structures have been loaded.".format(len(data)))
            
            self.add(function, data)

            if ase_db is not None and fmt != 'db':
                convert_to_ase_db(data, ase_db)
                print("save the structures to {:}.".format(ase_db))

        else:
            print(f"Features and precomputed descriptors exist: {self.name}.dat\n")


    def add(self, function, data):
        """ Add descriptors for all structures to database. """
        print('Computing the descriptors...')

        _N = deepcopy(function['N'])
        _cpu = deepcopy(function['ncpu'])
        
        N1 = len(data)
        if _N is not None and _N < N1:
            lists = range(_N)
        else:
            lists = range(N1)
            
        if _cpu == 1:
            for i, index in enumerate(lists):
                d = self.compute(function, data[index])
                self.append(d)
                self.length += 1
                print('\r{:4d} out of {:4d}'.format(i+1, len(lists)), flush=True, end='')

        else:
            # we do it for the reduced lists _data
            _data = [data[item] for item in lists]
            failed_ids = []
            with Pool(_cpu) as p:
                func = partial(self.compute, function)
                for i, d in enumerate(p.imap_unordered(func, _data)):
                    try:
                        self.append(d)
                        self.length += 1
                    except: # track the failed structure ids
                        failed_ids.append(i)
                        #pass
                    print('\r{:4d} out of {:4d}'.format(i+1, len(_data)), flush=True, end='')
                p.close()
                p.join()

            # compute the missing structures in parallel calcs
            for id in failed_ids: 
                d = self.compute(function, _data[id])
                self.append(d)
                self.length += 1

        print(f"\nSaving descriptor-feature data to {self.name}.dat\n")

  
    def compute(self, function, data):
        """ Compute descriptor for one structure to the database. """

        if function['type'] in ['BehlerParrinello', 'ACSF']:
            from pyxtal_ff.descriptors.ACSF import ACSF
            d = ACSF(function['parameters'],
                     function['Rc'], 
                     function['force'],
                     function['stress'], 
                     function['cutoff'], False).calculate(data['structure'])

        elif function['type'] in ['wACSF', 'wacsf']:
            from pyxtal_ff.descriptors.ACSF import ACSF
            d = ACSF(function['parameters'],
                     function['Rc'], 
                     function['force'],
                     function['stress'], 
                     function['cutoff'], True).calculate(data['structure'])
        
        elif function['type'] in ['SO4', 'Bispectrum', 'bispectrum']:
            from pyxtal_ff.descriptors.SO4 import SO4_Bispectrum
            d = SO4_Bispectrum(function['parameters']['lmax'],
                               function['Rc'],
                               derivative=function['force'],
                               stress=function['stress'],
                               normalize_U=function['parameters']['normalize_U'],
                               cutoff_function=function['cutoff']).calculate(data['structure'])
        
        elif function['type'] in ['SO3', 'SOAP', 'soap']:
            from pyxtal_ff.descriptors.SO3 import SO3
            d = SO3(function['parameters']['nmax'],
                    function['parameters']['lmax'],
                    function['Rc'],
                    alpha=function['parameters']['alpha'],
                    derivative=function['force'],
                    stress=function['stress']).calculate(data['structure'])

        elif function['type'] in ['EAD', 'ead']:
            from pyxtal_ff.descriptors.EAD import EAD
            d = EAD(function['parameters'],
                     function['Rc'],
                     function['force'], function['stress'],
                     function['cutoff']).calculate(data['structure'])
        
        elif function['type'] in ['SNAP', 'snap']:
            from pyxtal_ff.descriptors.SNAP import SO4_Bispectrum
            d = SO4_Bispectrum(function['weights'],
                               function['parameters']['lmax'],
                               function['Rc'],
                               derivative=function['force'],
                               stress=function['stress'],
                               normalize_U=function['parameters']['normalize_U'],
                               cutoff_function=function['cutoff'],
                               rfac0=function['parameters']['rfac']).calculate(data['structure'])

        else:
            msg = f"{function['type']} is not implemented"
            raise NotImplementedError(msg)

        if d['rdxdr'] is not None:
            N = d['x'].shape[0]
            L = d['x'].shape[1]
            rdxdr = np.zeros([N, L, 3, 3])
            for _m in range(N):
                ids = np.where(d['seq'][:,0]==_m)[0]
                rdxdr[_m, :, :, :] += np.einsum('ijkl->jkl', d['rdxdr'][ids, :, :, :])
            d['rdxdr'] = rdxdr.reshape([N, L, 9])[:, :, [0, 4, 8, 1, 2, 5]]
            #d['rdxdr'] = np.einsum('ijklm->iklm', d['rdxdr'])\
            #.reshape([shp[0], shp[2], shp[3]*shp[4]])[:, :, [0, 4, 8, 1, 2, 5]]  #need to change
        
        #print(len(data['structure']))
        if self.base_potential:
            base_d = self.base_potential.calculate(data['structure'])
        else:
            base_d = {'energy': 0., 'force': 0., 'stress': 0.}
        
        #print(data['energy'])
        #print(base_d['energy'])

        d['energy'] = np.asarray(data['energy'] - base_d['energy'])
        d['force'] = np.asarray(data['force']) - base_d['force']
        if data['stress'] is not None:
            d['stress'] = np.asarray(data['stress']) - base_d['stress'] / units.GPa
        else:
            d['stress'] = data['stress'] 
        d['group'] = data['group']

        return d


def compute_descriptor(function, structure):
    """ Compute descriptor for one structure. """

    if function['type'] in ['BehlerParrinello', 'ACSF']:
        from pyxtal_ff.descriptors.ACSF import ACSF
        d = ACSF(function['parameters'],
                 function['Rc'], 
                 function['force'],
                 function['stress'],
                 function['cutoff'], False).calculate(structure)

    elif function['type'] in ['wACSF', 'wacsf']:
        from pyxtal_ff.descriptors.ACSF import ACSF
        d = ACSF(function['parameters'],
                 function['Rc'], 
                 function['force'],
                 function['stress'],
                 function['cutoff'], True).calculate(structure)

    elif function['type'] in ['SO4', 'Bispectrum', 'bispectrum']:
        from pyxtal_ff.descriptors.SO4 import SO4_Bispectrum
        d = SO4_Bispectrum(function['parameters']['lmax'],
                           function['Rc'],
                           derivative=True,
                           stress=True,
                           normalize_U=function['parameters']['normalize_U'],
                           cutoff_function=function['cutoff']).calculate(structure)

    elif function['type'] in ['SO3', 'SOAP', 'soap']:
        from pyxtal_ff.descriptors.SO3 import SO3
        d = SO3(function['parameters']['nmax'],
                function['parameters']['lmax'],
                function['Rc'],
                derivative=True,
                stress=True).calculate(structure)

    elif function['type'] in ['EAD', 'ead']:
            from pyxtal_ff.descriptors.EAD import EAD
            d = EAD(function['parameters'],
                     function['Rc'],
                     True, True,
                     function['cutoff']).calculate(structure)

    elif function['type'] in ['SNAP', 'snap']:
        from pyxtal_ff.descriptors.SNAP import SO4_Bispectrum
        d = SO4_Bispectrum(function['weights'],
                           function['parameters']['lmax'],
                           function['Rc'],
                           derivative=True,
                           stress=True,
                           normalize_U=function['parameters']['normalize_U'],
                           cutoff_function=function['cutoff']).calculate(structure)

    else:
        msg = f"{function['type']} is not implemented"
        raise NotImplementedError(msg)
    
    if d['rdxdr'] is not None:
        N = d['x'].shape[0]
        L = d['x'].shape[1]
        rdxdr = np.zeros([N, L, 3, 3])
        for _m in range(N):
            ids = np.where(d['seq'][:,0]==_m)[0]
            rdxdr[_m, :, :, :] += np.einsum('ijkl->jkl', d['rdxdr'][ids, :, :, :])
        d['rdxdr'] = rdxdr.reshape([N, L, 9])[:, :, [0, 4, 8, 1, 2, 5]]
 
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
            try:
                if d['tags'][0] == 'Strain':
                    group = 'Elastic'
                else:
                    group = 'NoElastic'
            except:
                if d['group'] == 'Elastic':
                    group = 'Elastic'
                else:
                    group = 'NoElastic'
            #group = d['group']
            # vasp default output: XX YY ZZ XY YZ ZX
            # pyxtal_ff/lammps use: XX YY ZZ XY XZ YZ
            # Here we assume the sequence is lammps
            if d['group'] == 'Elastic' and 'Mo 3x3x3 cell' in d['description']:
                stress = None
                group = 'NoElastic'
            elif 'virial_stress' in d[key]: #kB to GPa
                s = [s/10 for s in d[key]['virial_stress']] 
                if d['group'] == 'Ni3Mo' or d['element'] == 'Cu': #just tentative fix
                    stress = [s[0], s[1], s[2], s[3], s[4], s[5]]
                else:
                    stress = [s[0], s[1], s[2], s[3], s[5], s[4]]

            elif 'stress' in d[key]: #kB to GPa
                s = [s/10 for s in d[key]['stress']]
                if d['group'] == 'Ni3Mo': #to fix the issue
                    stress = [s[0], s[1], s[2], s[3], s[4], s[5]]
                else:
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
                if info.find('dft_energy') >= 0 or info.find('DFT_energy') >= 0:
                    energy = float(infos[i+1].split()[0])
                elif info.find('config_type') == 0:
                    group = infos[i+1].split()[0]
                elif info.find('Lattice') == 8:
                    lat = []
                    tmp = infos[i+1].split()
                    for num in tmp:
                        num = num.replace('"','')
                        if num.replace('.', '', 1).replace('-', '', 1).isdigit():
                            lat.append(float(num))
                    lat = np.array(lat).reshape([3,3])
                elif info.find('virial') > 0:
                    s = []
                    tmp = infos[i+1].split()
                    for num in tmp:
                        num = num.replace('"','')
                        if num.replace('.', '', 1).replace('-', '', 1).isdigit():
                            s.append(float(num))
                    s = np.array(s).flatten()
                    stress = np.array([s[0], s[4], s[8], s[1], s[2], s[5]])
                elif info.find('Properties') >= 0: 
                    items = infos[i+1].split(':')
                    counts = 0
                    for i in range(int(len(items)/3)):
                        if items[i*3] == 'dft_force' or items[i*3] == 'DFT_force':
                            count_f = counts
                        elif items[i*3] == 'pos':
                            count_p = counts
                        counts += int(items[i*3+2])
                    break

            for i in range(number):
                infos = lines[count+2+i].split()
                symbols.append(infos[0])
                coords[i, :] = np.array([float(num) for num in infos[count_p:count_p+3]])
                forces[i, :] = np.array([float(num) for num in infos[count_f:count_f+3]])

            structure = Atoms(symbols=symbols,
                              positions=coords,
                              cell=lat, pbc=True)
            data.append({'structure': structure,
                         'energy': energy, 'force': forces,
                         'stress': stress, 'group': group})

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
                    # search for "   energy  without entropy="
                    line_number = 0
                    for tmp in range(24):
                        if lines[count+len(symbol_array)+tmp].find("entropy") > 0:
                            line_number = len(symbol_array) + tmp
                            break
                    if line_number == 0:
                        raise ValueError("Cannot identify the line starts with energy without entropy")

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
                
                energy = float(lines[count+line_number-1].split()[-4])
                structure = Atoms(symbols=symbol_array,
                                  positions=coor,
                                  cell=lat, pbc=True)
                data.append({'structure': structure,
                             'energy': energy, 'force': force,
                             'stress': None, 'group': 'random'})
                count += line_number - 1

            count += 1
            if count >= len(lines) or len(data) == N:
                break

    return data

def convert_to_ase_db(data, db_path='test.db'):
    from ase.db import connect
    
    with connect(db_path) as db:
        for d in data:
            struc = d['structure']
            d.pop('structure', None)
            db.write(struc, data=d)

def parse_ase_db(db_path, N=None, Random=False):
    from ase.db import connect

    data = []
    with connect(db_path) as db:
        if N is None:
            N = len(db)

        for i, row in enumerate(db.select()):
            structure = db.get_atoms(row.id)
            if "dft_stress" in row.data:
                stress = row.data["dft_stress"]
            elif "stress" in row.data:
                stress = row.data["stress"]
            else:
                stress = None

            if "group" in row.data:
                group = row.data["group"]
            else:
                group = None

            if "dft_energy" in row.data:
                eng = row.data["dft_energy"]
            else:
                eng = row.data["energy"]

            if "dft_force" in row.data:
                force = row.data["dft_force"]
            else:
                force = row.data["force"]
            data.append({'structure': structure,
                         'energy': eng,
                         'force': force,
                         'group': group,
                         'stress': stress})

            if i == (N-1):
                break

    return data

def parse_traj(structure_file):
    from ase.build import sort
    from ase.io.trajectory import Trajectory
    data = []
    out_traj = Trajectory(structure_file)

    for traj in out_traj:
        structure = sort(traj)
        energy = traj.get_potential_energy()
        force = traj.get_forces()
        try:
            # PyXtal_FF: XX  YY  ZZ  XY  XZ  YZ
            # ASE      : xx  yy  zz  yz  xz  xy
            # eV/A^3 to GPa
            stress = -(traj.get_stress()/units.GPa)[[0, 1, 2, 5, 4, 3]]
        except:
            stress = None
        xjson = {'structure':structure,
                 'energy':energy,
                 'force':force,
                 'stress':stress,
                 'group':'random'}
        data.append(xjson)

    return data

def parse_dat(structure_files):
    import glob
    data = []

    files = glob.glob(structure_files+'*.dat')
    for file in files:
        file_str = file.split('/')
        lines = open(file, 'r').readlines()
        if file_str[-1] == 'struct_info_1.dat':
            group = 'random'
        elif file_str[-1] in ['struct_info_2.dat', 'struct_info_3.dat']:
            group = 'no_stress'
        elif file_str[-1] == 'struct_info_4.dat':
            group = 'with_stress'
        elif file_str[-1] == 'struct_info_5.dat':
            group = 'with_stress'

        for i, line in enumerate(lines):
            content = line.split()

            if len(content) == 0:
                if group == 'no_stress':
                    stress = [0.]*6
                    structure = Atoms(symbols=atoms, positions=positions, cell=cell, pbc=True)
                    
                    xdata = {'structure': structure,
                             'energy': energy,
                             'force': forces,
                             'stress': stress,
                             'group': group}
                    data.append(xdata)
                    mode = None
                continue

            if content[0] == 'STRUCTURE':
                cell, atoms, positions, forces = [], [], [], []
                continue

            elif content[0] == 'CELL':
                mode = 'stress' if content[1] == 'STRESS' else 'cell'
                continue
                
            elif content[0] == 'ATOMIC':
                mode = 'position' if content[1] == 'NAME' else 'force'
                continue
                
            elif content[0] == 'TOTAL':
                mode = 'energy'
                continue

            if mode == 'energy':
                energy = float(line)

            elif mode == 'cell':
                cel = list(map(float, content))
                cell.append(cel)

            elif mode == 'position':
                pos = list(map(float, content[1:]))
                atoms.append(content[0])
                positions.append(pos)

            elif mode == 'force':
                force = list(map(float, content[1:]))
                forces.append(force)

            elif mode == 'stress':
                # Negative?
                # Convert to kbar?
                stress = list(map(float, content))#[[0, 3, 5, 1, 2, 4]]
                #stress = [stress[0], stress[3], stress[5], stress[1], stress[2], stress[4]]
                stress = [stress[0]*0.1, stress[3]*0.1, stress[5]*0.1, stress[1]*0.1, stress[2]*0.1, stress[4]*0.1]

                structure = Atoms(symbols=atoms, positions=positions, cell=cell, pbc=True)
                
                xdata = {'structure': structure,
                         'energy': energy,
                         'force': forces,
                         'stress': stress,
                         'group': group}
                data.append(xdata)
                mode = None

    return data
