# Running Bispectrum Calculations with LAMMPS
# Pymatgen based

import os
import subprocess
import numpy as np
from pymatgen.io.lammps.data import LammpsData


def make_js(twojmax, diagonal):
    js = []
    for j1 in range(0, twojmax + 1):
        if diagonal == 2:
            js.append([j1, j1, j1])
        elif diagonal == 1:
            for j in range(0, min(twojmax, 2 * j1) + 1, 2):
                js.append([j1, j1, j])
        elif diagonal == 0:
            for j2 in range(0, j1 + 1):
                for j in range(j1 - j2, min(twojmax, j1 + j2) + 1, 2):
                    js.append([j1, j2, j])
        elif diagonal == 3:
            for j2 in range(0, j1 + 1):
                for j in range(j1 - j2, min(twojmax, j1 + j2) + 1, 2):
                    if j >= j1:
                        js.append([j1, j2, j])
    return js


# Perhaps, this class can be optimized further more if we don't include stress
# calculations if there is no needed.
class Bispectrum:
    """This class prepares a lammps input file and calls the lammps executable
    to calculate bispectrum coefficients of a given structure.
    
    Parameters
    ----------
    structure: object
        Pymatgen crystal structure object.
    rcutfac: float
        Scale factor applied to all cutoff radii.
    element_profile: dict
        Elemental descriptions of each atom type in the structure.
        i.e. dict(Na=dict(r=0.3, w=0.9), Cl=dict(r=0.7, w=3.0)).
    twojmax: int
        Band limit for bispectrum components.
    diagonal: int
        diagonal value = 0 or 1 or 2 or 3.
        0 = all j1, j2, j <= twojmax, j2 <= j1
        1 = subset satisfying j1 == j2
        2 = subset satisfying j1 == j2 == j3
        3 = subset satisfying j2 <= j1 <= j
    rfac0: float
        Parameter in distance to angle conversion (0 < rcutfac < 1).
        Default value: 0.99363.
    rmin0: float
        Parameter in distance to angle conversion.
        Default value: 0.
    """
    def __init__(self, structure, rcutfac, element_profile, twojmax, 
                 diagonal=3, rfac0=0.99363, rmin0=0.):
        self.exe = 'lmp_serial'          # define your LAMMPS executable here
        self.pre_cmds = ['units metal',
                         'atom_style charge',
                         'box tilt large',
                         'read_data data.0',
                         'pair_style lj/cut 10', 
                         'pair_coeff * * 1 1']
        self.compute_cmds = ['compute sna all sna/atom ',
                             'compute snad all snad/atom ',
                             'compute snav all snav/atom ',
                             'dump 1 all custom 1 dump.element element',
                             'dump 2 all custom 1 dump.sna c_sna[*]',
                             'dump 3 all custom 1 dump.snad c_snad[*]',
                             'dump 4 all custom 1 dump.snav c_snav[*]',
                             'dump_modify 1 element ']
        self.post_cmds = ['run 0']
        self.input_file = 'in.sna'

        self.structure = structure
        self.rcutfac = rcutfac
        self.twojmax = twojmax
        self.elements = element_profile.keys()
        self.ele = []
        self.Rs = []
        self.Ws = []
        for key, value in element_profile.items():
            self.ele.append(key)
            self.Rs.append(element_profile[key]['r'])
            self.Ws.append(element_profile[key]['w'])
        
        self.rfac0 = rfac0
        self.rmin0 = rmin0
        assert diagonal in range(4), \
            'Invalid diagonal style, must be 0, 1, 2, or 3.'
        self.diagonal = diagonal
        
        self.calculate()
        
        os.remove("data.0")
        os.remove("in.sna")
        os.remove("log.lammps")

    
    def calculate(self):
        """Call the LAMMPS executable to compute bispectrum coefficients."""
        data = self.get_dump(self.structure, self.elements)
        data.write_file('data.0')
        self.get_lammps_input(self.input_file)
        p = subprocess.Popen([self.exe, '-in', self.input_file], 
                             stdout=subprocess.PIPE)
        stdout = p.communicate()[0]
        rc = p.returncode
        if rc != 0:
            error_msg = 'LAMMPS exited with return code %d' % rc
            msg = stdout.decode("utf-8").split('\n')[:-1]
            try:
                error = [i for i, m in enumerate(msg)
                        if m.startswith('ERROR')][0]
                error_msg += ', '.join([e for e in msg[error:]])
            except:
                error_msg += msg[-1]
            print(error_msg)
            #raise RuntimeError(error_msg)
        

    def get_dump(self, structure, elements):
        """Convert Pymatgen structure object to LAMMPS dump file."""
        data = LammpsData.from_structure(structure, elements)
 
        return data


    def get_lammps_input(self, input_file):
        """Create LAMMPS input file."""
        sna = f"1 {self.rfac0} {self.twojmax} "
        for R in self.Rs:
            R *= self.rcutfac
            sna += f"{R} "
        for W in self.Ws:
            sna += f"{W} "
        sna += f"diagonal {self.diagonal} rmin0 {self.rmin0} "

        self.compute_cmds[0] += sna + "bzeroflag 0"
        self.compute_cmds[1] += sna + "quadraticflag 0"
        self.compute_cmds[2] += sna + "quadraticflag 0"

        for el in self.ele:
            self.compute_cmds[-1] += f"{el} "

        self.CMDS = self.pre_cmds + self.compute_cmds + self.post_cmds

        with open(input_file, 'w') as f:
            for line in self.CMDS:
                f.write("%s\n" %line)


class Assembler:
    """
    A class to gather all bispectrum components, including force and stress
    bispectrum. 
    
    Parameters
    ----------
    atom_type: list of str
        String of all atom types in the structure, i.e. ['Na', 'Cl']
    volume: float
        Volume of a crystal structure. Volume is used to convert the stress
        bispectrum components from eV to GPa.
    force: bool
        If true, return force bispectrum components.
    stress: bool
        If true, return stress bispectrum components
    """
    def __init__(self, atom_type, volume, force=True, stress=False):
        self.d_bispectrum_coefficient = {}
        elements = self._read_dump("dump.element", dtype='str')

        # sna
        bias_weight = []
        for _ in range(len(atom_type)):
            bias_weight.append([1.0/len(atom_type)])

        sna = self._read_dump("dump.sna")
        self.sna = []
        for atom in atom_type:
            sum = np.zeros(len(sna[0]))
            for i, ele in enumerate(elements):
                if ele == atom:
                    sum += sna[i]
            self.sna.append(sum/(i+1))
        self.sna = np.hstack((bias_weight, self.sna))
        self.sna = [np.ravel(self.sna)]
        
        all = self.sna
        #print(f"This is self.sna:\n{self.sna}")

        # snad
        if force == True:
            snad = self._read_dump("dump.snad")
            snad = np.split(np.asarray(snad), len(atom_type), axis=1)
            depth, rows, columns = np.shape(snad)
            x, y, z = int(columns/3), int(columns*2/3), columns
        
            self.snad = []
            for i in range(len(atom_type)):
                temp = []
                for j in range(len(elements)):
                    temp.append(snad[i][j][0:x])   # x-direction
                    temp.append(snad[i][j][x:y])  # y-direction
                    temp.append(snad[i][j][y:z]) # z-direction
                if self.snad == []:
                    self.snad = np.hstack((np.zeros((len(temp), 1)), temp))
                else:
                    temp = np.hstack((np.zeros((len(temp), 1)), temp))
                    self.snad = np.hstack((self.snad,temp))
            
            all = np.concatenate((self.sna, self.snad))
            #print(f"This is self.snad:\n{self.snad}")

        # snav
        if stress == True:
            snav = self._read_dump("dump.snav")
            snav = np.asarray(snav)
            snav = np.split(snav.sum(axis=0), len(atom_type))
            
            self.snav = []
            for i in range(len(atom_type)):
                temp = np.reshape(snav[i], (6,len(sna[0])))
                if self.snav == []:
                    self.snav = np.hstack((np.zeros((len(temp), 1)), temp))
                else:
                    temp = np.hstack((np.zeros((len(temp), 1)), temp))
                    self.snav = np.hstack((self.snav, temp))
            self.snav = self.snav / volume * 160.21766208 # eV to GPa
            all = np.concatenate((all, self.snav))
            #print(f"This is self.snav:\n{self.snav}")
        
        #print(f"This is all:\n{all}")
        self.bispectrum_coefficients = all
        
        # Remove the dump files after usage due to write error.
        os.remove("dump.element")
        os.remove("dump.sna")
        os.remove("dump.snad")
        os.remove("dump.snav")
   

    @staticmethod
    def _read_dump(filename, dtype='float'):
        arr = []

        file = open(filename)
        lines = file.readlines()
        n_lines = int(lines[3])
        goods = lines[9:]
         
        for i in range(n_lines):
            l = goods[i].split()
            
            if dtype == 'float':
                l = np.asarray([float(z) for z in l])
                arr.append(l)

            else:
                l = l[0]
                arr.append(l)
               
        return arr
