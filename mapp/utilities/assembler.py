import os
import numpy as np

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
    def __init__(self, atom_type, volume, force=True, stress=True):
        
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


#if __name__ == "__main__":
#    assembler(atom_type=['Na', 'Cl'], volume=184.3842)
