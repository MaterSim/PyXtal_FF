import os
import numpy as np

class assembler(object):

    def __init__(self, atom_type, volume, force=True, stress=True):
        
        elements = self._read_dump("dump.element", dtype='str')
        

        # SNA
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
        self.sna = np.ravel(self.sna)
        print(f"This is self.sna:\n{self.sna}")
        

        # SNAD
        snad = self._read_dump("dump.snad")
        snad = np.split(np.asarray(snad), len(atom_type), axis=1)
        self.snad = []

        for i in range(len(atom_type)):
            temp = []
            for j in range(len(elements)):
                temp.append(snad[i][j][0:5]) 
                temp.append(snad[i][j][5:10])
                temp.append(snad[i][j][10:15])
            if self.snad == []:
                self.snad = np.hstack((np.zeros((len(temp), 1)), temp))
            else:
                temp = np.hstack((np.zeros((len(temp), 1)), temp))
                self.snad = np.hstack((self.snad,temp))
        print(f"This is self.snad:\n{self.snad}")


        # SNAV
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
        #print(f"This is self.snav:\n{self.snav}")

        print(self.sna.shape)
        print(self.snad.shape)
        print(self.snav.shape)
        if force == False and stress == False:
            self.bispectrum_coefficients = self.sna
        if force == True:
            self.bispectrum_coefficients = np.concatenate(([self.sna], self.snad))
        if stress == True:
            self.bispectrum_coefficients = np.concatenate((all, self.snav))

    
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
