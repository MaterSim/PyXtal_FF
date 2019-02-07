from pprint import pprint
import numpy as np

class assembler(object):

    def __init__(self, atom_type, volume, force=True, stress=True):
        
        elements = self._read_dump("dump.element", dtype='str')
        

        # SNA
        sna = self._read_dump("dump.sna")
        self.sna = []
        for atom in atom_type:
            sum = np.zeros(len(sna[0]))
            for i, ele in enumerate(elements):
                if ele == atom:
                    sum += sna[i]
            self.sna.append(sum/(i+1))
        bias = [[0.5], [0.5]]
        self.sna = np.hstack((bias, self.sna))
        #print(f"This is self.sna:\n{self.sna}")
        

        # SNAD
        snad = self._read_dump("dump.snad")
        snad = np.split(np.asarray(snad), 2, axis=1)
        self.snad = []

        for i in range(len(atom_type)):
            temp = []
            for j in range(len(elements)):
                temp.append(snad[i][j][0:5]) 
                temp.append(snad[i][j][5:10])
                temp.append(snad[i][j][10:15])
            if self.snad == []:
                self.snad = temp
            else:
                self.snad = np.vstack((self.snad,temp))
        self.snad = np.hstack((np.zeros((len(self.snad), 1)), self.snad))
        #print(f"This is self.snad:\n{self.snad}")


        # SNAV
        snav = self._read_dump("dump.snav")
        snav = np.asarray(snav)
        snav = np.split(snav.sum(axis=0), len(atom_type))
        self.snav = []
        for i in range(len(atom_type)):
            temp = np.reshape(snav[i], (6,len(sna[0])))
            if self.snav == []:
                self.snav = temp
            else:
                self.snav = np.vstack((self.snav, temp))
        self.snav = self.snav / volume * 160.21766208 # eV to GPa
        self.snav = np.hstack((np.zeros((len(self.snav), 1)), self.snav))
        #print(f"This is self.snav:\n{self.snav}")

        if force == True:
            all = np.concatenate((self.sna, self.snad))
        if stress == True:
            all = np.concatenate((all, self.snav))
        print(all)
    

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


if __name__ == "__main__":
    assembler(atom_type=['Na', 'Cl'], volume=184.3842)
