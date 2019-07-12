import numpy as np


class Neighborhood:
    def __init__(self, neighbors_coordinates, neighbors_species, neighbors_images, neighbors_indexes):
        self.neighbors_coordinates = neighbors_coordinates
        self.neighbors_species = neighbors_species
        self.neighbors_images = neighbors_images
        self.neighbors_indexes = neighbors_indexes
        
        self.elements_list = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 
                              'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 
                              'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 
                              'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 
                              'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 
                              'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 
                              'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La',
                              'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb',
                              'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta',
                              'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl',
                              'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac',
                              'Th', 'Pa', 'U', 'Np', 'Pu', 'Am']


    def get_coordinates(self,):
        rows_length = []
        for _ in self.neighbors_coordinates:
            rows_length.append(len(_))
        
        self.x = len(rows_length)
        self.y = max(rows_length)
        self.z = 3

        array = np.zeros((self.x, self.y, self.z))

        for i in range(self.x):
            for j in range(rows_length[i]):
                for k in range(self.z):
                    array[i][j][k] += self.neighbors_coordinates[i][j][k]
        
        self.rows_length = rows_length

        return array, self.rows_length


    def get_species(self,):
        array = np.zeros((self.x, self.y,), dtype=int)
        neighbors_species = Element(2, self.neighbors_species).convert()

        for i in range(self.x):
            for j in range(self.rows_length[i]):
                array[i][j] += neighbors_species[i][j]

        return array
        

    def get_images(self,):
        array = np.ones((self.x, self.y, self.z), dtype=int)
        
        for i in range(self.x):
            for j in range(self.rows_length[i]):
                for k in range(self.z):
                    array[i][j][k] *= self.neighbors_images[i][j][k]

        return array


    def get_indexes(self,):
        array = np.ones((self.x, self.y,), dtype=int)
        array *= -1

        for i in range(self.x):
            for j in range(self.rows_length[i]):
                array[i][j] *= -1
                array[i][j] *= self.neighbors_indexes[i][j]

        return array


class Element:
    def __init__(self, dimension, array):
        self.dimension = dimension
        self.array = array
        
        self.elements_list = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 
                              'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 
                              'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 
                              'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 
                              'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 
                              'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 
                              'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La',
                              'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb',
                              'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta',
                              'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl',
                              'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac',
                              'Th', 'Pa', 'U', 'Np', 'Pu', 'Am']
    
    
    def convert(self,):
        """Convert the array of string into the numerical representation."""
        if self.dimension == 1:
            
            array = []
            for element in self.array:
                array.append(self.elements_list.index(element)+1)
        
        elif self.dimension == 2:
            array = []
            for _array in self.array:
                arr = []
                for element in _array:
                    arr.append(self.elements_list.index(element)+1)
                array.append(np.asarray(arr))
        
        return np.asarray(array)
                
# Test

#array = [['Mo', 'Xe'], ['Ni', 'Ti']]
#el = Element(2, array).convert()
#print(el)
