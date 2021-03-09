"""
Module for handling atomic elements.
"""

class Element:
    """
    Class for storing and accessing atomic elements. 
    Args:
        input_value: The atomic number, symbol, or name of the element
    """
    def __init__(self, input_value):
        self.input = input_value

        # list with atomic number z, short name, full name, valence, 
                # valence electrons, covalent radius, vdW radius, metallic radius
        self.elements_list = [
            [1,   'H' ],  
            [2,   'He'],
            [3,   'Li'],
            [4,   'Be'],
            [5,   'B' ],
            [6,   'C' ],
            [7,   'N' ],
            [8,   'O' ],
            [9,   'F' ],
            [10,  'Ne'],
            [11,  'Na'],
            [12,  'Mg'],
            [13,  'Al'],
            [14,  'Si'],
            [15,  'P' ],
            [16,  'S' ],
            [17,  'Cl'],
            [18,  'Ar'],
            [19,  'K' ],
            [20,  'Ca'],
            [21,  'Sc'],
            [22,  'Ti'],
            [23,  'V' ],
            [24,  'Cr'],
            [25,  'Mn'],
            [26,  'Fe'],
            [27,  'Co'],
            [28,  'Ni'],
            [29,  'Cu'],
            [30,  'Zn'],
            [31,  'Ga'],
            [32,  'Ge'],
            [33,  'As'],
            [34,  'Se'],
            [35,  'Br'],
            [36,  'Kr'],
            [37,  'Rb'],
            [38,  'Sr'],
            [39,  'Y' ],
            [40,  'Zr'],
            [41,  'Nb'],
            [42,  'Mo'],
            [43,  'Tc'],
            [44,  'Ru'],
            [45,  'Rh'],
            [46,  'Pd'],
            [47,  'Ag'],
            [48,  'Cd'],
            [49,  'In'],
            [50,  'Sn'],
            [51,  'Sb'],
            [52,  'Te'],
            [53,  'I' ],
            [54,  'Xe'],
            [55,  'Cs'],
            [56,  'Ba'],
            [57,  'La'],
            [58,  'Ce'],
            [59,  'Pr'],
            [60,  'Nd'],
            [61,  'Pm'],
            [62,  'Sm'],
            [63,  'Eu'],
            [64,  'Gd'],
            [65,  'Tb'],
            [66,  'Dy'],
            [67,  'Ho'],
            [68,  'Er'],
            [69,  'Tm'],
            [70,  'Yb'],
            [71,  'Lu'],
            [72,  'Hf'],
            [73,  'Ta'],
            [74,  'W' ],
            [75,  'Re'],
            [76,  'Os'],
            [77,  'Ir'],
            [78,  'Pt'],
            [79,  'Au'],
            [80,  'Hg'],
            [81,  'Tl'],
            [82,  'Pb'],
            [83,  'Bi'],
            [84,  'Po'],
            [85,  'At'],
            [86,  'Rn'],
            [87,  'Fr'],
            [88,  'Ra'],
            [89,  'Ac'],
            [90,  'Th'],
            [91,  'Pa'],
            [92,  'U' ],
            [93,  'Np'],
            [94,  'Pu'],
            [95,  'Am'],
            [96,  'Cm'],
            [97,  'Bk'],
            [98,  'Cf'],
            [99,  'Es'],
            [100, 'Fm'],
            [101, 'Md'],
            [102, 'No'],
            [103, 'Lr'],
            [104, 'Rf'],
            [105, 'Db'],
        ]
        """A list of atomic numbers, symbols, names, and other information, up
        to atomic number 105"""
        
        Z, self.Z = [], []
        for element in self.elements_list:
            if element[1] in self.input:
                Z.append(element)


        for inp in self.input:
            for z in Z:
                if inp == z[1]:
                    self.Z.append(z[0])

    def get_Z(self,):
        return self.Z

