def calculate_atomic_energy(descriptor, index, symbol):
    """
    This function calculate the atomic energy of a crystal. The input is given as 
    the descriptor, and the output is calculated for the corresponding energy about a 
    specific atom. The sum of these energies is the total energy of the 
    crystal.
    
    Parameters
    ----------
    descriptor: list
        The descriptor.
    symbol: str
        The atomic symbol.

    Returns
    -------
    float
        Energy.
    """

