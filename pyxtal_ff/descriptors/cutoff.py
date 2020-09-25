import numpy as np

class Cutoff:
    """ A class for computing cutoff functions (including derivatives) for 
    atom-centered descriptors. The existing cutoff functions in PyXtal_FF 
    can be found in:
        Singraber, A. (2019). J. Chem. Theory Comput., 15, 1827-1840.

    Parameters
    ----------
    function: str
        The type of cutoff function:
            1. cosine
                f(x) = 0.5*(cos(pi*x)+1)
            2. tanh
                f(x) = (tanh(1-x))**3
            3. exp
                f(x) = exp(1-(1/(1-x**2)))
            4. poly1
                f(x) = x**2(2*x-3)+1
            5. poly2
                f(x) = x**3(x(15-6*x)-10)+1
            6. poly3
                f(x) = x**4(x(x(20*x-70)+84)-35)+1
            7. poly4
                f(x) = x**5(x(x(x(315-70*x)-540)+420)-126)+1

        where x = R_ij/R_c
    """
    def __init__(self, function):
        self.function = function


    def calculate(self, R, Rc):
        if self.function == 'cosine':
            cutoff = Cosine(R, Rc)
        else:
            msg = f"The {self.function} function is not implemented."
            raise NotImplementedError(msg)
        return cutoff

    def calculate_derivative(self, R, Rc):
        if self.function == 'cosine':
            cutoff_prime = CosinePrime(R, Rc)
        else:
            msg = f"The {self.function} function is not implemented."
            raise NotImplementedError(msg)
        return cutoff_prime


def Cosine(Rij, Rc):
    # Rij is the norm 
    ids = (Rij > Rc)
    result = 0.5 * (np.cos(np.pi * Rij / Rc) + 1.)
    result[ids] = 0
    return result


def CosinePrime(Rij, Rc):
    # Rij is the norm
    ids = (Rij > Rc)
    result = -0.5 * np.pi / Rc * np.sin(np.pi * Rij / Rc)
    result[ids] = 0
    return result
