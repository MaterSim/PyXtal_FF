import numpy as np

class Cutoff:
    """ 
    A class for computing cutoff functions (including derivatives) for 
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
            3. exponent
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
        elif self.function == 'tanh':
            cutoff = Tanh(R, Rc)
        elif self.function == 'poly1':
            cutoff = Poly1(R, Rc)
        elif self.function == 'poly2':
            cutoff = Poly2(R, Rc)
        elif self.function == 'poly3':
            cutoff = Poly3(R, Rc)
        elif self.function == 'poly4':
            cutoff = Poly4(R, Rc)
        elif self.function == 'exponent':
            cutoff = Exponent(R, Rc)
        else:
            msg = f"The {self.function} function is not implemented."
            raise NotImplementedError(msg)
        return cutoff

    def calculate_derivative(self, R, Rc):
        if self.function == 'cosine':
            cutoff_prime = CosinePrime(R, Rc)
        elif self.function == 'tanh':
            cutoff_prime = TanhPrime(R, Rc)
        elif self.function == 'poly1':
            cutoff_prime = Poly1Prime(R, Rc)
        elif self.function == 'poly2':
            cutoff_prime = Poly2Prime(R, Rc)
        elif self.function == 'poly3':
            cutoff_prime = Poly3Prime(R, Rc)
        elif self.function == 'poly4':
            cutoff_prime = Poly4Prime(R, Rc)
        elif self.function == 'exponent':
            cutoff_prime = ExponentPrime(R, Rc)
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


def Tanh(Rij, Rc):
    ids = (Rij > Rc)
    result = np.tanh(1-Rij/Rc)**3
    result[ids]= 0
    return result


def TanhPrime(Rij, Rc):
    ids = (Rij > Rc)
    tanh_square = np.tanh(1-Rij/Rc)**2
    result = - (3/Rc) * tanh_square * (1-tanh_square)
    result[ids] = 0
    return result


def Poly1(Rij, Rc):
    ids = (Rij > Rc)
    x = Rij/Rc
    x_square = x**2
    result = x_square * (2*x-3) + 1
    result[ids] = 0
    return result


def Poly1Prime(Rij, Rc):
    ids = (Rij > Rc)
    term1 = (6 / Rc**2) * Rij
    term2 = Rij/Rc - 1
    result = term1*term2
    result[ids] = 0
    return result


def Poly2(Rij, Rc):
    ids = (Rij > Rc)
    x = Rij/Rc
    result = x**3 * (x*(15-6*x)-10) + 1
    result[ids] = 0
    return result


def Poly2Prime(Rij, Rc):
    ids = (Rij > Rc)
    x = Rij/Rc
    result = (-30/Rc) * (x**2 * (x-1)**2)
    result[ids] = 0
    return result


def Poly3(Rij, Rc):
    ids = (Rij > Rc)
    x = Rij/Rc
    result = x**4*(x*(x*(20*x-70)+84)-35)+1
    result[ids] = 0
    return result


def Poly3Prime(Rij, Rc):
    ids = (Rij > Rc)
    x = Rij/Rc
    result = (140/Rc) * (x**3 * (x-1)**3)
    result[ids] = 0
    return result


def Poly4(Rij, Rc):
    ids = (Rij > Rc)
    x = Rij/Rc
    result = x**5*(x*(x*(x*(315-70*x)-540)+420)-126)+1
    result[ids] = 0
    return result


def Poly4Prime(Rij, Rc):
    ids = (Rij > Rc)
    x = Rij/Rc
    result = (-630/Rc) * (x**4 * (x-1)**4)
    result[ids] = 0
    return result


def Exponent(Rij, Rc):
    result = np.zeros_like(Rij)
    ids = (Rij < Rc)
    x = Rij[ids]/Rc
    result[ids] = np.exp(1 - 1/(1-x**2))
    #result[ids] = 0
    return result


def ExponentPrime(Rij, Rc):
    ids = (Rij > Rc)
    x = Rij/Rc
    result = -2*x * np.exp(1 - 1/(1-x**2)) / (1-x**2)**2
    result[ids] = 0
    return result
