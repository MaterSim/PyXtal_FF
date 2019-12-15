from copy import deepcopy

class gradient():
    """
    A class to compute the numerical gradient for different functions (df/dx)
    
    Parameters
    ----------
    fun: callable
        The objective function to be evaluated.
            `fun(x, *args) -> float`

    args: tuple
        arguments passed to the objective function

    epsilon: the stepwidth to evalute the gradients.
    **

    Usage:
    dfdx = gradient(f, args).dfdx()

    """
    def __init__(self, fun, args=(), epsilon=1e-6):
        self.fun = fun
        self.args = args
        self.epsilon = epsilon

    def get_perturbed_args(self, args, ids=[0,1], directions=['plus','plus']):
        args_list = deepcopy(list(args))
        for id, direction in zip(ids, directions):
            x = args_list[id]
            if direction == 'plus':
                x += self.epsilon
            elif direction == 'minus':
                x -= self.epsilon
        return tuple(args_list)

    def dfdx(self, id=0):
        """
        compute 2nd derivative d2fdx2
        """
        x0 = self.args[id]
        args_plus = self.get_perturbed_args(self.args, ids=[id], directions=['plus'])
        args_minus = self.get_perturbed_args(self.args, ids=[id], directions=['minus'])
        f_plus = self.fun(*tuple(args_plus))
        f_minus = self.fun(*tuple(args_minus))
        return (f_plus - f_minus)/(2*self.epsilon)

    def d2fdx2(self, id=0):
        """
        compute 2nd derivative d2fdx2
        """
        return self.d2fdxdy(ids=[id, id])

    def d2fdxdy(self, ids=[0, 1]):
        """
        compute 2nd derivative d2fdxdy
        Not working at the moment
        """
        args_plus_plus = self.get_perturbed_args(self.args, ids, directions=['plus', 'plus'])        
        args_plus_minus = self.get_perturbed_args(self.args, ids, directions=['plus', 'minus'])        
        args_minus_plus = self.get_perturbed_args(self.args, ids, directions=['minus', 'plus'])        
        args_minus_minus = self.get_perturbed_args(self.args, ids, directions=['minus', 'minus'])        
        f1 = self.fun(*tuple(args_plus_plus))
        f2 = self.fun(*tuple(args_plus_minus))
        f3 = self.fun(*tuple(args_minus_plus))
        f4 = self.fun(*tuple(args_minus_minus))
        return (f1-f2-f3+f4)/(4*self.epsilon**2)

if __name__ == '__main__':
    import numpy as np

    def f(x, y):
        return np.exp(x**2, y**2)
        #return x**2 + y**2

    x0 = np.linspace(0,1,5)
    y0 = np.linspace(0,1,5)
    grad = gradient(f, args=(x0, y0))
    print('\ninput x values:')
    print(x0)
    print('\ncomputing dfdx:')
    print(grad.dfdx())
    print('\ncomputing df2dx2:')
    print(grad.d2fdx2())
    print('\ncomputing df2dxdy:')
    print(grad.d2fdxdy([0,1]))
