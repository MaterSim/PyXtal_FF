# Global regression


import sys
sys.path.append("..")

from scipy.optimize import differential_evolution as optimizer
from models.model import LossFunction


class Regressor:
    """This class consists of global optimization methods.

    Parameters
    ----------
    method: str
        Type of optimization scheme.
    user_kwargs: dict
        Keywords for the optimization scheme.
    """
    def __init__(self, method='DifferentialEvolution', user_kwargs=None):
        self.method = method
        if self.method == 'DifferentialEvolution':
            kwargs = {'strategy': 'best1bin',
                      'maxiter': 1000,
                      'popsize': 15,
                      'tol': 0.0001}
        else:
            msg = "The method is not implemented yet."
            raise NotImplementedError(msg)

        if user_kwargs is not None:
            kwargs.update(user_kwargs)

        self.kwargs = kwargs


    def regress(self, model, bounds):
        """Run the optimization scheme here.
        
        Parameters
        ----------
        model: class
            Class representing the machine learning model.
        bounds: List of tuples
            The tuples describe the min and max values for the global 
            searching algorithm.
        
        Returns
        -------
        List
            List of the optimized parameters and loss value.
        """
        self.bounds = bounds

        f = LossFunction(model, lossprime=False)
        regression = optimizer(f.lossfunction, self.bounds)

        return [regression.x, regression.fun]
