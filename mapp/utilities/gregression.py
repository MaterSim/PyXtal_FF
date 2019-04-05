# Global regression

from ..model import LossFunction

class Regressor:
    """
    This class contains global optimization methods.

    Parameters
    ----------
    method: str
        Type of optimization scheme.
    user_kwargs: dict
        Keywords for the optimization scheme.
    """
    def __init__(self, method='DifferentialEvolution', user_kwargs=None):
        
        if method == 'DifferentialEvolution':
            from scipy.optimize import differential_evolution as optimizer
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
        """
        Run the optimization scheme here.
        """
        self.bounds = bounds

        f = LossFunction(model)
        optimize = optimizer(f.lossfunction(lossprime=False), self.bounds)

        return [optimize.x, optimize.fun]
