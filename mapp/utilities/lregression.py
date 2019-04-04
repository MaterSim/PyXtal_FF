# Local regression

from .model import LossFunction

class Regressor:
    """
    This class contains global optimization methods

    Parameters
    ----------
    method: str
        Type of minimization scheme. E.g.: 'BFGS'.
    user_kwargs: dict
        The arguments of the optimization function are passed by the keywords of the dict.
    """
    def __init__(self, method='BFGS', user_kwargs=None):

        if method == 'BFGS':
            from scipy.optimize import minimize as optimizer
            kwargs = {'method': 'BFGS',
                      'options': {'gtol': 1e-15, }}
        else:
            msg = "The method is not implemented yet."
            raise NotImplementedError(msg)
        
        if user_kwargs is not None:
            kwargs.update(user_kwargs)

        self.kwargs = kwargs


    def regress(self, model):
        """
        Run the optimization scheme here.

        Parameters
        ----------
        model: object
            Class representing the regression model.
        """
        self.kwargs.update({'jac': True,
                            'args': (True,)})
        parameters0 = model.vector.copy()
        function = LossFunction(model)
        
        opt = minimize(function.lossfunction, parameters0, **self.kwargs)

        return opt.x, opt.fun
