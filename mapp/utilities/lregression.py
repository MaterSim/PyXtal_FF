# Local regression
from ..models.model import LossFunction


class Regressor:
    """This class contains global optimization methods.

    Parameters
    ----------
    method: str
        Type of minimization scheme, e.g.: 'BFGS'.
    user_kwargs: dict
        The arguments of the optimization function are passed by the dict 
        keywords.
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

        self.optimizer = optimizer
        self.kwargs = kwargs


    def regress(self, model):
        """
        Run the optimization scheme here.

        Parameters
        ----------
        model: object
            Class representing the regression model.

        Returns
        -------
        List
            List of the optimized parameters and loss value.
        """
        self.kwargs.update({'jac': True,
                            'args': (True,)})

        parameters0 = model.vector.copy()
        
        f = LossFunction(model)
        regression = self.optimizer(f.llossfunction, parameters0, **self.kwargs)

        return [regression.x, regression.fun]
