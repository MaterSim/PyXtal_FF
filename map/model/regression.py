from scipy.optimize import minimize

class Regressor:
    """
    This class is built for optimizing the loss function of NeuralNetwork 
    model. The Regressor class is based on SciPy optimizers. The minimization
    methods include:
    - BFGS
    - etc.

    Parameters
    ----------
    method: str
        Type of minimization scheme. 
        e.g.: 'BFGS'
    kwargs: dict
        Keywords for the method
    """

    def __init__(self, method='BFGS', user_kwargs=None):

        if method == 'BFGS':
            kwargs = {'method': 'BFGS',
                      'options': {'gtol': 1e-15, }}
        else:
            raise NotImplementedError("The method is not implemented yet, "
                                      "or it doesn't exist in SciPy.")
        
        if user_kwargs is not None:
            kwargs.update(user_kwargs)

        self.kwargs = kwargs
        self.optimizer = minimize


    def regress(self, model):
        """
        Perform the optimization scheme here.
        """
        self.kwargs.update({'jac': True,
                            'args': (True,)})
        x0 = [0., 0., 0.] # fix this


if __name__ == '__main__':
    Regressor(method='BFGS')
