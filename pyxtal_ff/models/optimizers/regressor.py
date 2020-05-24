#!/usr/bin/env  python
# encoding: utf-8

class Regressor:
    """ This class defines the local optimization method.

    Parameters
    ----------
    method: str
        Type of minimization scheme, e.g.: 'LBFGS'.
    user_kwargs: dict
        The arguments for the optimization method. These arguments are 
        passed with a dict.
    """
    def __init__(self, method, user_kwargs):
        self.method = method
        
        kwargs = {'lr': 1,}
        if method in ['lbfgs', 'LBFGS']:
            from pyxtal_ff.models.optimizers.lbfgs import LBFGS as optimizer
            _kwargs = {'max_iter': 100,
                       'max_eval': 15000,
                       'tolerance_grad': 1e-7,
                       'tolerance_change': 1e-9,
                       'history_size': 10,
                       'line_search_fn': 'strong_wolfe'}
        
        elif self.method in ['SGD', 'sgd']:
            from torch.optim import SGD as optimizer
            _kwargs = {'lr': 0.001,
                       'momentum': 0.,
                       'dampening': 0.,
                       'weight_decay': 0,
                       'nesterov': False}
        
        elif self.method in ['Adam', 'ADAM', 'adam']:
            from torch.optim import Adam as optimizer
            _kwargs = {'lr': 0.001,
                       'betas': (0.9, 0.999), 
                       'eps': 1e-08,
                       'weight_decay': 0, 
                       'amsgrad': False}

        elif self.method in ['lbfgsb']:
            from pyxtal_ff.models.optimizers.lbfgsb import LBFGSScipy as optimizer
            _kwargs = {'max_iter': 100,
                       'max_eval': 15000,
                       'tolerance_grad': 1e-7,
                       'tolerance_change': 1e-9,
                       'history_size': 10}

        else:
            msg = f"The {method} is not implemented yet."
            raise NotImplementedError(msg)
            
        kwargs.update(_kwargs)
        
        if user_kwargs is not None:
            kwargs.update(user_kwargs)

        self.optimizer = optimizer
        self.kwargs = kwargs


    def regress(self, models):
        """ Define optimization scheme and return the optimizer to models.

        Parameters
        ----------
        models: object
            Class representing the regression model.

        Returns
        -------
        regressor
            PyTorch optimizer or Scipy LBFGS-B.
        """
        try:
            params = models['model'].parameters()
        except:
            params = [p for model in models.values() for p in model.parameters()]

        if self.method in ['LBFGS', 'lbfgs']:
            regressor = self.optimizer(params, 
                                       lr=self.kwargs['lr'],
                                       max_iter=self.kwargs['max_iter'],
                                       max_eval=self.kwargs['max_eval'],
                                       tolerance_grad=self.kwargs['tolerance_grad'],
                                       tolerance_change=self.kwargs['tolerance_change'],
                                       history_size=self.kwargs['history_size'],
                                       line_search_fn=self.kwargs['line_search_fn'])

        elif self.method in ['SGD', 'sgd']:
            regressor = self.optimizer(params,
                                       lr=self.kwargs['lr'],
                                       momentum=self.kwargs['momentum'],
                                       dampening=self.kwargs['dampening'],
                                       weight_decay=self.kwargs['weight_decay'],
                                       nesterov=self.kwargs['nesterov'])

        elif self.method in ['adam', 'ADAM', 'Adam']:
            regressor = self.optimizer(params,
                                       lr=self.kwargs['lr'],
                                       betas=self.kwargs['betas'],
                                       eps=self.kwargs['eps'],
                                       weight_decay=self.kwargs['weight_decay'],
                                       amsgrad=self.kwargs['amsgrad'])

        elif self.method in ['lbfgsb']: 
            regressor = self.optimizer(params,
                                       max_iter=self.kwargs['max_iter'],
                                       max_eval=self.kwargs['max_eval'],
                                       tolerance_grad=self.kwargs['tolerance_grad'],
                                       tolerance_change=self.kwargs['tolerance_change'],
                                       history_size=self.kwargs['history_size'])
       
        return regressor
