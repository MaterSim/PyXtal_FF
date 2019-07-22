class Model:
    """The model class contains general functions to be used for potential development."""
    
    def _convert(self, structures, descriptors):
       """Convert and return the descriptors based on the structures.
       There are two descriptor types:
       - BehlerParrinello (symmetry functions)
       - Bispectrum
       Returns
       -------
       d: dict
           The atom-centered descriptors. The descriptors of 2 crystal 
           structures will be presented in this form:
           {1: {'G': [('Na', [1, 2, 3, 4]), ('Cl', [3, 2, 1, 0]), ...],
                'Gprime': [ ... ]},
            2: {'G': [('Na', [2, 2, 2, 2]), ('Cl', [3, 3, 3, 3]), ...],
                'Gprime': [ ... ]}}
       """
       d = {}
       model = descriptors._type
       N = len(structures)
       
       if model == 'BehlerParrinello':
           for i in range(N):
               bp = descriptors.fit(structures[i])
               d[i] = bp
       
       else:
           msg = f"The {model} is invalid."
           raise NotImplementedError(msg)

       return d
   

class LossFunction:
    """General loss function for passing the model to the optimizer.
    Parameters
    ----------
    model: object
        The class representing the model.
    """
    def __init__(self, model):
        self.model = model
    
    # For global optimization.
    def glossfunction(self, parameters):
        """This loss function takes parameters (array type) provided by Scipy 
        and feeds the parameters to the model (i.e. Neural Network) for loss 
        function calculation.
        Parameters
        ----------
        parameters: list
            A list of parameters to be optimized.
        
        Returns
        -------
        float
            The loss value.
        """
        loss = self.model.calculate_loss(parameters)
        
        return loss
    
    
    # For local optimization
    def llossfunction(self, parameters, lossprime=True):
        """This loss function takes parameters (array) given by the Scipy and 
        feeds the parameters to the model (i.e. NeuralNetwork) for loss 
        function calculation.
        Parameters
        ----------
        parameters: list
            A list of parameters to be optimized.
        lossprime: bool
            If True, calculate the derivative of the loss function.
        
        Returns
        -------
        float
            If lossprime is true, this lossfunction returns both the loss and
            its derivative values.
        """
        loss, lossprime = self.model.get_loss(parameters, 
                                              lossprime=lossprime)

        return loss, lossprime



# HY: Later we can make this as model class which will be inherited by neuralnetwork



        
