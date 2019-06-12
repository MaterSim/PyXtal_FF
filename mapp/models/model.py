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
        loss, lossprime = self.model.calculate_loss(parameters, 
                                                    lossprime=lossprime)

        return loss, lossprime
