class LossFunction:
    """General loss function for passing the model to the optimizer.

    Parameters
    ----------
    model: object
        The class representing the model.
    """
    def __init__(self, model, lossprime=False):
        self.model = model
        self.lossprime= lossprime

    def lossfunction(self, parameters):
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
        if self.lossprime:
            Loss, LossPrime = self.model.calculate_loss(parameters)
            return Loss, LossPrime
        else:
            Loss = self.model.calculate_loss(parameters)
            return Loss


############################## AUX functions ##################################


def calculate_descriptor_range(images, descriptor):
    """Calculate the range (min and max values) for the descriptors 
    corresponding to the ASE images.
    
    Parameters
    ----------
    images: dict
        ASE atomic objects.
    descriptor: dict
        List of atomic descriptor based on the symmetry function.
        
    Returns
    -------
    dict
        The range (min and max values) of the descriptors for each element.
    """

    no_of_images = len(images)

    desrange = {}

    for i in range(no_of_images):
        for element, des in descriptor[i]:
            if element not in desrange.keys():
                desrange[element] = [[_, _] for _ in des]
            else:
                assert len(desrange[element]) == len(des)
                for j, temp in enumerate(des):
                    if temp < desrange[element][j][0]:
                        desrange[element][j][0] = temp
                    elif temp > desrange[element][j][1]:
                        desrange[element][j][1] = temp

    return desrange
