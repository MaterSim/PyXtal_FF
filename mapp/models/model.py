class LossFunction:
    """
    General loss function for passing the model to the optimizer.

    Parameters
    ----------
    model: object
        The class representing the model.
    """
    def __init__(self, model):
        self.model = model

    def lossfunction(self, parameters, lossprime=False):
        """
        The value of loss function and the derivative are stored here.

        Parameters
        ----------
        parameters: list
            A list of parameters to be optimized.
        lossprime: bool
            If True, calculate the derivative of the loss function.
        """
        if lossprime:
            Loss, LossPrime = self.model.calculate_loss(parameters, lossprime)
            return Loss, LossPrime
        else:
            Loss = self.model.calculate_loss(parameters, lossprime)
            return Loss

################### AUX functions ##################################

def calculate_descriptor_range(images, descriptor):
    """
    Calculate the range for the descriptors corresponding to the images.
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
