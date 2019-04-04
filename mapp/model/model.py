class LossFunction:
    def __init__(self, model):
        self.model = model

    def lossfunction(self, parameters, lossprime):

        Loss, LossPrime = self.model.calculate_loss(parameters, lossprime)

        if lossprime:
            return Loss, LossPrime
        else:
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
