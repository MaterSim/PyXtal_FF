import sys
from collections import OrderedDict
import numpy as np
np.set_printoptions(threshold=sys.maxsize)


def calculate_dnnEnergy_dParameters(descriptor, scalings, weights, desrange, activation='tanh'):
    """
    I still have no clue what this function does.
    """
    ravel = Raveler(weights, scalings)
    dE_dP = None
    hiddenlayers = (3, 3)
    
    for i, (element, des) in enumerate(descriptor):
        scaling = scalings[element]
        W = weights_wo_bias(weights)
        W = W[element]
        dnnEnergy_dParameters = np.zeros(90)
        
        dnnEnergy_dWeights, dnnEnergy_dScalings = ravel.to_dicts(dnnEnergy_dParameters)
        outputs = forward(hiddenlayers, des, weights[element], desrange[element])
        
        D, delta, ohat = get_D_delta_ohat(outputs, W, residual=1)
        dnnEnergy_dScalings[element]['intercept'] = 1.
        dnnEnergy_dScalings[element]['slope'] = float(outputs[len(outputs)-1])
        
        for j in range(1, len(outputs)):
            dnnEnergy_dWeights[element][j] = float(scaling['slope']) * np.dot(np.matrix(ohat[j-1]).T, np.matrix(delta[j]).T)                       

        dnnEnergy_dParameters = ravel.to_vector(dnnEnergy_dWeights, dnnEnergy_dScalings)

        if dE_dP is None:
            dE_dP = dnnEnergy_dParameters
        else: 
            dE_dP += dnnEnergy_dParameters
    
    return dE_dP


def forward(hiddenlayers, descriptor, weight, desrange, activation='tanh'):
    layer = 0
    fingerprint = descriptor
    len_fp = len(fingerprint)
    for _ in range(len_fp):
        if (desrange[_][1] - desrange[_][0] > (10.**(-8.))):
            fingerprint[_] = -1.0 + 2.0 * ((fingerprint[_] - desrange[_][0]) /
                    (desrange[_][1] - desrange[_][0]))

    out = {}
    lin = {}

    temp = np.zeros((1, len_fp+1))
    temp[0, len_fp] = 1.0
    for _ in range(len_fp):
        temp[0, _] = fingerprint[_]
    out[0] = temp

    # Neural network architecture
    for i, hl in enumerate(hiddenlayers):
        layer += 1
        lin[i+1] = np.dot(temp, weight[i+1])
        if activation == 'tanh':
            out[i+1] = np.tanh(lin[i+1])
        elif activation == 'sigmoid':
            out[i+1] = 1. / (1. + np.exp(-lin[i+1]))
        elif activation == 'linear':
            out[i+1] = lin[i+1]
        temp = np.zeros((1, hl+1))
        temp[0, hl] = 1.0
        for _ in range(hl):
            temp[0, _] = out[i+1][0][_]

    # The output (i.e. energies)
    lin[layer+1] = np.dot(temp, weight[layer+1])
    if activation == 'tanh':
        out[layer+1] = np.tanh(lin[layer+1])
    elif activation == 'sigmoid':
        out[layer+1] = 1. / (1. + np.exp(-lin[layer+1]))
    elif activation == 'linear':
        out[layer+1] = lin[layer+1]

    return out

def weights_wo_bias(weights):
    """
    Return weights without the bias.
    """
    W = {}
    for k in weights.keys():
        W[k] = {}
        w = weights[k]
        for i in range(len(w)):
            W[k][i+1] = w[i+1][:-1]
    return W

def get_D_delta_ohat(outputs, W, residual):
    """
    Calculate and store delta, ohat, and D.
    These parameters are needed in calculating the derivative of the output with respect to the weights.

    output: dict
        Outputs of the neural network nodes.
    W: dict
        The weights of the neural network without the bias.
    residual: float
        True energy minus the neural network predicted energy
    """
    activation = 'tanh'

    N = len(outputs)

    # Delete the bias for outputs[0]
    o = outputs[0].tolist()
    o[0].pop(-1)
    outputs[0] = np.asarray(o)

    D = {}
    for i in range(N):
        n = np.size(outputs[i])
        D[i] = np.zeros((n, n))
        for j in range(n):
            if activation == 'linear':
                D[i][j,j] = 1.
            elif activation == 'sigmoid':
                D[i][j,j] = float(outputs[i][0,j]) * \
                        float(1. - outputs[i][0,j])
            elif activation == 'tanh':
                D[i][j,j] = float(1. - outputs[i][0,j] * outputs[i][0,j])

    delta = {}
    delta[N-1] = D[N-1] # Missing the (o_j - t_j)

    for i in range(N-2, 0, -1):
        delta[i] = np.dot(D[i], np.dot(W[i+1], delta[i+1]))

    ohat = {}
    for i in range(1, N):
        n = np.size(outputs[i-1])
        ohat[i-1] = np.zeros((1,n+1))
        for j in range(n):
            ohat[i-1][0,j] = outputs[i-1][0,j]
        ohat[i-1][0,n] = 1.

    return D, delta, ohat

class Raveler:
    """(CP) Class to ravel and unravel variable values into a single vector.

    This is used for feeding into the optimizer. Feed in a list of dictionaries
    to initialize the shape of the transformation. Note no data is saved in the
    class; each time it is used it is passed either the dictionaries or vector.
    The dictionaries for initialization should be two levels deep.

    weights, scalings are the variables to ravel and unravel
    """
    # why would scalings need to be raveled?
    def __init__(self, weights, scalings):

        self.count = 0
        self.weightskeys = []
        self.scalingskeys = []
        for key1 in sorted(weights.keys()):  # element
            for key2 in sorted(weights[key1].keys()):  # layer
                value = weights[key1][key2]
                self.weightskeys.append({'key1': key1,
                                         'key2': key2,
                                         'shape': np.array(value).shape,
                                         'size': np.array(value).size})
                self.count += np.array(weights[key1][key2]).size

        for key1 in sorted(scalings.keys()):  # element
            for key2 in sorted(scalings[key1].keys()):  # slope / intercept
                self.scalingskeys.append({'key1': key1,
                                          'key2': key2})
                self.count += 1

        self.vector = np.zeros(self.count)

    def to_vector(self, weights, scalings):
        """Puts the weights and scalings embedded dictionaries into a single
        vector and returns it. The dictionaries need to have the identical
        structure to those it was initialized with."""
        vector = np.zeros(self.count)
        count = 0
        for k in self.weightskeys:
            lweights = np.array(weights[k['key1']][k['key2']]).ravel()
            vector[count:(count + lweights.size)] = lweights
            count += lweights.size
        for k in self.scalingskeys:
            vector[count] = scalings[k['key1']][k['key2']]
            count += 1

        return vector

    def to_dicts(self, vector):
        """Puts the vector back into weights and scalings dictionaries of the
        form initialized. vector must have same length as the output of
        unravel."""

        assert len(vector) == self.count
        count = 0
        weights = OrderedDict()
        scalings = OrderedDict()

        for k in self.weightskeys:
            if k['key1'] not in weights.keys():
                weights[k['key1']] = OrderedDict()
            matrix = vector[count:count + k['size']]
            matrix = matrix.flatten()
            matrix = np.matrix(matrix.reshape(k['shape']))
            weights[k['key1']][k['key2']] = matrix.tolist()
            count += k['size']
        for k in self.scalingskeys:
            if k['key1'] not in scalings.keys():
                scalings[k['key1']] = OrderedDict()
            scalings[k['key1']][k['key2']] = vector[count]
            count += 1
        return weights, scalings

des = [('Pt', [5.684992851444012, 0.2106192000039041, 5.390166558890312, 0.19175430208750674, 4.256043075034483, 0.12440779985226418, 1.6822001930225576, 0.01734595164908539]), ('Pt', [5.684992851444012, 0.2106192000039041, 5.390166558890311, 0.19175430208750674, 4.256043075034483, 0.12440779985226418, 1.682200193022558, 0.01734595164908539]), ('Pt', [5.6849928514440125, 0.2106192000039041, 5.390166558890312, 0.19175430208750674, 4.2560430750344835, 0.12440779985226418, 1.6822001930225579, 0.01734595164908539]), ('Pt', [5.684992851444012, 0.2106192000039041, 5.390166558890311, 0.19175430208750674, 4.256043075034483, 0.12440779985226418, 1.6822001930225576, 0.01734595164908539]), ('Pt', [5.684992851444012, 0.260573858333391, 5.390166558890312, 0.22966405101881365, 4.256043075034482, 0.12780810507947324, 1.682200193022558, 0.008416147189561756]), ('Pt', [5.6849928514440125, 0.8243158479011873, 5.3901665588903125, 0.7712983122426443, 4.2560430750344835, 0.5696899152382175, 1.682200193022558, 0.14929026171055762]), ('Pt', [5.6849928514440125, 0.3670045986144035, 5.390166558890312, 0.33419051273280376, 4.256043075034483, 0.21646548578019642, 1.6822001930225579, 0.028991313929706674]), ('Pt', [5.6849928514440125, 0.8139181843767319, 5.390166558890311, 0.7807053382256871, 4.256043075034483, 0.6519559354010992, 1.682200193022558, 0.33004677211841793]), ('Cu', [2.9157330669115304, 0.7930148160406122, 2.7121151409412696, 0.7597201780982163, 1.9654199684210791, 0.6334054295082175, 0.5781025869314682, 0.3262842633342541]), ('Cu', [0.19255622232979935, 0.7930148160406122, 0.17076028162870607, 0.7597201780982163, 0.09813067248696411, 0.6334054295082175, 0.008025714613117277, 0.3262842633342541])]

scale = OrderedDict([('Cu', OrderedDict([('intercept', 0.9391512796710966), ('slope', 0.0)])), ('Pt', OrderedDict([('intercept', 0.9391512796710966), ('slope', 0.0)]))])

desrange = {'Pt': [[5.684992851444012, 5.6849928514440125], [0.2106192000039041, 0.8243158479011873], [5.390166558890311, 5.3901665588903125], [0.19175430208750674, 0.7807053382256871], [4.256043075034482, 4.2560430750344835], [0.12440779985226418, 0.6519559354010992], [1.6822001930225576, 1.682200193022558], [0.008416147189561756, 0.33004677211841793]], 'Cu': [[0.19255622232979935, 2.9157330669115304], [0.7930148160406122, 0.7930148160406122], [0.17076028162870607, 2.7121151409412696], [0.7597201780982163, 0.7597201780982163], [0.09813067248696411, 1.9654199684210791], [0.6334054295082175, 0.6334054295082175], [0.008025714613117277, 0.5781025869314682], [0.3262842633342541, 0.3262842633342541]]}

wei = OrderedDict([('Cu', OrderedDict([(1, [[0.2867095370137138, 0.6875361816566139, 0.20476213316294667], [-0.5282152035270963, 0.674356752438863, -0.30010310453220046], [0.21116264315218602, -0.3670609269347796, -0.20599343188794972], [-0.4031180539118127, 0.04806716366139363, 0.2431901147743598], [0.6272732689144809, -0.6734555644026794, 0.3070776071226855], [-0.34663712951769116, -0.27767771706053307, -0.44635676783005673], [-0.32695424318597716, -0.022124317511847735, -0.5224865602533721], [0.6745258529536773, 0.730069054218148, -0.30709107801017516], [-0.6379086267057873, 0.6690823234948104, -0.2314067748723091]]), (2, [[-0.05904690294245363, 0.09741516986138676, -0.33703843374070663], [-0.375962067028371, 0.045949216179256513, 0.2573757666131258], [0.11272591345100125, -0.43174307135716483, -0.365715721992824], [-0.31019908928800377, 0.8441618469563312, -0.13067729202566558]]), (3, [[-0.43205921308446193], [0.8745335195895447], [-0.5190227503531618], [1.222304116155228]])])), ('Pt', OrderedDict([(1, [[0.025305492817724984, 0.47728438933465234, 0.13441905174151036], [-0.23471426725441036, -0.6733237796644945, 0.4808123288723025], [-0.2604613157513156, -0.4678808066317265, 0.29445795609070946], [0.18461513451860745, -0.4548513857948866, 0.41099546292922184], [-0.26019868179762246, -0.2916654954698349, 0.4121386524007844], [-0.15921803811769764, 0.14786709124868935, -0.16670022362685588], [0.5890316668873978, 0.5407793165510446, -0.36712008952897546], [-0.20647054225245642, 0.43881224438975597, -0.6978086139369933], [0.3455579410736713, 0.5489909985265007, 0.46933923501365715]]), (2, [[0.8477849157936153, -0.2860150015396956, 0.8264418009883525], [-0.14740383953557412, 0.6515580643328884, 0.9394932994704321], [-0.35855797274540313, 0.8750326936225523, -0.8646964512014754], [0.609057884495789, 0.3173731890518614, -0.47489405642098603]]), (3, [[-1.2105686748678948], [0.7003878745489875], [1.186328579783253], [-0.9278157763938284]])]))])

z = calculate_dnnEnergy_dParameters(descriptor=des, scalings=scale, weights=wei, desrange=desrange, activation='tanh')

print(z)
