import numpy as np

def forward(hiddenlayers, descriptor, weight, desrange, activation='tanh'):
    """
    This function is the Neural Network architecture. The input is given as 
    the descriptor, and the output is calculated for the corresponding energy about a 
    specific atom. The sum of these energies is the total energy of the 
    crystal.

    Parameters
    ----------
    hiddenlayers: tuple
        (3,3) means 2 layers with 3 nodes each.
    descriptor: array
        The descriptor.
    weight: dict
        The Neural Network weights.
    desrange: array
        The range of the descriptor. This is used for scaling.
    activation:
        The activation function.


    """
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

    # Neural Network Architecture
    for i, hl in enumerate(hiddenlayers):
        layer += 1
        lin[i+1] = np.dot(temp, weight[i+1])
        if activation == 'tanh':
            out[i+1] = np.tanh(lin[i+1])
        temp = np.zeros((1, hl+1))
        temp[0, hl] = 1.0
        for _ in range(hl):
            temp[0, _] = out[i+1][0][_]
    
    # The output (i.e. energies)
    lin[layer+1] = np.dot(temp, weight[layer+1])
    if activation == 'tanh':
        out[layer+1] = np.tanh(lin[layer+1])

    return out

"""
fprange = [[5.684992851444012, 5.6849928514440125], [0.2106192000039041, 0.8243158479011873], [5.390166558890311, 5.3901665588903125], [0.19175430208750674, 0.7807053382256871], [4.256043075034482, 4.2560430750344835], [0.12440779985226418, 0.6519559354010992], [1.6822001930225576, 1.682200193022558], [0.008416147189561756, 0.33004677211841793]]

weight = {}
weight[1] = [[0.6884488334301113, -0.5090985564375761, -0.6849644943748299], [-0.4265572970370345, -0.6469715040790461, -0.40813303609533974], [-0.1442857061320748, 0.5709961516591247, -0.301811310852237], [0.32153236191618584, 0.055143163156648356, 0.6650753638032474], [-0.5659198749197374, -0.13479944071076766, 0.3622751927336134], [0.4950562727732555, 0.27883135877609466, -0.7184967531511595], [0.21633461081726812, -0.1835599043634154, -0.25840429948790244], [0.12485449652412872, -0.7368881788326198, -0.5709777205750262], [0.10421688183741695, -0.3376036486330747, 0.4495365997585512]]
weight[2] = [[-0.11709373255671807, 0.2904060751869979, -0.7340916768740287], [-0.7901483933813163, 0.296132091045904, -0.27697893449602673], [0.8612521674517635, 0.4184229142283502, 0.1362163708865718], [0.09519839686172338, -0.3515405369954723, 0.3604959477641756]]
weight[3] = [[-0.01718971813634451], [-0.21564170180939723], [0.7822144419375192], [-0.6461847995953667]]

fingerprint = [5.684992851444012, 0.2106192000039041, 5.390166558890312, 0.19175430208750674, 4.256043075034483, 0.12440779985226418, 1.6822001930225576, 0.01734595164908539]

out = forward(hiddenlayers=(3, 3), descriptor=fingerprint, weight=weight, desrange=fprange,
                            activation='tanh')

print(out)
"""
