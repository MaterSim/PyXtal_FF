import numpy as np

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

W = {1: np.asarray([[0.30626331343281765 , -0.635159872006889, 0.333129207782179], [-0.2981300493267071, -0.26714556982341336, 0.23194650082748913],[ 0.014315477997336123, -0.031114476082672926, 0.09039754686848844], [-0.10487178799248487, 0.5067339273659874, 0.5702385981395524], [0.7284408396149826, 0.5253493880536295, -0.5983114182046548], [0.5423577170934386, -0.2260793880377736, 0.5719158284711581], [-0.19414545752808987,0.6408427236825027, -0.7160489313886075],[0.008401326264966258, -0.06577636148380228, -0.43192757943057547]]), 2: np.asarray([[-0.10132493676244714,  0.6303828551802579,0.772932042664181],[-0.5532540511729775  , -0.039716491776979446,-0.7132112998861042  ],[ 0.06658961749181946 ,  0.6971786868967293  ,-0.3126839221166089  ]]), 3: np.asarray([[ 1.0571425243058776],[-0.7456398666694508],[ 1.1318552304593563]])}
outputs = {1: np.asarray([[ 0.9995148465894812 , -0.21580617772596472, -0.9249558791240553 ]]), 2: np.asarray([[0.33025452708000125, 0.5329856551023132 , 0.897848649234301]]), 3:np.asarray([[0.751934983740263]]), 0: np.asarray([[ 5.684992851444012 , -1., 5.390166558890312, -1., 4.256043075034483 , -1., 1.6822001930225576, -0.9444716779597784]])}

D, d, o = get_D_delta_ohat(outputs=outputs, W=W, residual=1)
print(D)
print(d)
print(o)
