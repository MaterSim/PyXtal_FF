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
    print(outputs)

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

W = {1: np.asarray([[0.6021912447675087 ,  0.20159678884253546,  0.8698851617223289], [0.0442364132457865 , -0.6364249170857932 ,  0.19910202147714356],[0.2161450297882703 , -0.7464380441449925 , -0.16399216256960672], [-0.3331657701030139 , -0.7277670378648747 ,  0.35685693190197804], [0.8766301078338161 ,  0.685894119909297  , -0.2663169689289923], [-0.2536970692781613 , -0.03620470801446468, -0.816319078125163], [-0.8279806238417097 , -0.09137465948927743, -0.20172810356430884],[0.1701123838803728 , -0.9273677836099344 , -0.6437484319483292]]), 2: np.asarray([[0.5356876430134491 , -0.47750405434648174, -0.26878756160027106],[0.4283899625178425 ,  0.04434980436966754, -0.7655349945872325],[-0.7315767418867314 , -0.21772698157919718, -0.3978728621425652]]), 3: np.asarray([[1.110783828616279],[-0.3027953845763446],[-1.048976150170909]])}
outputs = {1: np.asarray([[-0.9381402712699044, -0.5753908265766916, -0.7174661997371591]]), 2: np.asarray([[-0.11043088069993907,  0.7871045728724445 ,  0.7264808824171741]]), 3:np.asarray([[-0.9434132394898601]]), 0: np.asarray([[-1., 0.7930148160406122, -1., 0.7597201780982163, -1., 0.6334054295082175, -1., 0.3262842633342541]])}

D, d, o = get_D_delta_ohat(outputs=outputs, W=W, residual=1)
#print(o)
