import numpy as np

class Layer:

    def __init__(self, rows, cols, args):
        init_range = args['init_range']
        hidden_act = args['hidden_act']


        self.w = np.zeros((rows, cols))
        self.b = np.full((cols), np.random.uniform(-init_range, init_range))
        self.b = self.b[:, None]
        self.a = np.zeros((cols))
        self.z = np.zeros((cols))
        self.delta = np.zeros((cols))
        self.w_grad = np.zeros((self.w.shape))
        self.b_grad = np.zeros((self.b.shape))
    
        self.f = hidden_act

        for i in range(rows):
            for j in range(cols):
                # initialize weights
                self.w[i][j] = np.random.uniform(-init_range, init_range)

