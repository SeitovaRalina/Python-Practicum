import numpy as np

class AdamOptimizer:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.v = None
        self.G = None
        self.t = 0
    def minimize(self, num_layers, weights, gradients):
        v = [np.zeros_like(w) for w in weights]
        G = [np.zeros_like(w) for w in weights]
        t = 0
        for i in range(num_layers):
            v[i] = self.beta1 * v[i] + (1 - self.beta1) * gradients[i]
            G[i] = self.beta2 * G[i] + (1 - self.beta2) * np.square(gradients[i])
            v_norm = v[i] / (1 - self.beta2 ** (t + 1))
            G_norm = G[i] / (1 - self.beta2 ** (t + 1))
            weight_update = -self.learning_rate * v_norm / (np.sqrt(G_norm) + self.epsilon)
            weights[i] += weight_update
            t += 1
        return weights

class RMSPropOptimizer:
    def __init__(self, learning_rate=0.001, a=0.9, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.a = a
        self.epsilon = epsilon
        self.v = None
        self.G = None
        self.t = 0

    def minimize(self, num_layers, weights, gradients):
        G = [np.zeros_like(w) for w in weights]
        for i in range(num_layers):
            G[i] = self.a * G[i] + (1 - self.a) * np.square(gradients[i])
            weight_update = -self.learning_rate * gradients[i] / (np.sqrt(G[i]) + self.epsilon)
            weights[i] += weight_update
        return weights


class SGD:
    def __init__(self, learning_rate=0.001):
        self.learning_rate = learning_rate

    def minimize(self, num_layers, weights, gradients):
        for i in range(num_layers):
            weights[i] -= self.learning_rate * gradients[i]
        return weights