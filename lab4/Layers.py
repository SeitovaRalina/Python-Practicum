import numpy as np
from scipy import signal
from functools import reduce

activation_func = {
    'sigmoid': (lambda x: 1/(1 + np.exp(-x))),
        'tanh': (lambda x: np.tanh(x)),
        'relu': (lambda x: x*(x > 0)),
        'linear': (lambda x: x),
        # исп. на последнем слое, чтобы превратить t[-1] в вероятности 
        'softmax' : (lambda x: np.exp(x)/np.sum(np.exp(x), axis=1, keepdims=True))
        }

activation_deriv = {
    'sigmoid': (lambda x: x*(1-x)),
        'tanh': (lambda x: 1-x**2),
        'relu': (lambda x: (x >= 0).astype(float)),
        'linear': (lambda x: 1),
        }

loss_func = {
    'SparseCrossEntropy' : (lambda z, y: -np.log(np.array([z[j, y[j]] for j in range(len(y))]))),
    'MSE' : (lambda z, y: np.array(((z - y) ** 2).mean(axis=1)).reshape(-1, 1))
}

loss_deriv = {
    'SparseCrossEntropy' : (lambda z, y: z - y),
    'MSE' : (lambda z, y: np.array(-2 * (z - y).mean(axis=1)).reshape(-1, 1))
}

class Dense:
    def __init__(self, output_shape, activation):
        self.output_shape = output_shape
        self.bias = np.random.randn(output_shape, 1)
        self.activation = activation

    def set_input_shape(self, input_shape):        
        self.input_shape = input_shape
        self.weights = np.random.randn(self.output_shape, input_shape)

    def forward(self, input):
        self.input = input
        output = np.dot(self.weights, self.input) + self.bias
        self.output = activation_func[self.activation](output)
        return self.output

    def backward(self, output_gradient, learning_rate):
        output_gradient *= activation_deriv[self.activation](self.output)
        weights_gradient = np.dot(output_gradient, self.input.T)
        input_gradient = np.dot(self.weights.T, output_gradient)

        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient

        return input_gradient
    
class Flatten:
    def __init__(self):
        pass

    def set_input_shape(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = reduce(lambda x, y: x * y, input_shape)

    def forward(self, input):
        return np.reshape(input, (self.output_shape, 1))

    def backward(self, output_gradient, learning_rate):
        return np.reshape(output_gradient, self.input_shape)

class Conv2D:
    def __init__(self, kernel_size, depth, activation): 
        self.depth = depth # число ядер - 5    
        self.activation = activation 
        self.kernel_size = kernel_size # ядро (фильтр) 3 x 3            

    def set_input_shape(self, input_shape):
        input_depth, input_height, input_width = input_shape # (1, 28, 28)
        self.input_shape = input_shape
        self.input_depth = input_depth # число каналов - 1
        self.output_shape = (self.depth, input_height - self.kernel_size + 1, input_width - self.kernel_size + 1) # (число ядер, высота-фильтр+1, длина-фильтр+1) = (6, 28, 28)
        self.kernels_shape = (self.depth, input_depth, self.kernel_size, self.kernel_size) # (число ядер, каналы, фильтр, фильтр) = (5, 1, 3, 3)
        self.kernels = np.random.randn(*self.kernels_shape)
        self.biases = np.random.randn(*self.output_shape)

    def forward(self, input):
        self.input = input
        self.output = np.copy(self.biases)
        for i in range(self.depth): # по числу ядер
            # kernel = (i, каналы, фильтр, фильтр)
            for j in range(self.input_depth): # по числу каналов
                self.output[i] += signal.correlate2d(self.input[j], self.kernels[i, j], "valid")
        self.output = activation_func[self.activation](self.output)
        return self.output

    def backward(self, output_gradient, learning_rate):
        output_gradient *= activation_deriv[self.activation](self.output)
        kernels_gradient = np.zeros(self.kernels_shape)
        input_gradient = np.zeros(self.input_shape)

        for i in range(self.depth):
            for j in range(self.input_depth):
                kernels_gradient[i, j] = signal.correlate2d(self.input[j], output_gradient[i], "valid")
                input_gradient[j] += signal.convolve2d(output_gradient[i], self.kernels[i, j], "full")

        self.kernels -= learning_rate * kernels_gradient
        self.biases -= learning_rate * output_gradient
        
        return input_gradient
