import matplotlib.pyplot as plt
import numpy as np

class CNN:
    def __init__(self, input_dim, layers, loss):
        self.input_dim = input_dim
        self.layers = layers
        self.loss = loss
        
        self.loss_func = {
            'BinaryCrossEntropy' : (lambda z, y: np.mean(-z * np.log(y) - (1-z)*np.log(1-y))),
            'MSE' : (lambda z, y: np.array(((z - y) ** 2).mean(axis=1)).reshape(-1, 1))
        }

        self.loss_deriv = {
            'BinaryCrossEntropy' : (lambda z, y: ((1 - z)/(1 - y) - z/y)/np.size(z)),
            'MSE' : (lambda z, y: np.array(-2 * (z - y).mean(axis=1)).reshape(-1, 1))
        }

    def _define_dimentions(self):
        shape = self.input_dim
        for layer in self.layers:
            layer.set_input_shape(shape)
            shape = layer.output_shape

    def fit(self, X_train, y_train, num_epochs=10, learning_rate=0.01):
        self.error_arr = []
        self._define_dimentions()
        for ep in range(num_epochs):
            error = 0
            for X, y in zip(X_train, y_train):
                # Forward
                output = self.predict(X)

                # Error
                error += self.loss_func[self.loss](y, output)

                # Backward
                grad = self.loss_deriv[self.loss](y, output)
                for layer in reversed(self.layers):
                    grad = layer.backward(grad, learning_rate)

            self.error_arr.append(error/len(X_train))
            print(f"Epoch {ep+1}/{num_epochs} - loss: {self.error_arr[-1]:.4f}")

    def predict(self, X):
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    @property
    def error_plot(self):
        plt.plot(self.error_arr)
        plt.xlabel("Epochs")
        plt.ylabel("Error")
        plt.grid(True)
        plt.show()