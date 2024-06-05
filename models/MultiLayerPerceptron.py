import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

class Layer:
    def __init__(self, neurons, activation):
        self.neurons = neurons
        self.activation = activation

class MLP:
    def __init__(self, input_dim, hidden_layers, loss, optimizer):
        self.input_dim = input_dim
        self.hidden_dim = [x.neurons for x in hidden_layers]
        self.num_layers = len(self.hidden_dim)
        self.activation = [x.activation for x in hidden_layers]
        self.W = []      # weights
        self.b = []      # biases
        self.loss = loss
        self.error_arr = []
        self.optimizer = optimizer

        current_dim = self.input_dim
        for out_dim in self.hidden_dim:
            weight = np.random.randn(current_dim, out_dim) / np.sqrt(current_dim)
            self.W.append(weight)
            bias = np.random.randn(1, out_dim) / np.sqrt(current_dim)
            self.b.append(np.array(bias))
            current_dim = out_dim

        self.activation_func = {
            'sigmoid': (lambda x: 1/(1 + np.exp(-x))),
                'tanh': (lambda x: np.tanh(x)),
                'relu': (lambda x: x*(x > 0)),
                'linear': (lambda x: x),
                # исп. на последнем слое, чтобы превратить t[-1] в вероятности 
                'softmax' : (lambda x: np.exp(x)/np.sum(np.exp(x), axis=1, keepdims=True))
                }
        
        self.activation_deriv = {
            'sigmoid': (lambda x: x*(1-x)),
                'tanh': (lambda x: 1-x**2),
                'relu': (lambda x: (x >= 0).astype(float)),
                'linear': (lambda x: 1)
                }
        
        self.loss_func = {
            'SparseCrossEntropy' : (lambda z, y: -np.log(np.array([z[j, y[j]] for j in range(len(y))]))),
            'MSE' : (lambda z, y: np.array(((z - y) ** 2).mean(axis=1)).reshape(-1, 1))
        }

        self.loss_deriv = {
            'SparseCrossEntropy' : (lambda z, y: z - y),
            'MSE' : (lambda z, y: np.array(-2 * (z - y).mean(axis=1)).reshape(-1, 1))
        }

    def _forward(self, x):
        t = []
        h = [x]
        for i in range(self.num_layers):
            t.append(h[i] @ self.W[i] + self.b[i])
            h.append(self.activation_func[self.activation[i]](t[i]))
        return (t, h)
    
    def _backward(self, y, t, h):
        if self.hidden_dim[-1] > 1:
            y_full = np.zeros((len(y), self.hidden_dim[-1]))
            for j, yj in enumerate(y):
                y_full[j, yj] = 1
        else:
            y_full = y

        dE_dt = [None] * self.num_layers
        dE_dW = [None] * self.num_layers
        dE_db = [None] * self.num_layers
        dE_dh = [None] * self.num_layers

        for i in range(self.num_layers-1, -1, -1):
            if i == self.num_layers-1:
                dE_dh[i] = 0                
                dE_dt[i] = self.loss_deriv[self.loss](h[i+1], y_full)
            else:
                dE_dh[i] = dE_dt[i+1] @ self.W[i+1].T
                dE_dt[i] = dE_dh[i] * self.activation_deriv[self.activation[i]](t[i])
            dE_dW[i] = h[i].T @ dE_dt[i]
            dE_db[i] = np.sum(dE_dt[i], axis=0, keepdims=True)
        return (dE_dW, dE_db)

 
    def fit(self, X, y, batch_size=25, num_epochs=10):
        for ep in range(num_epochs):
            indices = np.random.permutation(len(X))
            X_shuffle = X[indices]
            y_shuffle = y[indices]
            loss_arr = []
            for i in range(len(X) // batch_size):
                X_batch = X_shuffle[i*batch_size : i*batch_size + batch_size]
                y_batch = y_shuffle[i*batch_size : i*batch_size + batch_size]

                # Forward
                t, h = self._forward(X_batch)
                E = np.mean(self.loss_func[self.loss](h[-1], y_batch))
                loss_arr.append(E)

                # Backward
                dE_dW, dE_db = self._backward(y_batch, t, h)

                # Update with optimizers
                self.optimizer.minimize(num_layers = self.num_layers, weights=self.W, gradients=dE_dW)
                self.optimizer.minimize(self.num_layers, self.b, dE_db)

            self.error_arr.append(sum(loss_arr) / len(loss_arr))
            print(f"Epoch {ep+1}/{num_epochs} - loss: {self.error_arr[-1]:.4f}")


    def predict(self, X):
        _, h = self._forward(X)
        return h[-1]
    
    def get_weights(self):
        print(self.W, self.b)
    
    def error_plot(self):
        fig = plt.figure(figsize = (10, 5))

        plt.plot(self.error_arr)
        plt.xlabel("Epochs")
        plt.ylabel("Error")
        plt.title("Error plot")

        st.pyplot(fig)
