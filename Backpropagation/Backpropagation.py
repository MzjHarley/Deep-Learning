from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np


def get_data():
    x, y = make_moons(n_samples = 2000, noise=0.2, random_state=100)
    make_plot(x, y, "Classification Dataset Visualization ")
    print(x)
    print(x.shape)
    print(y)
    print(y.shape)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    return x_train, x_test, y_train, y_test

def make_plot(x, y, plot_name):
    plt.figure(figsize=(12,10))
    axes = plt.gca()
    axes.set(xlabel="x", ylabel="y")
    plt.title(plot_name, fontsize=30)
    plt.subplots_adjust(left=0.20)
    plt.subplots_adjust(right=0.80)
    plt.scatter(x[:, 0], x[:, 1], c=y.ravel(), s=40, cmap=plt.cm.Spectral)
    # cmap = plt.cm.Spectral : Points with label 1 are given one color, and points with label 0 are given another color.
    plt.show()
    plt.close()

class Layer:
    def __init__(self, n_input, n_neurons, activation=None, weights=None, bias=None):

        """
        :param int n_input: Number of input nodes
        :param int n_neurons: Number of output nodes
        :param string activation: Activation function type
        :param weights: Weight tensor,  generated inside the class by default
        :param bias: bias,generated inside the class by default
        """

        self.weights = weights if weights is not None else np.random.randn(n_input, n_neurons) * np.sqrt(1 / n_neurons)
        self.bias = bias if bias is not None else np.random.rand(n_neurons) * 0.1
        self.activation = activation
        self.last_activation = None
        self.error = None
        self.delta = None

    def activate(self, x):
        r = np.dot(x, self.weights) + self.bias
        self.last_activation = self._apply_activation(r)
        return self.last_activation

    def _apply_activation(self, r):
        if self.activation is None:
            return r
        elif self.activation == 'relu':
            return np.maximum(r, 0)
        elif self.activation == 'tanh':
            return np.tanh(r)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-r))
        return r

    def apply_activation_derivative(self, r):
        if self.activation is None:
            return np.ones_like(r)
        elif self.activation == 'relu':
            grad = np.array(r, copy=True)
            grad[r > 0] = 1.
            grad[r <= 0] = 0.
            return grad
        elif self.activation == 'tanh':
            return 1 - r ** 2
        elif self.activation == 'sigmoid':
            return r * (1 - r)
        return r


class NeuralNetwork:
    def __init__(self):
        self._layers = []

    def add_layer(self, layer):
        self._layers.append(layer)

    def feed_forward(self, x):
        for layer in self._layers:
            x = layer.activate(x)
        return x

    def backpropagation(self, x, y, learning_rate):
        output = self.feed_forward(x) #(2,)

        for i in reversed(range(len(self._layers))):
            layer = self._layers[i]

            if layer == self._layers[-1]:
                layer.error = output-y #(2,)
                layer.delta = layer.error * layer.apply_activation_derivative(output) #(2,)
            else:
                next_layer = self._layers[i + 1]
                layer.error = np.dot(next_layer.weights, next_layer.delta)
                # eg: np.dot((25,2),(2,))=>(25,2)@(2,1)
                layer.delta = layer.error * layer.apply_activation_derivative(layer.last_activation)

        for i in range(len(self._layers)):
            layer = self._layers[i]
            o_i = np.atleast_2d(x if i == 0 else self._layers[i - 1].last_activation)
            layer.weights -= layer.delta * o_i.T * learning_rate

    def train(self, x_train, x_test, y_train, y_test, learning_rate, max_epochs):
        y_onehot = np.zeros((y_train.shape[0], 2))
        y_onehot[np.arange(y_train.shape[0]), y_train] = 1
        mses = []
        accuracies=[]
        for i in range(max_epochs):
            for j in range(len(x_train)):
                self.backpropagation(x_train[j], y_onehot[j], learning_rate)
            if i % 10 == 0:
                mse = np.mean(np.square(y_onehot - self.feed_forward(x_train)))
                accuracy=self.accuracy(self.predict(x_test), y_test)
                mses.append(mse)
                accuracies.append(accuracy)
                print('Epoch: %s, MSE: %f' % (i, float(mse)))
                print('Accuracy: %.2f%%' % (accuracy * 100))
        return mses,accuracies

    def predict(self, x):
        return self.feed_forward(x)

    def accuracy(self, x, y):
        return np.sum(np.equal(np.argmax(x, axis=1), y)) / y.shape[0]

if __name__=='__main__':
    x_train, x_test, y_train, y_test=get_data()
    nn = NeuralNetwork()
    nn.add_layer(Layer(2, 25, 'sigmoid'))
    nn.add_layer(Layer(25, 50, 'sigmoid'))
    nn.add_layer(Layer(50, 25, 'sigmoid'))
    nn.add_layer(Layer(25, 2, 'sigmoid'))
    mses,accuracies=nn.train(x_train, x_test, y_train, y_test, 0.001, 200)

    pl = plt.figure()
    ax1 = pl.add_subplot(111)
    ax1.plot(range(1,21),mses)
    ax1.set_ylabel('MSE')
    ax2 = ax1.twinx()
    ax2.plot(range(1,21),accuracies, 'r')
    ax2.set_ylabel('Accuracy')
    ax1.legend('MSE',loc=9)
    ax2.legend('Acc',loc=8)
    plt.show()
    plt.close()
