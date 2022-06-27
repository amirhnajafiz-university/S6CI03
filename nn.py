import numpy as np



"""
Neural Network class
"""
class NeuralNetwork:
    """
    constructor
    """
    def __init__(self, *args, **kwargs):
        """
        Neural Network initialization.
        Given layer_sizes as an input, you have to design a Fully Connected Neural Network architecture here.
        :param args -- tuple of anonymous arguments
        :param kwargs -- dictionary of named arguments
        :param layer_sizes: A list containing neuron numbers in each layers. For example [3, 10, 2] means that there are
        3 neurons in the input layer, 10 neurons in the hidden layer, and 2 neurons in the output layer.
        """
        # TODO (Implement FCNNs architecture here)
        # getting the layer sizes
        self.layer_sizes = kwargs.get('layer_sizes')
        # initializing the parameters
        self.parameters = self.initialize_parameters_deep()

    @staticmethod
    def activation(x):
        """
        The activation function of our neural network, e.g., Sigmoid, ReLU.
        :param x: Vector of a layer in our network.
        :return: Vector after applying activation function.
        """
        # TODO (Implement activation function here)
        # sigmoid activation function
        return 1.0 / (1 + np.exp(-x))

    def forward(self, x):
        """
        Receives input vector as a parameter and calculates the output vector based on weights and biases.
        :param x: Input vector which is a numpy array.
        :return: Output vector
        """
        # calculate the deepness
        deepness = len(self.parameters) // 2

        # doing feedforward for each layer
        for le in range(1, deepness):  # using our linear activation forward
            x = self.linear_activation_forward(x, self.parameters['W' + str(le)], self.parameters['b' + str(le)])

        # last layer
        return self.linear_activation_forward(x, self.parameters['W' + str(deepness)], self.parameters['b' + str(deepness)])

    """
    using the activation function to perform a forwarding in
    feedforward steps.
    """
    def linear_activation_forward(self, a_prev, w, b):
        return self.activation((w @ a_prev) + b)

    """
    this method sets the parameters of our network.
    """
    def initialize_parameters_deep(self):
        parameters = {}

        for le in range(1, len(self.layer_sizes)):  # number of layers in the network
            parameters['W' + str(le)] = np.random.normal(size=(self.layer_sizes[le], self.layer_sizes[le - 1]))
            parameters['b' + str(le)] = np.zeros((self.layer_sizes[le], 1))

        return parameters

    def change_layer_parameters(self, new_layer_parameters, layer_num):
        self.parameters['W' + str(layer_num)] = new_layer_parameters['W']
        self.parameters['b' + str(layer_num)] = new_layer_parameters['b']
