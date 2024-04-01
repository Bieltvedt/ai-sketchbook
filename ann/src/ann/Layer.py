import logging

import numpy as np
from ann.func_parser import func_parser
from ann.mlp_func import mlp_func


class Layer:
    """
    Represents a single layer in an artificial neural network (ANN). This class can be extended to create different types of layers.
    Each layer has an input size, layer size, activation function, and optional initialization method and regularization.
    """
    def __init__(self, input_size, layer_size, activation_func, init_method = None, init_bias='const_bias', bias_const=0, regularization = None, reg_lambda = 0.01) -> None:
        """
        Initializes the layer with the specified parameters.

        Args:
            input_size (int): The size of the input vector.
            layer_size (int): The number of neurons in the layer.
            activation_func (callable): The activation function to use for neurons in the layer.
            init_method (callable, optional): The method used to initialize the weights. Defaults to None.
            regularization (callable, optional): The regularization function to apply to the weights. Defaults to None.
            reg_lambda (float, optional): The regularization lambda parameter. Defaults to 0.01.
        """
        self.input_size = input_size
        self.layer_size = layer_size
        self.activation_func = activation_func
        self.init_method = init_method

        self.freeze = False
        self.weights = None
        self.bias = None
        self.x_values = []
        self.logger = logging.getLogger('loggy')

        self._init_regularization(regularization, reg_lambda)
        
        ## Initialize weights if init method available
        if init_method: self.initialize_weights(init_method, init_bias, bias_const)

    def _init_regularization(self, regularization, reg_lambda):
        """
        Initializes the regularization method and lambda for the layer.

        Args:
            regularization (callable): The regularization function to apply to the weights.
            reg_lambda (float): The regularization lambda parameter.
        """
        if regularization == None: regularization = mlp_func.NONE
            
        self.regularization = regularization
        self.reg_lambda = reg_lambda   

    def _apply_regularization(self):
        """
        Applies regularization to the layer's weights based on the defined regularization method and lambda.

        Returns:
            The regularization penalty to be applied to the weights.
        """
        return self.regularization(self.weights, self.reg_lambda)  
        
    def initialize_weights(self, init_method, init_bias, bias_const):
        """
        Initializes the layer's weights and biases using the specified initialization method.

        Args:
            init_method (callable): The method used to initialize the weights.
        """
        self.weights = init_method(self.input_size, self.layer_size)
        self.bias = init_bias(self.layer_size, bias_const)
        self.logger.debug(f"initial bias : {self.bias}")
        self.init_bias = init_bias
        self.bias_const = bias_const
        self.init_method = init_method

        self.initial_weights = self.weights.copy()
        self.initial_bias = self.bias.copy()


    def forward(self, input_batch):
        """
        Performs the forward pass of the layer using the given input batch.
        Called from MLP when preforming forward pass

        Args:
            input_batch (np.ndarray): The input data batch to process.

        Returns:
            np.ndarray: The output of the layer after applying the weights, biases, and activation function.

        Raises:
            ValueError: If the input batch size does not match the expected input size of the layer.
        """
        if self.weights is None or not self.init_method:
            raise ValueError("Layer weights have not been initialized!")
        
        if input_batch.shape[1] != self.input_size:
            raise ValueError(f"Actual input size does not match defined, was {input_batch.shape[1]}, should be {self.input_size}")

        z = np.dot(input_batch, self.weights) + self.bias
        x = self.activation_func(z)
        
        self.last_input = input_batch
        self.x_values.append(x)

        return x
    
    def calculate_gradient(self, output_error_batch):
        """
        Calculates the gradients for the layer's weights and biases based on the output error.

        Args:
            output_error_batch (np.ndarray): The error between the predicted and actual outputs.

        Returns:
            tuple: A tuple containing the gradients for the weights, biases, and the error to propagate to the previous layer.
        """
        if self.activation_func != mlp_func.SOFTMAX:
            # Ensure you're applying the derivative to the pre-activation values (z)
            z = np.dot(self.last_input, self.weights) + self.bias
            activation_derivative = mlp_func.derv(self.activation_func)(z)
            delta = output_error_batch * activation_derivative
    
        else:
            delta = output_error_batch  # For Softmax with cross-entropy, output_error_batch is already delta

        # Calculate weight and bias gradient
        weight_grad = np.dot(self.last_input.T, delta) / output_error_batch.shape[0]
        bias_grad = np.sum(delta, axis=0) / output_error_batch.shape[0]

        # Apply regularization
        weight_grad += self.regularization(self.weights, self.reg_lambda)
        
        # Calculate error to propagate to next layer
        error_to_propagate = np.dot(delta, self.weights.T)
        
        return weight_grad, bias_grad, error_to_propagate
    
    def update_weights(self, weight_grad, bias_grad, learning_rate):
        """
        Updates the layer's weights and biases based on the calculated gradients and learning rate.

        Args:
            weight_grad (np.ndarray): The gradient of the weights.
            bias_grad (np.ndarray): The gradient of the biases.
            learning_rate (float): The learning rate to use for the update.
        """
        if not self.freeze:
            # Clip gradients
            clipped_weight_grad = np.clip(weight_grad, -5, 5)
            clipped_bias_grad = np.clip(bias_grad, -5, 5)
            
            # Update weights and biases with clipped gradients
            self.weights -= learning_rate * clipped_weight_grad
            self.bias -= learning_rate * clipped_bias_grad

    def info(self):
        """ Prints information about the layer, including its configuration and current state. """
        print(f"""
        Layer - 
            input size : {self.input_size}
            layer size : {self.layer_size}
            weights : {self.weights.shape}
            
            activation : {func_parser.convert_func(self.activation_func)}
            regularization : {func_parser.convert_func(self.regularization)}
            init : {func_parser.convert_func(self.init_method)}

            Frozen : {self.freeze}
        """)

    def print_values(self):
        """ 
        Prints the current weights, biases, last input, and last output of the layer, 
        as well as the initial weights and biases.
        """
        print(f"WEIGHTS\n{self.weights}")
        print(f"BIAS\n{self.bias}")
        print(f"LAST INPUT\n{self.last_input}")
        print(f"LAST OUTPUT\n{self.x_values[-1]}")
        print(f"INITIAL WEIGHTS\n{self.init_weights}")
        print(f"INITIAL BIAS\n{self.init_bias}")
    
    def freeze_parameters(self):
        """ Freezes the layer's parameters, preventing them from being updated during training. """
        self.freeze = True

    def defrost_parameters(self):
        """ Unfreezes the layer's parameters, allowing them to be updated during training. """
        self.freeze = False

    def get_history(self):
        """ Returns the history of outputs generated by the layer. """
        return self.x_values

    def get_last_output(self):
        """ Returns the last output produced by the layer. """
        return self.x_values[-1]
    
    def get_last_inputs(self):
        """ Returns the last set of inputs processed by the layer. """
        return self.last_input
    
    def get_weights(self):
        """ Returns the current weights of the layer. """
        return self.weights
    
    def reset(self):
        """ 
        Resets the layer's weights and biases to their initial values and clears the history 
        of inputs and outputs.
        """
        self.weights = self.initial_weights
        self.bias = self.initial_bias
        self.freeze = False
        self.x_values = []
        self.last_input = None