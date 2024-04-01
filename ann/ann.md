# Artificial Neural Network

Artififical Neural Network implemented from scratch using numpy.

## Provided classes and data

Data provided in Computational Intelligence project. Multi-class classification task with 7 target classes, and 10 numeric input features.

A template ANN class was also provided, but not used.

## Artificial Neural Network
Parameterized ANN implementation. Can solve classification, multi-class classifification, and regression problems.

#### Helpers
Helpers store relevant functions (implemented as lambdas, can be passed to mlp as function or string), perform parsing, and provide utility methods for MLP and the Optimizer
- mlp_func: Class that stores and verifies activation, loss, and regularization functions. Possibility for supporting custom functions [*NOT IMPLEMENTED*]
- init_func: Class that stores and verifies initialization functions for weights and biases. Supports custom initialization functions
- func_parser: Class that parses functions from strings and validates them using mlp_func and init_func. Also contains utility methods.
- Normalization: Stpres normalization methods.
- MLP_helper: Provides helper functions for MLP and Optimizer

#### MLP Implementation
Multi layer perceptron is implemented using a MLP class and a Layer class.

- Layer: Represents single layer in the MLP, and handles all computation for that layer.
- MLP: Handles back and forward propagation, training, testing, etc. Runs the neural network

## Grid Search Optimizer
Grid search optimizer implementation for optimizing MLP parameters.

Finds top k models on all combinations (optionally with subset and k-fold cross validation) and then evaluates top k on the full dataset to find the best model.

Takes param grid, fixed params, and training params. Param grid will be used for grid search.

