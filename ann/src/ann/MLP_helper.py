import logging

import numpy as np
from ann.Layer import Layer
from ann.func_parser import func_parser


class MLP_helper:

    logger = logging.getLogger("loggy")

    @classmethod
    def convert_and_validate_functions(cls, activation_hidden, activation_output, loss_function, regularization, init_method, init_bias):
        hidden = func_parser.convert_and_validate(activation_hidden, "hidden")
        output = func_parser.convert_and_validate(activation_output, "output")
        loss = func_parser.convert_and_validate(loss_function, "loss")
        reg = func_parser.convert_and_validate(regularization, "regularization")
        init = func_parser.convert_and_validate(init_method, "init")
        bias = func_parser.convert_and_validate(init_bias, 'bias')

        return hidden, output, loss, reg, init, bias
    
    @classmethod
    def create_layers(cls, sizes, act_hidden, act_output, init, init_bias, bias_const, reg, reg_lambda):
        layers = []
        for i in range(0, len(sizes) - 2):
            l = Layer(input_size=sizes[i], layer_size=sizes[i + 1], activation_func=act_hidden,
                      init_method=init, init_bias=init_bias, bias_const=bias_const, regularization=reg, reg_lambda=reg_lambda)
            layers.append(l)

        output = Layer(input_size=sizes[-2], layer_size=sizes[-1], activation_func=act_output,
                       init_method=init, init_bias=init_bias, bias_const=bias_const, regularization=reg, reg_lambda=reg_lambda)
        
        return layers + [output]
    
    @classmethod
    def get_subset(cls, input_data, target_data, subset_ratio = 0.2):
        total_samples = input_data.shape[0]
        subset_size = int(total_samples * subset_ratio)

        # Generate random indices
        indices = np.random.choice(total_samples, subset_size, replace=False)

        subset_input_data = input_data[indices]
        subset_target_data = target_data[indices]

        return (subset_input_data, subset_target_data)
    
    @classmethod
    def k_fold_split(cls, input_data, target_data, k_folds=5):
        # Handle case where k < 2
        if k_folds < 2:
            cls.logger.warning(f"Attempted to split data into {k_folds}, returned NONE")
            return None
            
        # Shuffle indices
        indices = np.arange(input_data.shape[0])
        np.random.shuffle(indices)
        
        # Create folds
        fold_sizes = input_data.shape[0] // k_folds
        folds = []
        for i in range(k_folds):
            start = i * fold_sizes
            end = (i + 1) * fold_sizes if i != k_folds-1 else input_data.shape[0]
            test_indices = indices[start:end]
            train_indices = np.concatenate([indices[:start], indices[end:]])
            folds.append((input_data[train_indices], target_data[train_indices], input_data[test_indices], target_data[test_indices]))
        return folds

    @classmethod
    def update_logger_level(cls, new_level):
        """Updates the logger's level."""
        level = getattr(logging, new_level.upper(), None)
        if level is not None:
            cls.logger.setLevel(level)
        else:
            cls.logger.warning("Invalid logging level:", new_level)