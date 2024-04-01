from inspect import isfunction
import inspect
import numpy as np


class mlp_func:
    """
    This class stores functions for activation functions, loss functions, and their derivatives.

    Also includes some shared functionality.

    The class is accessed:
        func.SIGMOID (func.FUNCTION_NAME)
    """

    ## ACTIVATIONS
    SIGMOID = staticmethod(lambda x: 1 / (1 + np.exp(-x)))
    DSIGMOID = staticmethod(lambda x: (1 / (1 + np.exp(-x))) * (1 - (1 / (1 + np.exp(-x)))))

    TANH = staticmethod(lambda x: np.tanh(x))
    DTANH = staticmethod(lambda x: 1 - np.power(np.tanh(x), 2))

    ReLU = staticmethod(lambda x: np.where(x > 0, x, 0))
    DReLU = staticmethod(lambda x: np.where(x > 0, 1, 0))

    LReLU = staticmethod(lambda x, eps=1e-5: np.where(x > 0, x, x*eps))
    DLReLU = staticmethod(lambda x, eps=1e-5: np.where(x > 0, 1, eps))

    SOFTMAX = staticmethod(lambda x: np.exp(x - np.max(x, axis=1, keepdims=True)) / np.sum(np.exp(x - np.max(x, axis=1, keepdims=True)), axis=1, keepdims=True))
    # No softmax derivitive implementation, only the "cheat" one with cross entropy

    REGRESSION = staticmethod(lambda x: x)
    DREGRESSION = staticmethod(lambda x: 1)


    ## LOSS FUNCTIONS
    MSE = staticmethod(lambda y_true, y_pred: np.mean(np.power((y_true - y_pred), 2)))
    DMSE = staticmethod(lambda y_true, y_pred: (y_pred - y_true) / y_true.shape[0])

    CROSS_ENTROPY = staticmethod(lambda y_true, y_pred, eps=1e-15: -np.mean(np.sum(y_true * np.log(y_pred + eps), axis=1)))
    DCROSS_ENTROPY = staticmethod(lambda y_true, y_pred: y_pred - y_true)


    ## REGULARIZATION
    L2REG = staticmethod(lambda weights, lambda_reg: 2 * lambda_reg * weights)
    L1REG = staticmethod(lambda weights, lambda_reg: np.sign(weights) * lambda_reg)
    NONE = staticmethod(lambda weights, lambda_reg: np.zeros_like(weights))


    @classmethod
    def validHiddenActivation(cls, function) -> bool:
        """Checks if function is a valid, predefined, hidden activation"""
        return function in [cls.SIGMOID, cls.TANH, cls.ReLU, cls.LReLU]
    
    @classmethod
    def validOutputActivation(cls, function) -> bool:
        """Checks if function is a valid, predefined, output activation"""
        return function in [cls.SIGMOID, cls.SOFTMAX, cls.REGRESSION]
    
    @classmethod
    def validLossFunction(cls, function) -> bool:
        """Checks if function is a valid, predefined, loss function"""
        return function in [cls.MSE, cls.CROSS_ENTROPY]
    
    @classmethod
    def validRegFunction(cls, function) -> bool:
        """Checks if function is a valid, predefined, regularization function"""
        return function in [cls.NONE, cls.L1REG, cls.L2REG]
    
    @classmethod
    def validate_custom(cls, function, role):
        """
        Validates custom function
        Checks args and output shape

            NOTE : No implementation for custom dervitives, so custom functions dont work!

            Args: 
                function: function to check
                role: role the function will fufill 
        """
        if not isfunction(function):
            raise ValueError(f"Custom {role} function must be callable, was {function.type()}")
        
        arg_spec = inspect.getfullargspec(function)
        args = arg_spec.args
        defaults = arg_spec.defaults if arg_spec.defaults is not None else []
        
        # Expected arguments
        expected_args = {
            "loss": ["y_true", "y_pred", "eps"],
            "output": ["x", "eps"],
            "hidden": ["x", "eps"],
            "regularization": ["weights", "lambda_reg", "eps"]
        }
        
        # Validate args
        if role not in expected_args:
            raise ValueError(f"Invalid function type: {role}. Must be one of {list(expected_args.keys())}")
        
        missing_args = [arg for arg in expected_args[role] if arg != 'eps' and arg not in args]
        if missing_args:
            raise ValueError(f"Missing required arguments for {role} function: {missing_args}")
        
        unexpected_args = [arg for arg in args if arg not in expected_args[role]]
        if unexpected_args:
            raise ValueError(f"Unexpected arguments for {role} function: {unexpected_args}")
        

        # Check for epsilon default value if eps is expected, must have a default value
        if 'eps' in expected_args[role] and 'eps' in args:
            eps_index = args.index('eps') - (len(args) - len(defaults)) 
            if eps_index >= 0:
                eps_default = defaults[eps_index]
                if eps_default is None:
                    raise ValueError("Epsilon parameter must have a default value")
            else:
                # Epsilon is expected but no default value is provided
                raise ValueError("Epsilon parameter is expected to have a default value but none was provided")


        # validate shape (can be changed later)
        dummy_input_shapes = {
            "loss": {"shapes": [(10, 1), (10, 1)], "validation":cls._validate_shape_loss},
            "output": {"shapes": (25, 50), "validation":cls._validate_shape_actiavation}, 
            "input": {"shapes": (25, 50), "validation":cls._validate_shape_actiavation}, 
            "regularization": {"shapes": (100, 125), "validation":cls._validate_shape_regularization}
        }

        dummy_input_shapes[role]["validation"](function, dummy_input_shapes[role]["shapes"])

    
    @classmethod
    def _validate_shape_loss(cls, function, shapes):
        """Validates shape for loss function"""
        dummy_y_true = np.random.rand(*shapes[0])
        dummy_y_pred = np.random.rand(*shapes[1])
        output = function(dummy_y_true, dummy_y_pred)

        if not np.isscalar(output) and output.shape != (shapes[0][0], 1):
            raise ValueError(f"loss function shape was unexpected, should be scalar or {(shapes[0][0], 1)}, was {output.shape}")
        
    @classmethod
    def _validate_shape_actiavation(cls, function, input_shape):
        """Validates shape for activation function"""
        dummy_input = np.random.rand(*input_shape)
        output = function(dummy_input)

        if not output.shape == input_shape:
            raise ValueError(f"activation function shape was unexpected. Should be {input_shape}, was {output.shape}")
        
    @classmethod
    def _validate_shape_regularization(cls, function, weights_shape):
        """Validates shape for regularization function"""
        dummy_weights = np.random.rand(*weights_shape)
        dummy_lambda = 0.001
        output = function(dummy_weights, dummy_lambda)

        if not output.shape == weights_shape:
            raise ValueError(f"regularization function shape was unexpected. Should be {weights_shape}, was {output.shape}")
        
    
    @classmethod
    def derv(cls, function):
        """
        Retrieves the derivative function of a given function.

        Args:
            function: The function to derivate

        Returns:
            The derivative function if it exists, otherwise raises a ValueError.

        """
        FUNC_TO_DERV = {cls.SIGMOID:cls.DSIGMOID, cls.TANH:cls.DTANH, cls.ReLU:cls.DReLU, cls.LReLU:cls.DReLU,
                        cls.REGRESSION:cls.DREGRESSION, cls.MSE:cls.DMSE, cls.CROSS_ENTROPY:cls.DCROSS_ENTROPY}

        if function == cls.SOFTMAX:
            raise ValueError("Softmax derivitive not implemented")
            
        if (not function in FUNC_TO_DERV):
            raise ValueError(f"Derivative not found for function : {function}")
        
        return FUNC_TO_DERV[function]