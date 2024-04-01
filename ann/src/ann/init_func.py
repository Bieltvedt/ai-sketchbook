from inspect import isfunction
import numpy as np


class init_func:
    """
    Class for initializing weights for a layer
    """

    ## Initialization strategies
    RANDOM = staticmethod(lambda input_size, layer_size: np.random.rand(input_size, layer_size))
    GUASSIAN = staticmethod(lambda input_size, layer_size: np.random.normal(0, 1, (input_size, layer_size)))
    XAVIER = staticmethod(lambda input_size, layer_size: np.random.normal(0, np.sqrt(1 / input_size), (input_size, layer_size)))
    HE = staticmethod(lambda input_size, layer_size: np.random.normal(0, np.sqrt(2. / input_size), (input_size, layer_size)))
    
    CONST_BIAS = staticmethod(lambda layer_size, const=0: np.ones(layer_size) * const)
    XAVIER_BIAS = staticmethod(lambda layer_size, const=1: const * np.random.uniform(-np.sqrt(1. / layer_size), np.sqrt(1. / layer_size), (layer_size)))
    ZERO_BIAS = staticmethod(lambda layer_size, const=None: np.zeros(layer_size))
        
    @classmethod
    def validate_custom_init(cls, init_method, input_size, layer_size):
        """
        validates custom initialization function
        - is function
        - takes 2 arguments
        - return ndarray with shape (layer_size, input_size)

        Args:
            init_method: custom init function to use
            input_size: size of the input (layer before)
            layer_size: size of the current layer

        Returns:
            initialized ndarray (layer_size, input_size)

        Raises:
            ValueError: if not a valid initialization function
        """
        if not isfunction(init_method):
            raise ValueError(f"custom initialization requires passing a custom function to init_method, was : {init_method.type()}")
        if init_method.__code__.co_argcount != 2: 
            raise ValueError(f"custom initialization function must take 2 paramaters, has {init_method.__code__.co_argcount}")
        
        w_test = init_method(input_size, layer_size)

        if not isinstance(w_test, np.ndarray):
            raise ValueError(f"custom initialization function must return ndarray, was {w_test.type()}")
        if w_test.shape != (input_size, layer_size):
            raise ValueError(f"Custome initialize function must return ndarray with shape {(layer_size, input_size)}, was {w_test.shape()}")
        return True

    @classmethod
    def validInitFunc(cls, init_method):
        """
        Validates initialization function

        Args:
            init_method: init function to validate

        Raises:
            ValueError: init_method is not a function or not in the list of valid functions
        """ 
        VALID_INIT_FUNCS = [cls.RANDOM, cls.GUASSIAN, cls.XAVIER, cls.HE]

        if not isfunction(init_method): 
            raise ValueError(f"Init method did not evaluate to function, was {init_method.type()}")
        
        if init_method in VALID_INIT_FUNCS: 
            return True
        
        if init_method not in VALID_INIT_FUNCS and cls.validate_custom_init(init_method):
            return True
        
        raise ValueError("Invalid custom init passed, this message show never be shown")
        
    @classmethod
    def validBiasFunc(cls, init_bias):
        VALID_BIAS = [cls.CONST_BIAS, cls.XAVIER_BIAS,init_func.ZERO_BIAS]
        return init_bias in VALID_BIAS 