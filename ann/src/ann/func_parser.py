from inspect import isfunction
from ann.init_func import init_func
from ann.mlp_func import mlp_func


class func_parser:
    """
    Parses and validates functions for the mlp. Also handles custom function validation

    Uses mlp_func and init_func
    """


    ## Conversion dicts
    STR_TO_FUNC = {
        "sigmoid":mlp_func.SIGMOID, "tanh":mlp_func.TANH, "relu":mlp_func.ReLU, 
        "lrelu":mlp_func.LReLU, "softmax":mlp_func.SOFTMAX, 'regression':mlp_func.REGRESSION,
        'mse':mlp_func.MSE, 'cross_entropy':mlp_func.CROSS_ENTROPY, 
        "l2":mlp_func.L2REG, 'l1': mlp_func.L1REG, 'none':mlp_func.NONE,
        "random":init_func.RANDOM, "guassian":init_func.GUASSIAN, 
        "xavier":init_func.XAVIER, "he":init_func.HE, "xavier_bias":init_func.XAVIER_BIAS, 
        "const_bias":init_func.CONST_BIAS, "zero_bias":init_func.ZERO_BIAS
    }
    ROLE_TO_VALIDATION = {
        "hidden":mlp_func.validHiddenActivation, "output":mlp_func.validOutputActivation, 
        "loss":mlp_func.validLossFunction, "regularization":mlp_func.validRegFunction,
        "init":init_func.validInitFunc, "bias":init_func.validBiasFunc
    }
    ROLE_TO_FUNC = {
        "hidden": [mlp_func.SIGMOID, mlp_func.TANH, mlp_func.ReLU, mlp_func.LReLU],
        "output": [mlp_func.SIGMOID, mlp_func.SOFTMAX, mlp_func.REGRESSION],
        "loss": [mlp_func.MSE, mlp_func.CROSS_ENTROPY],
        "regularization": [mlp_func.NONE, mlp_func.L1REG, mlp_func.L2REG],
        "init": [init_func.RANDOM, init_func.GUASSIAN, 
                 init_func.XAVIER, init_func.HE],
        "bias": [init_func.XAVIER_BIAS, init_func.CONST_BIAS, init_func.ZERO_BIAS]
    }


    @classmethod
    def convert_and_validate(cls, function, role=None):
        """
        Converts and validates input function

        Args:
            function: Input function. Can be valid function string, predefined function, or custom function
            role: What role? ["hidden", "output", "loss", "regularization", "init"] default None and 
                auto inferrence, must pass explicit role if using a custom function.

        Returns:
            converted and validated function

        Raises:
            ValueError if function cannot be converted or if function or role invalid
        """
        function = cls.convert(function)

        if role == None:
            role = cls._role_from_function(function)

        valid = cls.validate(function, role)
        if not valid:
            raise ValueError("Invalid function! [This error should not be raised]")
        
        return function

    @classmethod
    def convert(cls, function):
        """
        Converts string to function if needed.

        Args:
            function: Function to be converted. String or callable (goes straight through)
        
        Returns:
            valid function

        Raises:
            ValueError if function is not string or callable, or invalid function string.
        """
        if isfunction(function):
            return function
        
        elif isinstance(function, str):
            function = function.lower()
            if cls._valid_str_for_conversion(function):
                return cls.STR_TO_FUNC[function]
            raise ValueError(f"Invalid function string: {function}")
        
        raise ValueError(f"function must be of type function or string, was {function.type()}")

    @classmethod
    def validate(cls, function, role):
        """
        Validates function for role
        Checks if it is predefined or custom. Different validation

        Args:
            function: callable function
            role: What role? ["hidden", "output", "loss", "regularization", "init"]

        Returns:
            Boolean indicating wether the function was valid, invalid functions should raise errors

        Raises:
            ValueError if function is not callable, invalid for the role, or fails custom checks.
        """
        if not isfunction(function):
            raise ValueError(f"Function was not converted before validation!") 
        
        if cls._is_predifined_func(function) and not cls.ROLE_TO_VALIDATION[role](function):
            raise ValueError(f"Function {function} not valid for {role}")
        elif cls._is_predifined_func(function):
            return cls.ROLE_TO_VALIDATION[role](function)
        
        if role == "init": 
            return init_func.validate_custom_init(function, 10, 20)
        else: 
            return mlp_func.validate_custom(function, role)
                

    @classmethod
    def _valid_str_for_conversion(cls, function: str):
        """
        Checks wether a string is valid for conversion

        Args:
            function: function string

        Returns:
            Boolean indicating wether function string can be converted..
        """
        if function in cls.STR_TO_FUNC.keys():
            return True
        return False
    
    @classmethod
    def _is_predifined_func(cls, function):
        """
        Checks wether a function is a predefined one.

        Args:
            function: callable function to check

        Returns:
            boolean indicating wether function is predefined

        Raises:
            ValueError if function is not callable.
        """
        if not isfunction(function):
            raise ValueError(f"_is_predifined_func only accepts functions, was {function.type()}")
        return function in cls.STR_TO_FUNC.values()
    
    @classmethod
    def _find_key(cls, dict, val):
        """
        Finds key in dictionary based on presense of value

        Args:
            dict: dict to check, has to have collection of values
            val: val to look for

        Returns:
            first key that contains val in its values, None if none match
        """
        for key, values in dict:
            if val in values:
                return key
        return None

    @classmethod
    def _role_from_function(cls, function):
        """
        Infers role from function, only works for predefined ones.

        Args:
            function: callable function

        Returns:
            role inferred from function

        Raises:
            ValueError if role could not be inferred or if function passed is not predefined.
        """
        role = cls._find_key(cls.ROLE_TO_FUNC, function)
        if role == None and cls._is_predifined_func(function):
            raise ValueError("Role was not able to be inffered from function")
        elif role == None:
            raise ValueError("Role can only be inffered from predefined functions, not custom ones")
        
        return role
    
    @classmethod
    def convert_func(cls, function):
        """Converts function to its string representation"""
        if not isfunction(function):
            raise ValueError(f"convert_func must be called with function, was {function.type}")

        if function in cls.STR_TO_FUNC.values():
            for key, value in cls.STR_TO_FUNC.items():
                if value == function:
                    return key
                
        return None