import gc
import heapq
import itertools
import logging

import numpy as np
from ann.MLP_helper import MLP_helper


class GridSearchOptimizer:
    def _generate_combinations(self):
        """
        Generates all combinations of hyperparameter values.
        """
        keys, values = zip(*self.param_grid.items())
        combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        return combinations

    def _generate_hidden_sizes(self, min_layers, max_layers, neuron_options):
        """
        Generates a list of hidden layer sizes based on specified neuron options for each layer.
        
        Args:
            min_layers (int): Minimum number of layers.
            max_layers (int): Maximum number of layers.
            neuron_options (list): List of neuron numbers for each layer to choose from.
            
        Returns:
            list: A list of lists, where each list represents a possible combination of neuron counts.
        """
        sizes = []
        for num_layers in range(min_layers, max_layers + 1):
            for combination in itertools.product(neuron_options, repeat=num_layers):
                sizes.append(list(combination))
        return sizes
    
    
    def __init__(self, model, param_grid, train_data, val_data, batch_sizes, fixed_params, training_params, ohe=False, top_k = 5, logging_level="info"):
        """
        Initializes the Grid Search Optimizer.
        
        Args:
            model: A class representing your model, which should have a fit and evaluate method.
            param_grid: Dictionary with parameters names (str) as keys and lists of parameter settings to try as values.
            train_data: Tuple containing training data (input_train, target_train).
            val_data: Tuple containing validation data (input_val, target_val).
            batch_sizes: List of batch sizes to try
            fixed_params: Dict with param names (str) as keys and fixed param values
            training_params: Dict with training param names as keys and fixed param values
            ohe : boolean representing wether targets are one hot encoded
            top_k: top #k models to store at each time. Set to GridOptimizer.estimate_combinations to keep all
            logging_level: sets logging level for logger, reset when finished optimizing

            Note: 
            params not included in param_grid, fixed_params, or training_params will evaluate
            to their default values
        """
        # Set model and param dicts
        self.model = model
        self.param_grid = param_grid.copy()
        self.fixed_params = fixed_params.copy()
        self.training_params = training_params.copy()

        # Set data
        self.train_data = (train_data[0].copy(), train_data[1].copy())
        self.val_data = (val_data[0].copy(), val_data[1].copy())
        self.batch_sizes = batch_sizes

        # Initialize class params
        self.ohe = ohe
        self.top_k = top_k
        self.top_models = []
        self.top_k_models = []
        self.counter = 0

        # Setup logger
        self.logger = logging.getLogger("loggy")
        self.prev_logging_level = logging.getLevelName(self.logger.level)
        MLP_helper.update_logger_level(logging_level)
        
        # Setup hidden sizes and combinations
        self.param_grid['hidden_sizes'] = self._generate_hidden_sizes(*param_grid['hidden_sizes'])
        self.combinations = self._generate_combinations()
        self.n_combinations = len(self.combinations) * len(self.batch_sizes)
        self.logger.info("OPTIMIZER INITIALIZED")
        self.logger.info(f"N COMBINATIONS : {self.n_combinations}\n")

    def optimize(self, k_folds = 5, use_subset=True, subset_ratio = 0.2, penalty_type = "none", penalty_factor = 0, stop_after_top_k = False, store_train_acc = False):
        """
        Finds optimal hyperparameter combination

            1. Finds top k preforming models (preferably on subset with k-fold cv) out of all possible
                combinations of hyperparams
            2. Resets, Trains, and Evaluates top k preforming models on full datasets
            3. finds top k models, resets heap and logger level
            

            NOTE : this method can take really long to run, use estimate combinations to check if
                it is feasible to run (each one will involve training a model)
            NOTE : storing top k model instances is really expensive, dont make top_k too big
            NOTE : get top model with get_top_model() and list of top k models with get_top_k_models()
                Format - (acc, params, batch_size, model)

            Args:
                k_folds: #folds to use with k-fold cv in _find_top_k(), no k-fold cv if k_folds < 2
                use_subset: boolean representing wether to use a subset in _find_top_k()
                subset_ration: ratio of subset to original set
                penalty_type: penalty type to use when calculating accuracy. ['none', 'abs', 'std', 'ratio']
                penalty_factor: factor to apply to penalty before substracting, different depending on penalty_type
                    automatically scaled for std : [-inf;0.05] -> [0.1;1] | [0.05;0.1] -> [1;5] | std ~ [0.1;inf] -> [5;10]
                    f.x for std factor ~ [0.1, 1] (scaled), for ratio factor ~= 1/2
                stop_after_top_k: stops after after _find_top_k
                store_train_acc: store training accuracy
        """
        # Store params
        self.k_folds = k_folds
        self.use_subset = use_subset
        self.subset_ratio = subset_ratio
        self.penalty_type = penalty_type
        self.penalty_factor = penalty_factor
        self.stop_after_top_k = stop_after_top_k
        self.store_train_acc = store_train_acc

        # Warning if not using subsets on large search space
        if not use_subset and self.n_combinations > 100:
            self.logger.warning(f"Optimizing for {self.n_combinations} with full dataset!")
        
        # Find top k on reduced dataset
        self._find_top_k() 

        # Find best of top k models on full dataset
        self._final_evaluate()

        # reset logger level
        MLP_helper.update_logger_level(self.prev_logging_level)

        # Output top model acc & params
        model_info = self.get_top_model()
        print(f"\tTOP MODEL\t--\tACCURACY : {model_info['val_acc']}" + ('\t\t' + k + ':' + v) for k,v in model_info.items())


    def _find_top_k(self):
        """
        Executes the grid search optimization (preferably with k-fold CV and smaller subset)
        Finds top k performing models
        """
        # Setup
        self.logger.info(f"FIND TOP {self.top_k} MODELS")
        self.logger.info(f"Fixed Training Parameters : {self.training_params}")
        self.logger.info(f"Fixed Model Parameters : {self.fixed_params}" + ('\n' if self.k_folds > 1 else ''))
        if self.k_folds < 2:
            self.logger.info(f"No k-fold cross validation!\n")
        training_params = self.training_params.copy()

        # Setup data (no copy could be bad?)
        train_set = self.train_data
        val_set = self.val_data

        if self.use_subset:
            train_set = MLP_helper.get_subset(train_set[0], train_set[1], self.subset_ratio)
            val_set = MLP_helper.get_subset(val_set[0], val_set[1], self.subset_ratio)

        folds = [(train_set + val_set)] if self.k_folds < 2 else MLP_helper.k_fold_split(k_folds=self.k_folds, *train_set)

        # For progress bar
        percentages = range(0, 101, 2)
        previous_progress = set() 
        top_acc = - np.inf

        # Iterate through all possible combinations
        for combo in self.combinations:
            # Create model instance with params
            params = {**self.fixed_params, **combo}
            model_instance = self.model(**params)

            # Iterate through all possible batch sizes
            for batch_size in self.batch_sizes:
                # Progress bar implementation
                progress = self.counter / self.n_combinations * 100
                for threshold in percentages:
                    if threshold <= progress and threshold not in previous_progress:
                        previous_progress.add(threshold) 
                        self.logger.info(f"Progress: {threshold}% completed ({self.counter} / {self.n_combinations})\tbest accuracy : {top_acc}")  

                # Setup training params
                training_params["learning_rate"] = params["learning_rate"]
                training_params['batch_size'] = batch_size
                
                # Get accuracy and update top models
                val_accs, train_accs = self._validate_k_folds(model_instance, training_params, folds)
                accuracy = self._acc_with_penalty(val_accs, train_accs, self.penalty_type, self.penalty_factor)
                top_acc = max(top_acc, accuracy)
                
                model_info = {'model':model_instance, 'params':params, 'val_acc':accuracy, 'batch_size':batch_size}
                if self.store_train_acc: model_info['train_acc'] = np.mean(train_accs)
                    
                self._update_top_models(model_info)
            
                # Log and update counter
                self.logger.debug(f'Training model with params: {combo} and batch_size: {batch_size}')
                self.logger.debug(f"Validation accuracy: {accuracy}")
                self.counter+=1
        self.logger.info(f"Progress: {percentages[-1]}% completed ({self.counter} / {self.n_combinations})\tbest accuracy : {top_acc}\n") 
        gc.collect()
    
    def _final_evaluate(self):
        """
        Performs final evaluation with top k models on full training set
        Updates top models and extracts them to a list, resets heap
        """
        # Setup 
        self.logger.info("FINAL EVALUATION")
        training_params = self.training_params.copy()
        self.top_k_models = []
        
        while self.top_models:
            _, _, model_info = heapq.heappop(self.top_models)
            # setup params
            training_params['batch_size'] = model_info['batch_size']
            training_params['learning_rate'] = model_info['params']['learning_rate']
            model_instance = model_info['model']

            if not self.stop_after_top_k:
                # Reset and retrain model on full datsets, evaluate on validation set
                model_instance.reset()
                model_instance.set_epochs(1000)
                val_acc, train_acc = self._train_and_evaluate(model_instance, training_params, self.train_data, self.val_data)

                # Set model info
                model_info['val_acc'] = val_acc
                if self.store_train_acc: model_info['train_acc'] = train_acc
                
                self.counter +=1
                self.logger.info(
                    f"Model {self.counter - self.n_combinations} / {self.top_k}\t accuracy = {val_acc}" + 
                    ('\n' if self.counter - self.n_combinations >= self.top_k - 1 else '')
                )
                gc.collect()
            
            # Update top models
            self.top_k_models.append(model_info)

        # convert top models to list, sort, and format
        sorted_models = sorted(self.top_k_models, key=lambda x: x['val_acc'], reverse=True)
        
        # reset top models
        self.top_models = []
        self.top_k_models = sorted_models

    def _train_and_evaluate(self, model_instance, training_params, train_data, val_data):
        """
        Trains and evaluates model

        Args:
            model_instance : Initialzied model
            training_params : dict of training params, expects all except data
            input : tuple containing train and val input data
            targets: tuple containing train and val target data

        Returns:
            val_acc: validation accuracy
            train_acc: training accuracy
        """
        # Unwrap data tuples
        training_params['input'], training_params['targets'] = train_data
        training_params['input_val'], training_params['target_val'] = val_data

        # Train model instance
        model_instance.train(**training_params)

        # Get training and validation accuracy
        val_acc = model_instance.test( 
            input = training_params['input_val'], targets = training_params['target_val'],
            batch_size = training_params['batch_size'], ohe = self.ohe)
        train_acc = model_instance.test(
            input = training_params['input'], targets = training_params['targets'],
            batch_size = training_params['batch_size'], ohe = self.ohe)
        gc.collect()

        return val_acc, train_acc

    def _validate_k_folds(self, model_instance, training_params, folds):
        """
        Validates model performance using k-fold cv
        Sets up data (optional subset), accumulates accuracies over folds
        If self.k_folds < 2 -> evaluates model on full train and validation sets

        Args:
            model_instance: model instance to validate
            training_params: training params to use

        Returns:
            validation and training accuracy
        """
        # accumulate accuracies
        val_accs = []
        train_accs = []
        for input_train, targets_train, input_val, targets_val in folds:
            val_acc, train_acc = self._train_and_evaluate(model_instance, training_params, (input_train, targets_train), (input_val, targets_val))
            val_accs.append(val_acc)
            train_accs.append(train_acc)
        
        return val_accs, train_accs

    def _update_top_models(self, model_info):
        """
        Updates the list of top models
        Implemented with min heap, pushes untill k in heap, then pushes and pops smallest accuracy

        Args:
            model_info: val_acc, 
        """
        entry = (-model_info['val_acc'], self.counter, model_info)
        
        if len(self.top_models) < self.top_k:
            heapq.heappush(self.top_models, entry)
        else:
            heapq.heappushpop(self.top_models, entry)

    def _acc_with_penalty(self, val_accs, train_accs, penalty_type, penalty_factor):
        """
        Adjusts accuracy with an overfitting penalty, ensuring penalty_factor scales within [0, 1].

        Args:
            val_accs: Validation accuracies as a numpy array or a scalar.
            train_accs: Training accuracies as a numpy array or a scalar.

        Returns:
            Adjusted validation accuracy accounting for the penalty.
        """
        val_accs = np.atleast_1d(val_accs)
        train_accs = np.atleast_1d(train_accs)

        val_acc_avg = np.mean(val_accs)

        if penalty_type is None or penalty_factor <= 0 or penalty_type.lower() == "none":
            return val_acc_avg

        # Calculate average absolute difference between validation and training accuracies
        avg_diff = np.mean(np.abs(val_accs - train_accs))

        if penalty_type.lower() == "abs":
            penalty = penalty_factor * avg_diff

        elif penalty_type.lower() == "std":
            std_diff = np.std(val_accs - train_accs)
            penalty = penalty_factor * (1 / (1 + np.exp(-std_diff)))

        elif penalty_type.lower() == "ratio":
            ratio = avg_diff / val_acc_avg
            penalty = penalty_factor * min(ratio, 1)

        else:
            self.logger.warning(f"Unknown penalty type: {penalty_type}")
            return val_acc_avg

        adjusted_acc = max(val_acc_avg - penalty, 0)
        if adjusted_acc <= 1e-3: 
            self.logger.info(f"Found overfit model : acc = {val_acc_avg}  penalty = {penalty}  avg diff = {avg_diff}")
        return adjusted_acc
    
    
    def get_top_model(self):
        """
        Returns the top model based on validation accuracy.
        """
        if not self.top_k_models:
            raise ValueError(f"Optimization has not been run yet or no models were evaluated.")
        return self.top_k_models[0]

    def get_top_k_models(self):
        """
        Returns a list of the top k models sorted by validation accuracy.
        """
        if not self.top_k_models:
            raise ValueError(f"Optimization has not been run yet or no models were evaluated.")
        return self.top_k_models
    

    @staticmethod
    def estimate_combinations(param_grid, batch_sizes):
        """
        Estimates the total number of hyperparameter combinations, including different batch sizes.

        Args:
            param_grid (dict): The grid of parameters to search, with each key being a parameter name
                               and each value being a list of values to try.
            batch_sizes (list): List of batch sizes to try.

        Returns:
            int: The estimated number of combinations.
        """
        total_combinations = 1  # Start with a base of 1

        # Iterate through each parameter in the grid
        for key, values in param_grid.items():
            if key == 'hidden_sizes':
                # Special handling for hidden_sizes
                min_layers, max_layers, neuron_options = values
                # Generate all possible hidden layer sizes and count them
                layer_sizes_count = sum(len(neuron_options)**layers for layers in range(min_layers, max_layers + 1))
                total_combinations *= layer_sizes_count
            else:
                total_combinations *= len(values)

        # Multiply by the number of batch sizes
        total_combinations *= len(batch_sizes)

        return total_combinations