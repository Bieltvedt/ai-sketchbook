import logging

import numpy as np
from ann.MLP_helper import MLP_helper
from ann.mlp_func import mlp_func


class MLP:

    def _init_functions(self, hidden, output, loss, reg, init, init_bias):
        hidden, output, loss, reg, init, init_bias = MLP_helper.convert_and_validate_functions(hidden, output, loss, reg, init, init_bias)
        self.act_hidden = hidden
        self.act_output = output
        self.loss = loss
        self.reg = reg
        self.init = init
        self.init_bias = init_bias

    def _init_sizes(self, input_size, output_size, hidden_sizes):
        self.input_size  = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.sizes = [input_size] + hidden_sizes + [output_size]

    def _init_layers(self, sizes, act_hidden, act_output, init, init_bias, bias_const, reg, reg_lambda):
        layers = MLP_helper.create_layers(sizes, act_hidden, act_output, init, init_bias, bias_const, reg, reg_lambda)
        self.layers = layers
            

    def __init__(
            self, input_size, output_size, hidden_sizes = [], 
            activation_function_hidden = "sigmoid", activation_function_output = "sigmoid", 
            init_method = "He", init_bias = "const_bias", bias_const=0, regularization = 'none', reg_lambda = 0.01, 
            loss_function="mse", learning_rate=0.01, epochs=1000, decay_rate = None,
            logging_level = "warning"):
        """
        Initializes an MLP instance.

        Args:
            input_size: Number of input features.
            output_size: Number of output neurons.
            hidden_sizes: List of sizes for hidden layers.

            activation_function_hidden: Activation function for hidden layers.
            activation_function_output: Activation function for output layer.
            loss_function: Loss function for training ('mse', 'cross_entropy')
            init_method: Weight initialization method ('random', 'Gaussian', 'Xavier', 'He')
            regularization: regularization method (l1, l2, none)

            reg lambda: controlls amount of regularizarion
            learning_rate: Learning rate for gradient descent.
            epochs: Number of training epochs.
            decay_rate : for learning rate decrease. learning_rate = learning_rate * (1 - decay_rate * epoch)

            Dervitives for custom funcs!!!
        """
        # Initialize "basic" values
        self.reg_lambda = reg_lambda
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.epochs = epochs
        self.bias_const = bias_const

        self.x_values = []
        self.validation = []
        self.training_accuracies = []
        self.validation_accuracies = []
        self.y_true = None
        self.y_pred = None
        self.stop_training = False

        # logging
        self.logger = logging.getLogger("loggy")
        self.prev_logging_level = logging.getLevelName(self.logger.level)
        MLP_helper.update_logger_level(logging_level)

        # Initialize and validate functions
        self._init_functions(activation_function_hidden, activation_function_output, loss_function, regularization, init_method, init_bias)

        # Initialize sizes 
        self._init_sizes(input_size, output_size, hidden_sizes)
        
        # create layers with self.top_models.append((model_instance, model_params, val_accuracy))
        self._init_layers(self.sizes, self.act_hidden, self.act_output, self.init, self.init_bias, self.bias_const, self.reg, self.reg_lambda)
        self.logger.debug(f"MLP intialized\t--\tlayers : {self.sizes}")

    def activation_function_hidden_f(self, x, eps = None):
        if eps:
            return self.act_hidden(x, eps)
        return self.act_hidden(x)
    
    def activation_function_output_f(self, x, eps=None):
        if eps:
            return self.act_output(x, eps)
        return self.act_output(x)
    
    def loss_function_f(self, y_true, y_pred, eps=None):
        if self.loss == mlp_func.CROSS_ENTROPY and self.act_output == mlp_func.SOFTMAX:
            y_true = self._ensure_ohe(y_true)
        
        if eps:
            return self.loss(y_true, y_pred, eps)
        return self.loss(y_true, y_pred)

    def activation_function_hidden_derivative_f(self, x, eps = None):
        if eps:
            return mlp_func.derv(self.act_hidden)(x, eps)
        return mlp_func.derv(self.act_hidden)(x)
    
    def activation_function_output_derivative_f(self, x, eps = None):       
        # Should never be called for softmax, derv not implemented! 
        if eps:
            return mlp_func.derv(self.act_output)(x, eps)
        
        return mlp_func.derv(self.act_output)(x)
    
    def loss_function_derivative_f(self, y_true, y_pred, eps = None):
        if self.loss == mlp_func.CROSS_ENTROPY and self.act_output == mlp_func.SOFTMAX:
            y_true = self._ensure_ohe(y_true)

        if eps:
            return mlp_func.derv(self.loss)(y_true, y_pred, eps)
        return mlp_func.derv(self.loss)(y_true, y_pred)
    
        
    def forward(self, input_batch):
        """
        Performs forwards pass on the network

        Args:
            input_batch: Input batch to work on, (batch_size, feature_size)
        
        Returns:
            final layer output
        """
        self.input = input_batch
        self.x_values = [input_batch]  # Store the initial input

        for layer in self.layers:
            x = layer.forward(self.x_values[-1])
            self.x_values.append(x)

        # Set predicted labels
        self.y_pred = self.x_values[-1]
        return self.x_values[-1]


    def backward(self, y_true_batch, y_pred_batch, learning_rate):
        gradients = []

        cur_error = self._calculate_output_error(y_true_batch, y_pred_batch)
        for i in reversed(range(len(self.layers))):
            grad_w, grad_b, cur_error = self.layers[i].calculate_gradient(cur_error)
            gradients.append({"grad_w":grad_w, "grad_b":grad_b})
        gradients.reverse()
        return gradients
        
    # Could be issues with this? what if softmax output and mse loss?
    def _calculate_output_error(self, y_true_batch, y_pred_batch):
        if self.act_output == mlp_func.SOFTMAX and self.loss == mlp_func.CROSS_ENTROPY:
            output_error = y_pred_batch - self._ensure_ohe(y_true_batch)
            
        elif y_pred_batch.shape != y_true_batch.shape:
            raise ValueError(f"y_pred and y_true shape mismatch\tpred:{y_pred_batch.shape} true:{y_true_batch.shape}")
        
        else:
            output_error = self.loss_function_derivative_f(y_true_batch, y_pred_batch)
        
        return output_error
    
    def _ensure_ohe(self, labels):
        num_classes = self.output_size

        if labels.ndim > 1:
            labels = labels.flatten()
            
        if labels.dtype != int:
            labels = labels.astype(int)
            
        one_hot_encoded = np.zeros((labels.size, num_classes))
        
        # Subtract 1 from labels to convert labels from 1-based to 0-based indexing
        labels_zero_based = labels - 1
        
        # Use np.arange to generate row indices and labels_zero_based for column indices
        one_hot_encoded[np.arange(labels.size), labels_zero_based] = 1

        assert self._test_ohe(one_hot_encoded, labels)
        
        return one_hot_encoded
    
    def _test_ohe(self, ohe, labels):
        return np.sum(np.argmax(ohe, axis=1) + 1 == labels) == labels.shape[0]
    
    def _get_learning_rate(self, learning_rate, epoch):
        """
        This function implements a linear learning rate decay scheduler.

        Args:
            epoch: Current epoch number (int).
            initial_lr: Initial learning rate (float).
            decay_rate: Decay rate per epoch (float).

        Returns:
            Learning rate for the current epoch (float).
        """
        if self.decay_rate == None:
            return learning_rate
        return learning_rate * (1 - self.decay_rate * epoch)


    def train(
            self, input, targets, learning_rate, batch_size = 1, 
            input_val = None, target_val = None, validate_n_epochs = 1, 
            accumulation_steps = 5,  early_stopping=True, patience=10,
            acc_threshold = 0.98, loss_change_threshold = 0.001):
        """
        Trains the MLP on the provided dataset using mini-batch gradient descent.

        Args:
            X: Input data, numpy array of shape (num_samples, num_features).
            y: Labels, numpy array of shape (num_samples,) or (num_samples, num_outputs).
            batch_size: Size of each mini-batch. Defaults to one
            learning_rate: Learning rate for the gradient descent update rule.
        """
        n_samples = input.shape[0]
        n_batches = int(np.ceil(n_samples / batch_size))
        accumulated_gradients = [[] for _ in range(len(self.layers))]

        # early stopping 
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.epochs):
            if self.stop_training:
                break
            # Decaying learning rate
            learning_rate = self._get_learning_rate(learning_rate, epoch)
            
            # Shuffle data at beginning of each epoch (apperently helps)
            indices = np.arange(n_samples)     
            np.random.shuffle(indices)
            X_shuffled = input[indices]
            y_shuffled = targets[indices] 

            # Train on batches
            predictions = []
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, n_samples)
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]

                # Accumulate gradients
                batch_gradients, prediction_tuple = self._train_on_batch(X_batch, y_batch, learning_rate)
                predictions.append(prediction_tuple)
                for layer_idx in range(len(batch_gradients)):
                    accumulated_gradients[layer_idx].append(batch_gradients[layer_idx])
                
                if (i + 1) % accumulation_steps == 0:
                    # Update weights using accumulated gradients
                    for layer_idx in range(len(self.layers)):
                        total_grad_w = np.sum([grad["grad_w"] for grad in accumulated_gradients[layer_idx]], axis=0)
                        total_grad_b = np.sum([grad["grad_b"] for grad in accumulated_gradients[layer_idx]], axis=0)
                        
                        self.layers[layer_idx].update_weights(total_grad_w, total_grad_b, learning_rate)
                        accumulated_gradients[layer_idx] = []  # Reset accumulated gradients

            # Early Stopping Check (inneffecient/unclean after adding accuracy saves)
            if input_val is not None and target_val is not None:
                val_acc = self.test(input=input_val, targets=target_val, batch_size=batch_size, ohe=False)
                self.validation_accuracies.append(val_acc)
                
                if epoch % validate_n_epochs == 0:
                    self._validate(input_val, target_val, input_val_acc=val_acc)
                    self._early_stop(patience, acc_threshold, loss_change_threshold)

            # Store training accuracy and log
            train_acc = sum([x[0] for x in predictions]) / sum([x[1] for x in predictions])
            self.training_accuracies.append(train_acc)

            # Store validation accuracy
            self.logger.debug(f"Epoch {epoch}/{self.epochs} completed.")

    def _train_on_batch(self, x_batch, y_true_batch, learning_rate):
        y_pred_batch = self.forward(x_batch)
        gradients = self.backward(y_true_batch, y_pred_batch, learning_rate)
        train_acc, correct_preds, total_preds  = self.test_batch(
            input_batch=None, target_batch=y_true_batch, ohe=False, predictions_input=y_pred_batch)
        return gradients, (correct_preds, total_preds)


    def _validate(self, input_val, target_val, input_val_acc = None):
        targets_pred_val = self.predict(input_val)
        val_loss = self.loss_function_f(target_val, targets_pred_val)
        val_acc = self.test(input = input_val, targets=target_val) if input_val_acc is None else input_val_acc
        self.validation.append((val_loss, val_acc))

        self.logger.debug(f"validation loss: {val_loss} \tvalidation accuracy : {val_acc}")

    def _early_stop(self, patience, accuracy, threshold):
        # check for increasing validation loss trend, indicating potential overfitting
        if len(self.validation) > patience:
            is_increasing = True  # assume the trend is increasing to start
            for i in range(len(self.validation) - patience, len(self.validation) - 1):
            # if the current loss is greater or equal to the next loss, trend is not strictly increasing
                if self.validation[i][0] >= self.validation[i + 1][0]:
                    is_increasing = False
                    break
            if is_increasing:
                self.logger.debug("Validation loss has been increasing for a while, indicating the model has started to overfit, stopping the training.")
                self.stop_training = True
                return
        # stop training if the target accuracy has been reached on the validation set
        if self.validation[-1][1] >= accuracy:
            self.logger.debug(f"The target accuracy on a validation set of {accuracy * 100:.2f}% has been achieved, stopping the training.")
            self.stop_training = True
            return
        # stop training if the reduction in validation loss falls below a specified threshold
        if len(self.validation) >= patience:
            flag = True
            for i in range(patience - 1):
                if self.validation[-2 - i][0] - self.validation[-1 - i][0] > threshold: # previous loss is bigger than the next one by more than threshold
                    flag = False
                    break
            if flag:
                self.logger.debug(f"The change in validation loss stopped exceeding {threshold}, stopping training.")
                self.stop_training = True

            
            
    def predict(self, input_batch):
        """Predicts outputs for a batch of input samples."""
        self.predictions = [input_batch]
        for layer in self.layers:
            x = layer.forward(self.predictions[-1])
            self.predictions.append(x) 
            
        return self.predictions[-1]
    
    def test(self, input, targets, batch_size = 10, ohe = False):
        n_samples = targets.shape[0]
        n_batches = int(np.ceil(n_samples / batch_size))

        total_correct_pred = 0
        total_pred = 0
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, n_samples)
            input_batch = input[start_idx:end_idx]
            target_batch = targets[start_idx:end_idx]

            batch_accuracy, correct_pred, n_pred = self.test_batch(input_batch, target_batch, ohe)
            total_correct_pred += correct_pred
            total_pred += n_pred
        
        accuracy = total_correct_pred / total_pred
        self.logger.debug(f"Accuracy : {accuracy}\t|\t{total_correct_pred} / {total_pred}")
        return accuracy

    def test_batch(self, input_batch, target_batch, ohe=False, predictions_input=None):
        if predictions_input is None:
            predictions = self.predict(input_batch)
        else:
            predictions = predictions_input

        if ohe and self.act_output == mlp_func.SOFTMAX:
            predicted_labels = np.argmax(predictions, axis=1) + 1
            target_labels = np.argmax(target_batch, axis=1) + 1

        elif self.act_output == mlp_func.SOFTMAX:
            # assumes target labels are not ohe
            predicted_labels = np.argmax(predictions, axis=1) + 1
            target_labels = target_batch
        else:
            predicted_labels = predictions
            
            
        predicted_labels = np.squeeze(predicted_labels)  # Reduce to 1D if not already
        target_labels = np.squeeze(target_labels)  # Ensure target_labels is also 1D
            
        correct_predictions = np.sum(predicted_labels == target_labels)
        
        self.logger.debug(f"correct preds : {correct_predictions} / {target_batch.shape[0]}")
        
        accuracy = correct_predictions / target_batch.shape[0]
        return accuracy, correct_predictions, target_batch.shape[0]


    def visualize(self):
        """
        Prints the current weights, biases, and intermediate outputs of the network.
        """
        for i, layer in enumerate(self.layers):
            print(f"Layer {i}:")
            layer.info()
            layer.print_values()
            print("\n\n")

    def get_accuracies(self):
        """Returns accuracies (train_accs, val_accs). accs are saved every n epochs (variable)"""
        return self.training_accuracies, self.validation_accuracies
    def get_epochs(self):
        return self.epochs
    def set_epochs(self, epochs):
        self.epochs = epochs

    def reset_logger(self):
        MLP_helper.update_logger_level(self.prev_logging_level)

    def reset(self):
        self.x_values = []
        self.validation = []
        self.training_accuracies = []
        self.validation_accuracies = []
        self.y_true = None
        self.y_pred = None
        self.stop_training = False
        self.predictions = []
        for i in range(len(self.layers)):
            self.layers[i].reset()