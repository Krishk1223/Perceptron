import numpy as np

class Perceptron:
    def __init__(self, input_size, alpha=0.01, decay_rate = 0, epochs = 100):
        self.alpha = alpha # learning rate for model
        self.initial_alpha = alpha #initial alpha useful for resetting the learning rate
        self.epochs = epochs #number of iterations to run the model for
        self.weights = np.random.randn(input_size+1)*0.01 #randomised weights to break symmetry
        self.decay_rate = decay_rate #optional decay rate for exponential decay
        self.training_history = {
            "epoch_error": [],
            "accuracies": []
        }
    def decay_learning(self, epoch):
        """
        Calculates value of the learning rate via exponential decay rate at the given epoch if the decay rate is non zero.

        Parameters:
        -----------
        epoch : The iteration being run at that time.
            type: int
        """
        if self.decay_rate > 0: #to make sure if decay rate is not selected this will perform normally
            return self.initial_alpha * np.exp(-1*(self.decay_rate)*epoch)
        else:
            return self.initial_alpha

    def step_function(self,product):
        """
        Binary classification function with x as the weighted sum of inputs which is the dot product of w and x (w*x).
        Returns either 1 or 0 based on the value of the dot product.

        Parameters:
        -----------
        product : Dot product of Input vector and Weights.
            type: float
        """
        return 1 if product>=0 else 0 
    
    def predict(self, x):
        """
        makes a prediction for the x input vector (a vector containing all inputs). Return step up activation value
        for the dot product of weight vector along with the biases with the input vector. Essentially finding w*x.
        Returns the binary classification of either 1 or 0 based on the dot product of input vector and weights.
        Parameters:
        -----------
        x : Input vector
            type: numpy.ndarray
        """
        x = np.insert(x,0,1) #adds a bias term of 1 at the start to account for w0 (creating input as [1,x1,x2...])
        return self.step_function(np.dot(self.weights, x)) #returns the step up activation value

    def fit(self, X_train, Y_train, X_validation = None, Y_validation = None, patience = 5):
        """
         This function trains the perceptron model on the inputs and their respective labels.
         Can also use a validation data set to further validate the training data and uses early stoppage to prevent overfitting 
         if the validation data is provided.

         Parameters:
         -----------
         X_train: Training input vectors 
            type: numpy.ndarray
         Y_train: Outputs to corresponding training input vectors. Training data labels.
            type: numpy.ndarray
         X_validation: Validation input vectors
            type: numpy.ndarray
         Y_validation: Outputs to corresponding validation input vectors. Validation data labels.
            type: numpy.ndarray
        """

        #error handling (basic)
        if X_train.shape[0] != Y_train.shape[0]:
            raise ValueError("Training data labels and inputs are unequal")
        if not np.all(np.isin(Y_train, [0,1])):
            raise ValueError("Training data labels should only be 0 or 1")

        best_validation_accuracy = 0 #track best accuracy encountered
        stopping_patience = patience #how many epochs of no improvement before early stopping
        no_improvement_count = 0 #counting how many epochs of no improvement
    
        for i in range(self.epochs):
            current_alpha = self.decay_learning(i) #applies exponential decay if non zero otherwise runs the same.
            epoch_total_error = 0 #tracks total errors in prediction the epoch and is reset each epoch.

            for xi, y_actual in zip(X_train, Y_train): # iterates over each pair of training input and actual label.
                xi = np.insert(xi, 0, 1) #adds bias term to the training data value
                y_prediction = self.step_function(np.dot(self.weights, xi)) #does the binary classification for label
                difference = y_actual - y_prediction
                updated = current_alpha * (difference)
                epoch_total_error += abs(difference) #tracks total error 
                self.weights += updated*xi #updates weight values based on difference and learning rate

            #Validation Phase if values are not provided then it does not make a difference to data
            if X_validation is not None and Y_validation is not None:
                #error checks (basic)
                if X_validation.shape[0] != Y_validation.shape[0]:
                    raise ValueError("Validation data labels and inputs are unequal")
                if not np.all(np.isin(Y_validation, [0,1])):
                    raise ValueError("Validation data labels should only be 0 or 1")

                validation_accuracy = self.accuracy_check(X_validation, Y_validation) #calculate accuracy on validation set
                self.training_history["accuracies"].append(validation_accuracy) #appends validation accuracy

                #check how good current model is compared to best model
                if validation_accuracy > best_validation_accuracy:
                    best_validation_accuracy = validation_accuracy
                    no_improvement_count = 0 #reset the no improvement counter as model has improved if this is executed
                else:
                    no_improvement_count += 1 #no improvement from previous model so the counter increases by 1
                #Early stoppage:
                if no_improvement_count >= stopping_patience: #if improvement count goes above what our declared patience is then
                    print(f"Early stopping due to no further improvement. Stopped at epoch {i}") #stop improvements
                    #this prevents overfitting and wasting computation time
                    break 

            self.training_history["epoch_error"].append(epoch_total_error) #storing the differences between predicted and actual values
        self.alpha = self.initial_alpha #resetting the epoch rate to allow fit to be called more than once.

    def accuracy_check(self, X_test, Y_test):
        """
         Checks the accuracy of the tested data predictions to their output values returns a floating value of the accuracy between
         0 which is completely inaccurate and 1 which is always accurate.
         Parameters:
         -----------
         X_test: Test data input vectors 
            type: numpy.ndarray
         Y_test: Outputs to corresponding test data input vectors. Test data labels.
            type: numpy.ndarray

        """
        #error checks (basic)
        if X_test.shape[0] != Y_test.shape[0]:
            raise ValueError("Test data labels and inputs are unequal")
        if not np.all(np.isin(Y_test, [0,1])):
            raise ValueError("Test data labels should only be 0 or 1")
    
        predictions = np.array([self.predict(i) for i in X_test]) #updates predicted values to a numpy array
        accuracy = np.mean(predictions == Y_test) #compare and find mean of predictions being accurate between 0 and 1
        return accuracy











