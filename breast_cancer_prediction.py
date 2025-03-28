import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from perceptron import Perceptron

#load data and assign to x and y:
breast_cancer = load_breast_cancer()
x = breast_cancer.data
y = breast_cancer.target 

#do training, validation, testing splits (70% training, 15% validation, 15% split):
X_train, X_temp, Y_train, Y_temp = train_test_split(x, y, test_size=0.3, random_state=42)
X_validation, X_test, Y_validation, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

#feature scaling (ensure normalisation for perceptron model to work):
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_validation = scaler.fit_transform(X_validation)
X_test = scaler.fit_transform(X_test)

#perceptron initialisation and training
perceptron = Perceptron(input_size = X_train.shape[1], epochs = 1000, alpha= 0.01, decay_rate=0.001, patience= 5)
perceptron.fit(X_train, Y_train, X_validation, Y_validation)

#prediction accuracy test
accuracy = perceptron.accuracy_check(X_test, Y_test)
percentage = accuracy*100
print(f"Perceptron accuracy in predicting breast cancer: {percentage:.2f} %")







