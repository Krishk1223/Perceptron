# Perceptron

This repository contains an implementation of a Perceptron model in Python.

## Description

The Perceptron is a type of artificial neural network and one of the simplest forms of a neural network model. It is used for binary classification tasks. 
This specific implementation of the perceptron model currently only focuses on sample by sample based learning approach making it slightly less efficient for larger datasets.
The model has some additional features that allow for a train/validation/test split of dataset and an exponential decay based learning rate if needed.
Early stoppage is also implemented to prevent overfitting, the user can choose their patience for early stoppage as how they see fit (default is 5 epochs of no improvement).


## Files

- `perceptron.py`: Contains the implementation of the Perceptron model.
- `pyproject.toml`: Lists the dependencies required to run the project.
- `breast_cancer_prediction.py`: Uses perceptron model and scikit learn's breast cancer dataset to predict patients with breast cancer. Then lists out the model accuracy as a percentage.

