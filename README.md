# Machine Learning Algorithms - Implementation Comparison
Compares the implementation of the Pocket Algorithm and Linear Regression using Python and NumPy with their counterparts in scikit-learn. It would demonstrate the fundamental understanding of the algorithms by coding them from scratch and then evaluate the performance against the established modules in scikit-learn.

## Pocket Algorithm with NumPy vs SciKit
The task requires the creation of four key functions: fit_Perceptron for training the algorithm, errorPer for calculating the error rate, pred for making predictions, and confMatrix for evaluating the performance through a confusion matrix. These functions are designed to work together to identify the optimal linear boundary that separates two classes of data points, demonstrating the fundamental concepts of machine learning and algorithmic implementation.

 The test_SciKit function takes training and test datasets, along with their respective labels, to train a Perceptron model and output a confusion matrix, providing a clear comparison of model performance against a custom NumPy implementation. This segment underlines the practical application of scikit-learn's Perceptron and metrics modules.

## Linear Regression with NumPy vs SciKit
The Linear Regression implementation in Python using NumPy, focusing on solving the least squares problem. It will describe the fit_LinRegr function for computing regression coefficients, the mse function for mean squared error calculation, and the pred function for making predictions. The document will highlight the use of NumPy functions like shape, hstack, ones, dot, and linalg.inv and will also reference the subsetFn utility for cases with non-full-rank matrices, providing a comprehensive guide to linear regression analysis.

The implementation of Linear Regression using scikit-learn in Python will detail the test_SciKit function, which takes in training and test datasets to train a regression model, and computes mean squared error using scikit-learn's metrics library.
