import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

from numpy.linalg import inv
from sklearn.linear_model import LinearRegression

from numpy.linalg import pinv


def fit_LinRegr(X_train, y_train):
  
  # Add a column of ones to X_train for the bias term
  X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
  
  # Applying the minimizer to find the best-fit weights
  # w = (X^T * X)^-1 * X^T * y
  w = pinv(X_train.T.dot(X_train)).dot(X_train.T).dot(y_train)

  return w

def mse(X, y, w):
  
  # Predict the values using the weights w
  predictions = pred(X, w)
  # Compute the mean squared error between the predicted and actual values
  avgError = np.mean((predictions - y) ** 2)

  return avgError

def pred(X_train, w):
  
  # Add a column of ones to X_train for the bias term
  X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
  # Compute the predictions: y_pred = X * w
  predictions = X_train.dot(w)

  return predictions

def test_SciKit(X_train, X_test, Y_train, Y_test):
  
  # Initialize the linear regression model
  LR = LinearRegression()

  # Fit the model on the training data
  LR.fit(X_train, Y_train)

  # Predict the target values for the test set
  Y_pred = LR.predict(X_test)

  # Calculate the mean squared error using the true and predicted values
  error = mean_squared_error(Y_test, Y_pred)

  return error

def subtestFn():
    # This function tests if your solution is robust against singular matrix

    # X_train has two perfectly correlated features
    X_train = np.asarray([[1, 2], [2, 4], [3, 6], [4, 8]])
    y_train = np.asarray([1,2,3,4])
    
    try:
      w=fit_LinRegr(X_train, y_train)
      print ("weights: ", w)
      print ("NO ERROR")
    except:
      print ("ERROR")

def testFn_Part2():
    X_train, y_train = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X_train,y_train,test_size=0.2)
    
    w=fit_LinRegr(X_train, y_train)
    
    #Testing Part 2a
    e=mse(X_test,y_test,w)
    
    #Testing Part 2b
    scikit=test_SciKit(X_train, X_test, y_train, y_test)
    
    print("Mean squared error from Part 2a is ", e)
    print("Mean squared error from Part 2b is ", scikit)

print ('------------------subtestFn----------------------')
subtestFn()

print ('------------------testFn_Part2-------------------')
testFn_Part2()
