import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import confusion_matrix 

def pred(X_i, w):
  # Function to make predictions using the perceptron model
  class_label = 1 if np.dot(X_i, w) > 0 else -1
  
  return class_label

def errorPer(X_train, y_train, w):
  
  # Add a column of ones to X_train for the bias term if not already included
  if X_train.shape[1] != w.shape[0]:
    X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
    
  # Initialize the error count
  errors = 0
    
  # Loop through all training samples
  for i in range(len(X_train)):
    # Predict class label for each sample
    predicted = pred(X_train[i], w)
        
    # Check if prediction is incorrect
    if predicted != y_train[i]:
      errors += 1
            
  # Calculate the average number of misclassifications
  avgError = errors / len(X_train)
    
  return avgError
  

def fit_Perceptron(X_train, y_train):
  
  max_epochs = 5000
  # Add a column of ones to X_train for the bias term
  X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
    
  # Initialize weights to zero
  w = np.zeros(X_train.shape[1])
  pocket_w = np.copy(w)
  pocket_error = errorPer(X_train, y_train, w)
    
  # Main loop to fit the data
  for j in range(max_epochs):
    for i in range(X_train.shape[0]):
      # Check if the current sample is misclassified
      if y_train[i] * np.dot(X_train[i], w) <= 0:
        # Update weights
        w += y_train[i] * X_train[i]
        
        # Calculate error with the updated weights
        current_error = errorPer(X_train, y_train, w)
        # If the updated weights have a lower error, update the pocket weights
        if current_error < pocket_error:
          pocket_w = np.copy(w)
          pocket_error = current_error
  # Return the best weights from the pocket
  return pocket_w

def confMatrix(X_train, y_train, w):
  
  # Initialize the confusion matrix
  confusion_matrix = np.array([[0, 0], [0, 0]])
    
  # Add a column of ones to X_train for the bias term if not already included
  if X_train.shape[1] != w.shape[0]:
    X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
    
  # Loop through all training samples
  for i in range(len(X_train)):
    # Predict class label for each sample
    prediction = pred(X_train[i], w)
        
    # Update confusion matrix based on predictions
    if prediction == 1 and y_train[i] == 1:
      confusion_matrix[1][1] += 1  # True Positive
    elif prediction == 1 and y_train[i] == -1:
      confusion_matrix[0][1] += 1  # False Positive
    elif prediction == -1 and y_train[i] == -1:
      confusion_matrix[0][0] += 1  # True Negative
    elif prediction == -1 and y_train[i] == 1:
      confusion_matrix[1][0] += 1  # False Negative
    
  return confusion_matrix

def test_SciKit(X_train, X_test, Y_train, Y_test):
  
  # Initialize the Perceptron classifier
  clf = Perceptron(tol=1e-3, random_state=0)
    
  # Fit the Perceptron classifier on the training data
  clf.fit(X_train, Y_train)
    
  # Make predictions on the test data
  Y_pred = clf.predict(X_test)
    
  # Compute the confusion matrix between the true test labels and predictions
  conf_matrix = confusion_matrix(Y_test, Y_pred, labels=np.unique(Y_train))
    
  return conf_matrix
  

def test_Part1():
    from sklearn.datasets import load_iris
    X_train, y_train = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X_train[50:],y_train[50:],test_size=0.2)

    #Set the labels to +1 and -1
    y_train[y_train == 1] = 1
    y_train[y_train != 1] = -1
    y_test[y_test == 1] = 1
    y_test[y_test != 1] = -1

    #Pocket algorithm using Numpy
    w=fit_perceptron(X_train,y_train)
    cM=confMatrix(X_test,y_test,w)

    #Pocket algorithm using scikit-learn
    sciKit=test_SciKit(X_train, X_test, y_train, y_test)
    
    #Print the result
    print ('--------------Test Result-------------------')
    print("Confusion Matrix is from Part 1a is: ",cM)
    print("Confusion Matrix from Part 1b is:",sciKit)
    

test_Part1()
