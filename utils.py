import numpy as np

# Helper function to calculate the sigmoid
def sigmoid(z):
    sig = 1 / (1 + np.exp(-z))
    
    return sig

# Helper function to initialize weights and bias with zero
def initialize_parameters(dimension):
    weights = np.zeros((dimension, 1))
    bias = 0
    
    return weights, bias

# Helper function to do a single forward pass
def forward_propagate(X, Y, weights, bias):
    # Get the number of training examples
    m = X.shape[1]
    
    # Calculate the z
    z = np.dot(weights.T, X) + bias
    
    # Calculate activation A using sigmoid activation
    A = sigmoid(z)
    
    # Calculate for the cost
    cost = (-1/m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
    
    cost = np.squeeze(cost)
    
    return A, cost

# Helper function to do a single backward pass
def backward_propagate(X, Y, A):
    # Calculate for the derivatives
    m = X.shape[1]
    dz = A - Y
    dw = (1/m) * np.dot(X, dz.T)
    db = (1/m) * np.sum(dz)
    
    return dw, db

# Helper function to optimize weights and bias using gradient descent algorithm
def gradient_descent(X, Y, weights, bias, iterations, learning_rate, print_cost=False):
    costs = []
    
    for i in range(iterations):
        # Compute for cost and gradient using propagate
        A, cost = forward_propagate(X, Y, weights, bias)
        dw, db = backward_propagate(X, Y, A)
        
        # Update weights and bias parameters
        weights = weights - learning_rate * dw
        bias = bias - learning_rate * db
        
        # Record cost
        if i % 100 == 0:
            costs.append(cost)
        
        # Print the cost every 100 iterations
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
            
    return weights, bias, dw, db, costs

def predict(X, weights, bias):
    m = X.shape[1]
    Y_pred = np.zeros((1, m))
    
    A = sigmoid(np.dot(weights.T, X) + bias)
    
    for i in range(A.shape[1]):
        if A[0,i] <= 0.5:
            Y_pred[0,i] = 0
        else:
            Y_pred[0,i] = 1
    
    return Y_pred
