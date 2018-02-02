import numpy as np
from random import shuffle

# -----------------------------------------------------------------------------------
# Function to compute the loss (naive version)
# -----------------------------------------------------------------------------------
def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
        - W: A numpy array of shape (D, C) containing weights.
        - X: A numpy array of shape (N, D) containing a minibatch of data.
        - y: A numpy array of shape (N,) containing training labels; y[i] = c means 
             that X[i] has label c, where 0 <= c < C.
        - reg: (float) regularization strength

    Returns a tuple of:
        - gradient with respect to weights W; an array of same shape as W
    
    """   
    loss = 0                   #
    num_classes = W.shape[1]   # number of classes
    num_train = X.shape[0]     # number of training samples
    
    ###########################################################
    # TODO: Implement the non-vectorized version for the multi-
    # class SVM
    ###########################################################
    for i in range(num_train):
        # compute the scores
        score_i = np.dot(X[i], W)
        loss_i = 0

        for j in range(num_classes):
            # compute the loss incurred by class j
            if j == y[i]:
                continue
            loss_i += max(0, score_i[j] - score_i[y[i]] + 1)
            
        
        loss += loss_i
            
    # add the regularization term
    loss = loss/num_train + reg*np.sum(W**2)
       
    ###########################################################
    # End of code
    ###########################################################
     
    return loss
    
# -----------------------------------------------------------------------------------
# Function to compute the loss (vectorized version)
# -----------------------------------------------------------------------------------
def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    pass
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

#     return loss

# -----------------------------------------------------------------------------------
# Function to compute the gradient (naive version)
# -----------------------------------------------------------------------------------
def svm_gradient_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
        - W: A numpy array of shape (D, C) containing weights.
        - X: A numpy array of shape (N, D) containing a minibatch of data.
        - y: A numpy array of shape (N,) containing training labels; y[i] = c means 
             that X[i] has label c, where 0 <= c < C.
        - reg: (float) regularization strength

    Returns a tuple of:
        - loss as single float
        - gradient with respect to weights W; an array of same shape as W
    
    """
    num_classes = W.shape[1]    # C  
    num_train = X.shape[0]      # N
         
    # initialize the gradient as zero
    dW = np.zeros(W.shape)      # (shape = D x C)
        
    # compute the gradient    
    for i in range(num_train):

        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
            
        dscale = np.zeros(num_classes)
        for j in range(num_classes):
            if j == y[i]:
                continue    
                    
            margin = scores[j] - correct_class_score + 1      
            if margin > 0:
                dscale[j] = 1           # for the weights of other labels (w_j), scale x_i if it does not meet the margin, else set to 0
                            
        dscale[y[i]] = -np.sum(dscale)  # for the weights of the true label (w_yi), scale x_i by number of classes that do not meet the margin criteria
        dW += np.outer (X[i], dscale)   # gradient (shape = D x C) = outer product of x_i (shape = D) and dscale (shape = C) 
       
    dW /= num_train         
    dW += 2*reg*W
        
    return dW
    
# -----------------------------------------------------------------------------------
# Function to compute the gradient (vectorized version)
# -----------------------------------------------------------------------------------
def svm_gradient_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_gradient_naive.
    """
    num_samples = X.shape[0]         
    scores = np.dot(X, W)                                           # shape = (#samples, #classes)

    scores_yi = scores[np.arange(num_samples), y].reshape(-1, 1)    # shape = (#samples, 1)
    margin = scores - scores_yi + 1.0                               # shape = (#samples, #classes)
        
    ds = margin > 0                                                 # shape = (#samples, #classes), binary
    ds = ds.astype(np.float64)                                      # convert to float

    ds[np.arange(num_samples), y] = -ds.sum(axis = 1)               # scale x_i by by number of classes that do not meet the margin criteria
        
    dW = np.dot(X.T, ds) / num_samples + 2*reg*W                    # shape = (#dimension, # classes)
    
    return dW