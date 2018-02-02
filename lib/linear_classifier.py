from __future__ import print_function

import numpy as np
from lib.linear_svm import *
import math

# --------------------------------------------------------------------------------
#  Linear Classifier (parent class for LinearSVM)
# --------------------------------------------------------------------------------
class LinearClassifier(object):

    def __init__(self):
        self.W = None
                
    def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_epochs=5, batch_size=200, max_iter = -1, vectorized = True, verbose=False):
        """
        Train this linear classifier using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
          training samples each of dimension D.
        - y: A numpy array of shape (N,) containing training labels; y[i] = c
          means that X[i] has label 0 <= c < C for C classes.
        - learning_rate: (float) learning rate for optimization.
        - reg: (float) regularization strength.
        - num_epochs: (integer) number of epochs to take when optimizing
        - batch_size: (integer) number of training examples to use at each step.
        - verbose: (boolean) If true, print progress during optimization.

        Outputs:
        A list containing the value of the loss function at each training iteration.
        """
        num_train, dim = X.shape
        num_classes = np.max(y) + 1 # assume y takes values 0...K-1 where K is number of classes
        
        # Initialization
        if self.W is None:
            self.W = 0.001 * np.random.randn(dim, num_classes)

        loss_history = []                                           # to store the loss value over all iterations (this will be used to plot figures later)
        iters_per_epoch = int(math.ceil(num_train/batch_size))      # compute number of iterations over each epoch
        num_iter = 0                                                # keeps track of number of batch iterations
        
        # Train for num_epochs time. When training completes one epoch, it has seen all the samples in the whole dataset once.
        for epoch in range(num_epochs):
        
            X_batch = None
            y_batch = None
                       
            #########################################################################
            # TODO:                                                                 #
            # Create a shuffled index. The data will be provided following this     #
            # sequence                                                              #
            #########################################################################
            #  Your code here                                                       #
            #########################################################################
            #                  END OF YOUR CODE                                     #
            #########################################################################

            for it, idx in enumerate(range(0, num_train, batch_size)):
                
                #########################################################################
                # TODO:                                                                 #
                # Get the current batch                                                 #
                #########################################################################
                #  Your code here                                                       #
                # indices = ...
                X_batch = X[idx:idx+batch_size]
                y_batch = y[idx:idx+batch_size]
                #########################################################################
                #                  END OF YOUR CODE                                     #
                #########################################################################

                # Evaluate loss and gradient
                loss, grad = self.loss(X_batch, y_batch, reg, vectorized)
                loss_history.append(loss)
                
                #########################################################################
                # TODO:                                                                 #
                # Perform parameter update                                              #
                #########################################################################
                
                self.W = self.W - learning_rate*grad                                                 
                #########################################################################
                #                  END OF YOUR CODE                                     #
                #########################################################################

                # print result
                num_iter += 1
                if verbose and ((it % 100 == 0) or (it == iters_per_epoch - 1)):
                    print('epoch {:d} / {:d}, iteration {:d} / {:d}: loss {:f}'.format(epoch, num_epochs, it, iters_per_epoch - 1, loss))

                # stop if reach maximum number of batch iteration
                if max_iter > 0 and num_iter >= max_iter:
                    break
                    
        if verbose:
            print('Total number of iterations: {:d}'.format(num_iter))
        
        return loss_history

        
    def predict(self, X):
        """
        Use the trained weights of this linear classifier to predict labels for
        data points.

        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
          training samples each of dimension D.

        Returns:
        - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
          array of length N, and each element is an integer giving the predicted
          class.
        """
        y_pred = np.zeros(X.shape[0])
        ###########################################################################
        # TODO:                                                                   #
        # Implement this method. Store the predicted labels in y_pred.            #
        ###########################################################################
        
        y_pred = np.argmax(np.dot(X, self.W), axis=1)
        
        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return y_pred
  
    def loss(self, X_batch, y_batch, reg, vectorized = True):

        """
        Compute the loss function and its derivative. 
        Subclasses will override this.

        Inputs:
        - X_batch: A numpy array of shape (N, D) containing a minibatch of N
          data points; each point has dimension D.
        - y_batch: A numpy array of shape (N,) containing labels for the minibatch.
        - reg: (float) regularization strength.

        Returns: A tuple containing:
        - loss as a single float
        - gradient with respect to self.W; an array of the same shape as W
        """
        pass

# --------------------------------------------------------------------------------
#  LinearSVM  classifier(inherits from LinearClassifier)
# --------------------------------------------------------------------------------
class LinearSVM(LinearClassifier):
  """ A subclass that uses the Multiclass SVM loss function """

  def loss(self, X_batch, y_batch, reg, vectorized = True):
  
    if vectorized:
        loss        = svm_loss_vectorized (self.W, X_batch, y_batch, reg)
        gradient    = svm_gradient_vectorized (self.W, X_batch, y_batch, reg)
    else:
        loss        = svm_loss_naive(self.W, X_batch, y_batch, reg)
        gradient    = svm_gradient_naive (self.W, X_batch, y_batch, reg)
    
    return loss, gradient
    