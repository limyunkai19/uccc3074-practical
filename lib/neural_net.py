from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import math

class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network. The net has an input dimension of
    N, a hidden layer dimension of H, and performs classification over C classes.
    We train the network with a softmax loss function and L2 regularization on the
    weight matrices. The network uses a ReLU nonlinearity after the first fully
    connected layer.
    
    In other words, the network has the following architecture:
    
    input - fully connected layer - ReLU - fully connected layer - softmax
    The outputs of the second fully-connected layer are the scores for each class.
    """
    
    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:
        
        W1: First layer weights; has shape (D, H)
        b1: First layer biases; has shape (H,)
        W2: Second layer weights; has shape (H, C)
        b2: Second layer biases; has shape (C)
        
        Inputs:
        - input_size: The dimension D of the input data.
        - hidden_size: The number of neurons H in the hidden layer.
        - output_size: The number of classes C.
        """
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
    
    def loss (self, X, y=None, reg=0.0):
        """
        Compute the loss and gradients for a two layer fully connected neural
        network.

        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
          an integer in the range 0 <= y[i] < C. This parameter is optional; if ep_it
          is not passed then we only return scores, and if ep_it is passed then we
          instead return the loss and gradients.
        - reg: Regularization strength.

        Returns:
        If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
        the score for class c on input X[i].

        If y is not None, instead return a tuple of:
        - loss: Loss (data loss and regularization loss) for this batch of training
          samples.
        - grads: Dictionary mapping parameter names to gradients of those parameters
          with respect to the loss function; has the same keys as self.params.
        """
        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape

        # Compute the forward pass
        scores = None
        #############################################################################
        # TODO (Exercise 1):                                                        #
        # Perform the forward pass, computing the class scores for the input.       #
        # Store the result in the scores variable, which should be an array of      #
        # shape (N, C).                                                             #
        #############################################################################
        z1 = X.dot(W1) + b1
        a1 = np.maximum(z1,0)
        scores = a1.dot(W2) + b2
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        
        # If the targets are not given then jump out, we're done
        if y is None:
            return scores

        # Compute the loss
        loss = None
        
        #############################################################################
        # TODO (Exercise 2):                                                        #
        # Finish the forward pass, and compute the loss. This should include        #
        # both the data loss and L2 regularization for W1 and W2. Store the result  #
        # in the variable loss, which should be a scalar. Use the Softmax           #
        # classifier loss.                                                          #
        #############################################################################       
        probs = np.exp(scores) / np.sum(np.exp(scores),axis=1,keepdims=True)
        loss = np.mean(-np.log(probs[range(N),y])) + reg*(np.sum(W1**2) + np.sum(W2**2))
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        # Backward pass: compute gradients
        grads = {}
        
        # compute the gradient on scores
        dscores = probs
        dscores[range(N),y] -= 1
        dscores /= N

        # backprop to W2 and b2
        grads['W2'] = np.dot(a1.T, dscores)
        grads['b2'] = np.sum(dscores, axis=0)
        
        # next backprop into hidden layer
        dhidden = np.dot(dscores, W2.T)
        
        # backprop the ReLU non-linearity
        dhidden[a1 <= 0] = 0
        
        # finally into W,b
        grads['W1'] = np.dot(X.T, dhidden)
        grads['b1'] = np.sum(dhidden, axis=0)

        # add regularization gradient contribution
        grads['W2'] += 2*reg * W2
        grads['W1'] += 2*reg * W1
        
        return loss, grads
        
    
    def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=5e-6, num_epochs = 20, max_iters = -1,
            batch_size=200, verbose=False):
        """
        Train this neural network using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) giving training data.
        - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
          X[i] has label c, where 0 <= c < C.
        - X_val: A numpy array of shape (N_val, D) giving validation data.
        - y_val: A numpy array of shape (N_val,) giving validation labels.
        - learning_rate: Scalar giving learning rate for optimization.
        - learning_rate_decay: Scalar giving factor used to decay the learning rate
          after each epoch.
        - reg: Scalar giving regularization strength.
        - num_epochs: Number of epochs to take when optimizing.
        - max_iters: Maximum number of iterations. Set to -1 to disable
        - batch_size: Number of training examples to use per step.
        - verbose: boolean; if true print progress during optimization.
        """
        num_train = X.shape[0]
  
        loss_history = []                                               # used to store results over iterations
        train_acc_history = []
        val_acc_history = []
        
        it = 0                                                          # keep track of number of iterations
        num_iters_per_epoch = int(math.ceil(num_train/batch_size))      # number of iterations over each epoch
        if max_iters < 0:
            max_iters = num_epochs * num_iters_per_epoch                # maximum number of iterations
        else:
            num_epochs = min(num_epochs, int(math.ceil(max_iters/num_iters_per_epoch)))
                
        # Use SGD to optimize the parameters in self.model       
        for epoch in range(num_epochs):
        
            shuffled_index = np.arange(num_train)  # shuffle training set at the begining of each epoch
            np.random.shuffle(shuffled_index)
             
            for ep_it, ep_idx in enumerate(range(0, num_train, batch_size)):
                
                # get batch
                indices = shuffled_index[ep_idx:ep_idx+batch_size]
                #indices = np.random.choice(num_train, batch_size)
                X_batch = X[indices]
                y_batch = y[indices]    

                # Compute loss and gradients using the current minibatch
                loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
                loss_history.append(loss)

                #########################################################################
                # TODO (Exercise 4):                                                    #
                # Use the gradients in the grads dictionary to update the               #
                # parameters of the network (stored in the dictionary self.params)      #
                # using stochastic gradient descent. You'll need to use the gradients   #
                # stored in the grads dictionary defined above.                         #
                #########################################################################               
                self.params['W1'] -= learning_rate * grads['W1']
                self.params['b1'] -= learning_rate * grads['b1']
                self.params['W2'] -= learning_rate * grads['W2']
                self.params['b2'] -= learning_rate * grads['b2']
                #########################################################################
                #                             END OF YOUR CODE                          #
                #########################################################################                

                # End of current epoch or last iteration
                if ep_it == num_iters_per_epoch - 1 or it == max_iters -1:
                    
                    #########################################################################
                    # TODO (Exercise 5):                                                    #
                    # Decay the learning rate according to a predefined schedule so that    #
                    # learning becomes more refined (slower). For our task, we shall decay  #
                    # our learning rate by 'learning_rate_decay' at the end of EVERY epoch. #
                    # Multiply 'learning_rate' by 'learning_rate_decay' to refine the       #
                    # learning rate                                                         #
                    #########################################################################
                    learning_rate = learning_rate * learning_rate_decay
                    #########################################################################
                    #                             END OF YOUR CODE                          #
                    #########################################################################
               
                    train_acc = 0
                    val_acc = 0               
                    
                    #########################################################################
                    # TODO (Exercise 6):                                                    #
                    # At the end of each epoch, check the training and validation accuracy  #
                    # Invoke 'predict' function to predict the labels for the training and  #
                    # validation set. Then, append the score to 'train_acc_history' and     #
                    # 'val_acc_history'                                                     #
                    #########################################################################
                    y_pred = self.predict(X)
                    train_acc = np.mean(y_pred == y)
                    train_acc_history.append(train_acc)
                    y_pred = self.predict(X_val)
                    val_acc = np.mean(y_pred == y_val)
                    val_acc_history.append(val_acc)
                    #########################################################################
                    #                             END OF YOUR CODE                          #
                    #########################################################################
                    
                if verbose:
                    if ep_it == num_iters_per_epoch - 1 or it == max_iters - 1:
                        print('epoch {:>4d}/{:<4d} it {:<4d} | epit {:>4d}/{:<4d} | loss {:.4f} | train_acc {:.4f} | val_acc {:.4f}'.format(epoch+1, num_epochs, it+1, ep_it+1, num_iters_per_epoch, loss, train_acc, val_acc))
                    elif it % 100 == 0: 
                        print('                it {:^4d} | epit {:>4d}/{:<4d} | loss {:.4f}'.format(it+1, ep_it+1, num_iters_per_epoch, loss))
        
                # stop if reach maximum number of iterations
                it += 1
                if it >= max_iters:
                    break

            if it >= max_iters:
                break
                
        return {
          'loss_history': loss_history,
          'train_acc_history': train_acc_history,
          'val_acc_history': val_acc_history,
        }

    def predict(self, X):
        """
        Use the trained weights of this two-layer network to predict labels for
        data points. For each data point we predict scores for each of the C
        classes, and assign each data point to the class with the highest score.

        Inputs:
        - X: A numpy array of shape (N, D) giving N D-dimensional data points to
          classify.

        Returns:
        - y_pred: A numpy array of shape (N,) giving predicted labels for each of
          the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
          to have class c, where 0 <= c < C.
        """
        y_pred = None

        ###########################################################################
        # TODO (Exercise 3):                                                      #
        # Implement this function; ep_it should be VERY simple!                   #
        ###########################################################################
        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape

        # Compute the forward pass
        scores = None

        z1 = X.dot(W1) + b1
        a1 = np.maximum(z1,0)
        scores = a1.dot(W2) + b2
        
        y_pred = np.argmax(scores,axis=1)
        ###########################################################################
        #                              END OF YOUR CODE                           #
        ###########################################################################

        return y_pred


