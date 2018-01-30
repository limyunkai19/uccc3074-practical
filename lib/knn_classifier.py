import numpy as np
from collections import Counter

class KNN(object):

    """ a kNN classifier with L2 distance """

    def __init__(self):
        self.X_train = None
        self.y_train = None

    def train(self, X, y):
        """
        Train the classifier. For k-nearest neighbors this is just
        memorizing the training data.

        Inputs:
        - X: A numpy array of shape (num_train, D) containing the training data
            consisting of num_train samples each of dimension D.
        - y: A numpy array of shape (N,) containing the training labels, where
            y[i] is the label for X[i].
        """
        #####################################################################
        # TODO:                                                             #
        # Train the classifier. Save the training data as X_train           #
        # and y_train                                                       #
        #####################################################################
        # Put your code here

        self.X_train = X
        self.y_train = y

        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################


    def compute_distances_two_loops(self, X):

        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a nested loop over both the training data and the
        test data.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data.

        Returns:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          is the Euclidean distance between the ith test point and the jth training
          point.
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):

                #####################################################################
                # TODO:                                                             #
                # Compute the l2 distance between the ith test point and the jth    #
                # training point, and store the result in dists[i, j]. You should   #
                # not use a loop over dimension.                                    #
                #####################################################################
                # Put your code here

                dists[i, j] = np.sum((X[i] - self.X_train[j])**2)**0.5

                #####################################################################
                #                       END OF YOUR CODE                            #
                #####################################################################
        return dists


    def compute_distances_one_loop(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a single loop over the test data.

        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            #######################################################################
            # TODO:                                                               #
            # Compute the l2 distance between the ith test point and all training #
            # points, and store the result in dists[i, :].                        #
            #######################################################################
            #
            # Put your code here
            #

            dists[i] = np.sqrt(np.sum((np.square(self.X_train - X[i])), axis=1))

            #######################################################################
            #                         END OF YOUR CODE                            #
            #######################################################################

        return dists

    def compute_distances_no_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using no explicit loops.

        Input / Output: Same as compute_distances_two_loops
        """
        #########################################################################
        # TODO:                                                                 #
        # Compute the l2 distance between all test points and all training      #
        # points without using any explicit loops, and store the result in      #
        # dists.                                                                #
        #                                                                       #
        # You should implement this function using only basic array operations; #
        # in particular you should not use functions from scipy.                #
        #                                                                       #
        # HINT: Try to formulate the l2 distance using matrix multiplication    #
        #       and two broadcast sums.                                         #
        #########################################################################
        #
        #  Put your code here
        #  Step 1: s1 = get the sum of square of X (a vector of size 500)
        #  Step 2: s2 = get the sum of square of X_train (a vector of size 5000)
        #  Step 3: s1s2 = sum s1 and s2 for each test and training sample pair, use broadcast sum (matrix of size 500 x 5000)
        #  Step 4: s3   = get the dot product between X and X_train (matrix of size 500 x 500)
        #  Step 5: result = simple formula to combine s1s2 and s3
        #

        s1 = np.sum(X**2, axis=1)
        s2 = np.sum(self.X_train**2, axis=1)
        s1s2 = s1.reshape(s1.size, 1) + s2.reshape(1, s2.size)
        s3 = np.dot(X, self.X_train.transpose())

        dists = (s1s2-2*s3)**0.5
        #########################################################################
        #                         END OF YOUR CODE                              #
        #########################################################################
        return dists

    def predict(self, X, which_ver=0, k=1):
        """
        Predict labels for test data using this classifier.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data consisting
            of num_test samples each of dimension D.
        - k: The number of nearest neighbors that vote for the predicted labels.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for
            the test data, where y[i] is the predicted label for the test point X[i].
        """

        if which_ver == 0:
            dists = self.compute_distances_no_loops(X)    # get the distance matrix (use the fully vectorized version)
        elif which_ver == 1:
            dists = self.compute_distances_one_loop(X)    # get the distance matrix (use the fully vectorized version)
        elif which_ver == 2:
            dists = self.compute_distances_two_loops(X)    # get the distance matrix (use the fully vectorized version)

        num_test = len(X)                             # number of test samples
        y_pred = np.zeros(num_test, dtype = 'int')    # use this to store result

        for i in range(num_test):

            closest_y = []     # a list of length k storing the labels of the k nearest neighbors to the ith test point.

            #########################################################################
            # TODO:                                                                 #
            # Use the distance matrix to find the k nearest neighbors of the ith    #
            # testing point, and use self.y_train to find the labels of these       #
            # neighbors. Store these labels in closest_y.                           #
            # Hint: Look up the function numpy.argsort                              #
            #########################################################################
            #
            # Put your code here
            #

            closest_y = np.take(self.y_train, np.argsort(dists[i]))

            #########################################################################
            # TODO:                                                                 #
            # Now that you have found the labels of the k nearest neighbors, you    #
            # need to find the most common label in the list closest_y of labels.   #
            # Store this label in y_pred[i]. Break ties (multiple labels with the   #
            # same frequency) by choosing the smaller label.                        #
            # Hint: Look up the function numpy.bincount and numpy.argmax            #
            #########################################################################
            #
            # Put your code here
            #

            y_pred[i] = np.argmax(np.bincount(closest_y[:k]))

            #########################################################################
            #                           END OF YOUR CODE                            #
            #########################################################################

        return y_pred
