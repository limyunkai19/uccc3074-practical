import numpy as np
import os
# from scipy.misc import imread
from six.moves import cPickle
import sys

def get_classes ():
    return ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def load_batch(fpath):
    """Internal utility for parsing CIFAR data.

    Arguments:
        fpath: path the file to parse.
        label_key: key for label data in the retrieve
          dictionary.

    Returns:
        A tuple `(data, labels)`.
    """
    print ('Load', fpath)
    f = open(fpath, 'rb')
    if sys.version_info < (3,):
        d = cPickle.load(f)
    else:
        d = cPickle.load(f, encoding='bytes')
        # decode utf8
        d_decoded = {}
        for k, v in d.items():
          d_decoded[k.decode('utf8')] = v
        d = d_decoded
    f.close()
    data = d['data']
    labels = d['labels']
    data = data.reshape(data.shape[0], 3, 32, 32)
    return data, labels


def load_data(path, data_format = 'channels_last'):
    """ load all of cifar """
    num_train_samples = 50000
    num_test_samples = 10000

    X_train = np.zeros((num_train_samples, 3, 32, 32), dtype='uint8')
    y_train = np.zeros((num_train_samples,), dtype='uint8')

    for i in range(1, 6):
        fpath = os.path.join(path, 'data_batch_%d' % (i, ))
        data, labels = load_batch(fpath)
        X_train[(i - 1) * 10000:i * 10000, :, :, :] = data
        y_train[(i - 1) * 10000:i * 10000] = labels

    fpath = os.path.join(path, 'test_batch')
    X_test, y_test = load_batch(fpath)
    X_test = X_test.astype('uint8')
    y_test = np.array(y_test, dtype='uint8')

    if data_format == 'channels_last':
        X_train = X_train.transpose(0, 2, 3, 1)
        X_test = X_test.transpose(0, 2, 3, 1)

    return X_train, y_train, X_test, y_test


def get_CIFAR10_data(path, num_training=49000, num_validation=1000, num_test=1000, num_dev=500, b_center = True, b_add_bias = False):
    """
    1. Load the CIFAR-10 dataset from disk (size = [# samples, 32, 32, 3])
    2. Extract the training set, validation set, testing set and development set).
       The development set is used for sanity check during coding.
       The training and validation set is used for hyperparameter tuning and model building
       The testing set is used for testing and evaluation.
    3. Reshape the shape of the samples from [#sample, 32, 32, 3] to [#sample, 3072]
    4. Center the data
    5. Add bias to dataset. Shape of the samples changes to [#sample, 3073]

    Arguments:
        path: location of the CIFAR-10 directory (cifar-10-batches-py)
        num_training: number of training samples (must be < 50000)
        num_validation: number of validation samples (must be < 50000 and < num_training, num_training + num_validation < 500000)
        num_test: number of testing samples (must be < 10000)
        num_dev: number of samples for sanity check during coding (must be < num_training)
        b_preprocess: True: perform centering and add bias
    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test, X_dev, y_dev
    """
    # Load the raw CIFAR-10 data
    X_train, y_train, X_test, y_test = load_data(path)

    # Convert uint8 to double type
    X_train = X_train.astype('double')
    X_test  = X_test.astype('double')

    # subsample the data
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask]
    y_val = y_train[mask]

    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]

    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]

    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_val = np.reshape(X_val, (X_val.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))

    if b_center:

        # Normalize the data: subtract the mean image
        mean_image = np.mean(X_train, axis = 0)
        X_train -= mean_image
        X_val -= mean_image
        X_test -= mean_image

    if b_add_bias:
        # add bias dimension (last column)
        X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
        X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
        X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])


    if num_dev > 0:
        mask = np.random.choice(num_training, num_dev, replace=False)
        X_dev = X_train[mask]
        y_dev = y_train[mask]

        if b_add_bias:
            X_dev = np.reshape(X_dev, (X_dev.shape[0], -1))

        if b_preprocess:
            X_dev -= mean_image
            X_dev = np.hstack([X_dev, np.ones((X_dev.shape[0], 1))])

        return X_train, y_train, X_val, y_val, X_test, y_test, X_dev, y_dev

    return X_train, y_train, X_val, y_val, X_test, y_test
