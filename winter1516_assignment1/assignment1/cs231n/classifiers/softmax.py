import numpy as np
from random import shuffle


def softmax_loss_naive(W, X, y, reg):
    """
  Softmax loss function, naive implementation (with loops)

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
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_train = X.shape[0]
    num_class = W.shape[1]
    num_dim = X.shape[1]

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    for i in range(0, num_train):
        fs = np.dot(X[i, :], W)
        max_fs = max(fs)
        fs -= max_fs
        f_correct = fs[y[i]]
        loss += -np.log(np.exp(f_correct) / np.sum(np.exp(fs)))

        p1 = np.exp(fs) / np.sum(np.exp(fs))
        correct_y = np.zeros(num_class)
        correct_y[y[i]] = 1
        d = p1 - correct_y
        dW += X[i, :].reshape((num_dim, 1)) * d
    loss /= num_train
    loss += 0.5 * reg * np.sum(W * W)
    dW /= num_train
    dW += reg * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_train = X.shape[0]
    num_class = W.shape[1]
    num_dim = X.shape[1]

    XW = np.dot(X, W)
    max_per_row = np.max(XW, axis=1)
    XW -= max_per_row.reshape((num_train, 1))
    dominator = np.sum(np.exp(XW), axis=1)
    XW_y = np.choose(y, XW.T)
    numerator = np.exp(XW_y)

    p_matrix = np.exp(XW) / np.sum(np.exp(XW), axis=1).reshape((num_train), 1)
    y_matrix = np.zeros((num_train, num_class))
    y_matrix[np.arange(y_matrix.shape[0]), y] = 1
    d = p_matrix - y_matrix
    dW = np.dot(X.T, d)
    loss = np.sum(-np.log(numerator / dominator), axis=0)
    loss /= num_train
    loss += 0.5 * reg * np.sum(W * W)
    dW /= num_train
    dW += reg * W
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW