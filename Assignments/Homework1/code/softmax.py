import numpy as np
from layers import *


class SoftmaxClassifier(object):
    """
    A fully-connected neural network with
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecture should be fc - softmax if no hidden layer.
    The architecture should be fc - relu - fc - softmax if one hidden layer

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=28*28, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with fc weights                                  #
        # and biases using the keys 'W' and 'b'                                    #
        ############################################################################

        if (hidden_dim == None):

            self.params['W1'] = np.random.randn(input_dim, num_classes) * weight_scale
            self.params['b1'] = np.zeros(num_classes)

        else:

            self.params['W1'] = np.random.randn(input_dim, hidden_dim) * weight_scale
            self.params['b1'] = np.zeros(hidden_dim)
            self.params['W2'] = np.random.randn(hidden_dim, num_classes) * weight_scale
            self.params['b2'] = np.zeros(num_classes)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the one-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################

        if ('W2' not in self.params.keys()):

            out1, cache1 = fc_forward(X, self.params['W1'], self.params['b1'])
            out = out1

        else:

            out1, cache1 = fc_forward(X, self.params['W1'], self.params['b1'])
            out2, cache2 = relu_forward(out1)
            out3, cache3 = fc_forward(out2, self.params['W2'], self.params['b2'])
            out = out3

        scores = out

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the one-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################

        loss, dout = softmax_loss(out, y)

        if ('W2' not in self.params.keys()):
            loss += 0.5 * self.reg * np.sum(self.params['W1'] ** 2)
            dx, dw, db = fc_backward(dout, cache1)
            grads['W1'] = dw + self.reg * self.params['W1']
            grads['b1'] = db

        else:
            loss += 0.5 * self.reg * np.sum(self.params['W1'] ** 2)
            loss += 0.5 * self.reg * np.sum(self.params['W2'] ** 2)
            dout2, dw2, db2 = fc_backward(dout, cache3)
            dout1 = relu_backward(dout2, cache2)
            dx, dw, db = fc_backward(dout1, cache1)
            grads['W1'] = dw + self.reg * self.params['W1']
            grads['b1'] = db
            grads['W2'] = dw2 + self.reg * self.params['W2']
            grads['b2'] = db2

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
