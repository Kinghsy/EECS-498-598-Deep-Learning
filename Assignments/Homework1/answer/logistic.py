import numpy as np

from layers import *

class LogisticClassifier(object):
  """
  A logistic regression model with optional hidden layers.
  
  Note that this class does not implement gradient descent; instead, it
  will interact with a separate Solver object that is responsible for running
  optimization.

  The learnable parameters of the model are stored in the dictionary
  self.params that maps parameter names to numpy arrays.
  """
  
  def __init__(self, input_dim=100, hidden_dim=None, weight_scale=1e-3, reg=0.0):
    """
    Initialize a new network.

    Inputs:
    - input_dim: An integer giving the size of the input
    - hidden_dim: An integer giving the size of the hidden layer
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - reg: Scalar giving L2 regularization strength.
    """
    self.params = {}
    self.reg = reg
    
    ############################################################################
    # TODO: Initialize the weights and biases of the model. Weights            #
    # should be initialized from a Gaussian with standard deviation equal to   #
    # weight_scale, and biases should be initialized to zero. All weights and  #
    # biases should be stored in the dictionary self.params, with first layer  #
    # weights and biases using the keys 'W1' and 'b1' and second layer weights #
    # and biases (if any) using the keys 'W2' and 'b2'.                        #
    if (hidden_dim == None):

        W1 = np.random.normal(0, weight_scale ,(input_dim,1))
        b1 = np.zeros(((1)))

        self.params['W1'] = W1
        self.params['b1'] = b1


    else:

        # W1 = np.random.normal(0, weight_scale, (input_dim, hidden_dim))
        #
        # b1 = np.zeros(((hidden_dim )))
        #
        # W2 = np.random.normal(0, weight_scale, (hidden_dim, hidden_dim))
        # b2 = np.zeros((( hidden_dim)))

        W1 = np.random.normal(0, weight_scale, (input_dim, hidden_dim))

        b1 = np.zeros(((hidden_dim)))

        W2 = np.random.normal(0, weight_scale, (hidden_dim, 1))

        b2 = np.zeros(((1)))

        self.params['W1'] = W1
        self.params['b1'] = b1
        self.params['W2'] = W2
        self.params['b2'] = b2
        # self.params['W3'] = W3
        # self.params['b3'] = b3
    ############################################################################

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################


  def loss(self, X, y=None):
    """
    Compute loss and gradient for a minibatch of data.

    Inputs:
    - X: Array of input data of shape (N, D)
    - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

    Returns:
    If y is None, then run a test-time forward pass of the model and return:
    - scores: Array of shape (N,) where scores[i] represents the logit for X[i]

    If y is not None, then run a training-time forward and backward pass and
    return a tuple of:
    - loss: Scalar value giving the loss
    - grads: Dictionary with the same keys as self.params, mapping parameter
      names to gradients of the loss with respect to those parameters.
    """
    score = None


    ############################################################################
    # TODO: Implement the forward pass for the model, computing the            #
    # scores for X and storing them in the scores variable.                    #

    ############################################################################
    if (self.params['W1'].shape[1] == 1):
        out1, cache1 = fc_forward( X, self.params['W1'], self.params['b1'])

    else:

        out11, cache_forward1 = fc_forward(X, self.params['W1'], self.params['b1'])
        # out2, cache_relu1 =  relu_forward( out11)
        # out3, cache_forward2 = fc_forward(out2, self.params['W2'], self.params['b2'])
        # out4, cache_relu2 = relu_forward(out2.dot(self.params["W2"]))
        # out5, cache_forward3 = fc_forward(out4, self.params['W3'], self.params['b3'])

        out2, cache_relu1 = relu_forward(out11)
        out3, cache_forward2 = fc_forward(out2, self.params['W2'], self.params['b2'])
        out1 = out3
        # print("out2", out2[1:10, :])
        # print("out3", out2[0:10,:].dot(self.params['W2'][0:10,:]) )
        # print()
        # print("out4", out4[1:10, :])
      #  print("out3",out3)

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If y is None then we are in test mode so just return scores

    if y is None:
        score = sigmoid(out1)
        return score[:,0]-0.5

    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the model. Store the loss          #
    # in the loss variable and gradients in the grads dictionary. Compute data #
    # loss and make sure that grads[k] holds the gradients for self.params[k]. #
    # Don't forget to add L2 regularization.                                   #
    #                                                                          #
    ############################################################################
    if  ( self.params['W1'].shape[1] == 1):
        loss, temp = logistic_loss(out1[:,0], y)

        temp = np.array([temp]).transpose()

        loss += 1.0 / 2 * self.reg *np.power(self.params['W1'],2).sum()

        _, dW1, db1 = fc_backward(temp, cache1)

        dW1 = dW1 + self.reg *  dW1

        grads['W1'] = dW1
        grads['b1'] = db1


    else:


      #  print(out1.sum(axis=1).shape)
        loss, temp = logistic_loss(out1[:,0], y)

        temp = np.array([temp]).transpose()

        loss = loss + 1.0 / 2 * self.reg * np.power(self.params['W1'], 2).sum()

        # dW3 = fc_backward(temp, cache_forward3)[1]
        #
        # db3 = fc_backward(temp, cache_forward3)[2]
        #
        # temp1 = fc_backward(temp, cache_forward3)[0]
        #
        # drelu = relu_backward(temp1, cache_relu2)

        # dW2 = fc_backward(temp, cache_forward2)[1]
        #
        # db2 = fc_backward(temp, cache_forward2)[2]
        #
        # dforword2 = fc_backward(temp, cache_forward2)[0]

     #   dW2 = dW2 + self.reg * dW2

     #   drelu1 = relu_backward(dforword2, cache_relu1)

        dx ,dW2,db2=  fc_backward(temp, cache_forward2)

        d1 = relu_backward(dx, cache_relu1)

        _, dW1,db1 = fc_backward(d1, cache_forward1)


        #
        # print("W1",dW1[0:10])
        # print("W2", dW2[0:10])
        # print("real W1", self.params["W1"][0:10])
        # print("real W2", self.params["W2"][0:10])
    # dW1 = dW1 + self.reg *  dW1

        grads['W1'] = dW1
        grads['b1'] = db1

        grads['W2'] = dW2
        grads['b2'] = db2

        # grads['W3'] = dW3
        # grads['b3'] = db3
        #
        # print("grads W3",grads["W3"])
        #
        # print("W2", cache_forward3[2])
        # print("W2",dW2)
        # print("W3",dW3)

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads

