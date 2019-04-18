import numpy as np

from layers import *


class ConvNet(object):
  """
  A convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - fc - softmax

  You may also consider adding dropout layer or batch normalization layer. 
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(1, 28, 28), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32, drop_out=False):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################

    self.drop_out = drop_out

    self.params['W1'] = np.random.randn(num_filters, input_dim[0], filter_size, filter_size)*weight_scale
    dim = int(num_filters * (input_dim[1] - filter_size + 1) / 2 * (input_dim[2] - filter_size + 1) / 2)
    self.params['W2'] = np.random.randn(dim, hidden_dim)*weight_scale
    self.params['W3'] = np.random.randn(hidden_dim, num_classes)*weight_scale

    self.params['b2'] = np.zeros(hidden_dim)
    self.params['b3'] = np.zeros(num_classes)

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.items():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1 = self.params['W1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    # filter_size = W1.shape[2]
    # conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################



    if not (self.drop_out):

      out1, cache1 = conv_forward(X, W1)
      out2, cache2 = relu_forward(out1)
      out3, cache3 = max_pool_forward(out2, pool_param)
      out4, cache4 = fc_forward(out3, W2, b2)
      out5, cache5 = relu_forward(out4)
      out6, cache6 = fc_forward(out5, W3, b3)
      out = out6
      scores = out

    else:
      out1, cache1 = conv_forward(X, W1)
      out2, cache2 = relu_forward(out1)
      out3, cache3 = max_pool_forward(out2, pool_param)
      out4, cache4 = fc_forward(out3, W2, b2)
      out5, cache5 = relu_forward(out4)
      out6, cache6 = fc_forward(out5, W3, b3)
      out = out6
      scores = out

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################

    loss, dout = softmax_loss(out, y)
    loss += 0.5 * self.reg * np.sum(self.params['W1'] ** 2)
    loss += 0.5 * self.reg * np.sum(self.params['W2'] ** 2)
    loss += 0.5 * self.reg * np.sum(self.params['W3'] ** 2)

    if not (self.drop_out):
      dout5, dw5, db5 = fc_backward(dout, cache6)
      dout4 = relu_backward(dout5, cache5)
      dout3, dw3, db3 = fc_backward(dout4, cache4)
      dout2 = max_pool_backward(dout3, cache3)
      dout1 = relu_backward(dout2, cache2)
      dout, dw = conv_backward(dout1, cache1)
      grads['W1'] = dw + self.reg * self.params['W1']
      grads['W2'] = dw3 + self.reg * self.params['W2']
      grads['b2'] = db3 + self.reg * self.params['b2']
      grads['W3'] = dw5 + self.reg * self.params['W3']
      grads['b3'] = db5 + self.reg * self.params['b3']
    else:
      dout5, dw5, db5 = fc_backward(dout, cache6)
      dout4 = relu_backward(dout5, cache5)
      dout3, dw3, db3 = fc_backward(dout4, cache4)
      dout2 = max_pool_backward(dout3, cache3)
      dout1 = relu_backward(dout2, cache2)
      dout, dw = conv_backward(dout1, cache1)
      grads['W1'] = dw + self.reg * self.params['W1']
      grads['W2'] = dw3 + self.reg * self.params['W2']
      grads['b2'] = db3 + self.reg * self.params['b2']
      grads['W3'] = dw5 + self.reg * self.params['W3']
      grads['b3'] = db5 + self.reg * self.params['b3']

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  
pass
