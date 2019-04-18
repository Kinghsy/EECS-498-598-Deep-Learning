from builtins import range
import numpy as np


def fc_forward(x, w, b):
    """
    Computes the forward pass for a fully-connected layer.

    The input x has shape (N, Din) and contains a minibatch of N
    examples, where each example x[i] has shape (Din,).

    Inputs:
    - x: A numpy array containing input data, of shape (N, Din)
    - w: A numpy array of weights, of shape (Din, Dout)
    - b: A numpy array of biases, of shape (Dout,)

    Returns a tuple of:
    - out: output, of shape (N, Dout)
    - cache: (x, w, b)
    """
    ###########################################################################
    # TODO: Implement the forward pass. Store the result in out.              #
    ###########################################################################

    xs = np.shape(x)
    out = np.dot(np.reshape(x, (xs[0], int(np.prod(xs)/xs[0]))), w) + b
    cache = (x, w, b)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def fc_backward(dout, cache):
    """
    Computes the backward pass for a fully_connected layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, Dout)
    - cache: Tuple of:
      - x: Input data, of shape (N, Din)
      - w: Weights, of shape (Din, Dout)
      - b: Biases, of shape (Dout,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, Din)
    - dw: Gradient with respect to w, of shape (Din, Dout)
    - db: Gradient with respect to b, of shape (Dout,)
    """

    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################

    x, w, b = cache
    xs = np.shape(x)
    x__ = np.reshape(x, (xs[0], int(np.prod(xs)/xs[0])))
    db = np.sum(dout, axis=0)
    dw = np.zeros((np.shape(x__)[1], np.shape(dout)[1]))
    for i in range(0, np.shape(x__)[0]):
        dw = dw + np.outer(x__[i], dout[i])
    dx = np.dot(dout, w.T).reshape(xs)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################

    xs = np.shape(x)
    x__ = x.copy().reshape(1, np.prod(xs))
    xss = np.shape(x__)
    for i in range(0, np.prod(xs)):
        x__[0][i] = max(0, x__[0][i])
    out = x__.reshape(xs)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """

    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################

    x = cache
    dx = dout.copy()
    dx = dx * ( x>0 )

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        # 
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################

        x_mean = np.mean(x, axis = 0)
        x_var = np.var(x, axis = 0)

        x_diff = x - x_mean
        xivar = np.sqrt(x_var + eps)
        x_normed = x_diff / xivar
        out = gamma * x_normed + beta

        running_mean = running_mean * momentum + (1-momentum) * x_mean
        running_var = running_var * momentum + (1-momentum) * x_var
        cache = ( x_mean, x_var, x_diff, xivar, x_normed, gamma, beta, x, eps)

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################

        x_normed = (x-running_mean)/(np.sqrt(running_var + eps))
        out = gamma * x_normed + beta

        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################

    x_mean, x_var, x_diff, xivar, x_normed, gamma, beta, x, eps = cache
    xs = np.shape(x)
    dx_hat = dout * gamma
    dvar = np.sum(- dx_hat * x_diff * np.power(x_var + eps, -1.5) / 2, axis = 0)
    dmean = np.sum(- dx_hat / xivar, axis = 0)
    dx = dx_hat / xivar + 2 * dvar * x_diff / xs[0] + dmean / xs[0]
    dgamma = np.sum(dout * x_normed, axis = 0)
    dbeta = np.sum(dout, axis = 0)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.

    NOTE: Implement the vanilla version of dropout.

    NOTE 2: Keep in mind that p is the probability of **keep** a neuron
    output; this might be contrary to some sources, where it is referred to
    as the probability of dropping a neuron output.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################

        mask = np.zeros_like(x)
        prob = np.random.rand(*np.shape(x))
        mask[prob <= p] = 1
        out = mask * x

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################

        out = x

        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################

        dx = dout * mask

        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward(x, w):
    """
    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW. Assume that stride=1 
    and there is no padding. You can ignore the bias term in your 
    implementation.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = H - HH + 1
      W' = W - WW + 1
    - cache: (x, w)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################

    N, C, H, W = np.shape(x)
    F, C, HH, WW = np.shape(w)
    H_target = H - HH + 1
    W_target = W - WW + 1
    out = np.zeros((N, F, H_target, W_target))
    # ww = np.zeros_like(w)
    # for f__ in range(0, F):
    #     for c__ in range(0, C):
    #         ww[f__, c__, :, :] = w[f__, c__, :, :].reshape((WW*HH))[::-1].reshape((HH, WW))

    for n__ in range(0, N):
        for f__ in range(0, F):
            for h__ in range(0, H_target):
                for w__ in range(0, W_target):
                    for c__ in range(0, C):
                        mat1 = x[n__, c__, h__:h__+HH, w__:w__+WW]
                        mat2 = w[f__, c__, :, :]
                        out[n__, f__, h__, w__] = out[n__, f__, h__, w__] + np.sum(mat1 * mat2)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w)
    return out, cache


def conv_backward(dout, cache):
    """
    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w) as in conv_forward

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    """
    dx, dw = None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################

    x, w = cache
    N, C, H, W = np.shape(x)
    F, C, HH, WW = np.shape(w)

    dx = np.zeros_like(x).astype(np.float32)
    # dww = np.zeros_like(w)
    dw = np.zeros_like(w).astype(np.float32)
    # ww = np.zeros_like(w)
    # for f__ in range(0, F):
    #     for c__ in range(0, C):
    #         ww[f__, c__, :, :] = w[f__, c__, :, :].reshape((HH*WW))[::-1].reshape((HH, WW))

    for n__ in range(0, N):
        for f__ in range(0, F):
            for h__ in range(0, H-HH+1):
                for w__ in range(0, W-WW+1):
                    dw[f__, :, :, :] += dout[n__, f__, h__, w__] * x[n__, : , h__:h__+HH, w__:w__+WW]
                    dx[n__, :, h__:h__+HH, w__:w__+WW] += dout[n__, f__, h__, w__] * w[f__, :, :, :]

    # for f__ in range(0, F):
    #     for c__ in range(0, C):
    #         dw[f__, c__, :, :] = dww[f__, c__, :, :].reshape((HH*WW))[::-1].reshape((HH, WW))

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw


def max_pool_forward(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here. Output size is given by 

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################

    height = pool_param['pool_height']
    width = pool_param['pool_width']
    stride = pool_param['stride']
    N, C, H, W = x.shape
    h_target = int( 1 + (H - height) / stride )
    w_target = int( 1 + (W - width) / stride )

    out = np.zeros((N, C, h_target, w_target))
    for n__ in range(0, N):
        for c__ in range(0, C):
            for h__ in range(0, h_target):
                for w__ in range(0, w_target):
                    out[n__, c__, h__, w__] = np.max(x[n__, c__, h__*stride:h__*stride+height, w__*stride:w__*stride+width])

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################

    x, pool_param = cache
    height = pool_param['pool_height']
    width = pool_param['pool_width']
    stride = pool_param['stride']
    N, C, H, W = x.shape
    h_target = int( 1 + (H - height) / stride )
    w_target = int( 1 + (W - width) / stride )
    dx = np.zeros_like(x)

    for n__ in range(0, N):
        for c__ in range(0, C):
            for h__ in range(0, h_target):
                for w__ in range(0, w_target):
                    mat = x[n__, c__, h__ * stride:h__ * stride + height, w__ * stride:w__ * stride + width]
                    pos = np.where(mat == np.max(mat))
                    dx[n__, c__, h__*stride+pos[0], w__*stride+pos[1]] += dout[n__, c__, h__, w__]

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def svm_loss(x, y):
    """
    Computes the loss and gradient for binary SVM classification.
    Inputs:
    - x: Input data, of shape (N,) where x[i] is the score for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i]
    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    xs = np.shape(x)
    temp = 1 - y.reshape(xs[0])*x.reshape(xs[0])
    loss = np.sum( max(0, temp[i]) for i in range(0, xs[0])) / xs[0]
    dx =( - (np.sign(temp) + 1)  * (y.reshape(xs[0])) / (2 * xs[0]) ).reshape(xs)

    return loss, dx


def logistic_loss(x, y):
    """
    Computes the loss and gradient for binary classification with logistic
    regression.
    Inputs:
    - x: Input data, of shape (N,) where x[i] is the logit for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i]
    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """

    x_sig = sigmoid(x)
    loss = -np.sum((y * np.log(x_sig) + (1.0 - y) * np.log((1.0 - x_sig)))) / x.shape[0]
    dx = (- 1.000 / x_sig.shape[0]) * ((y / x_sig) - (1 - y) / (1.000 - x_sig)) * dsigmoid(x)

    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.
    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
        for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
        0 <= y[i] < C
    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """

    xs = np.shape(x)
    loss = 0
    dx = np.zeros_like(x)
    for i in range(0, xs[0]):
        total = np.sum(np.exp(x[i]))
        loss = loss - np.log(np.exp(x[i][y[i]]) / total)
        dx[i] = (np.exp(x[i]) / total).copy()
        dx[i][y[i]] = dx[i][y[i]] - 1
        dx[i] = dx[i] / xs[0]
    loss = loss / xs[0]

    return loss, dx

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def dsigmoid(x):
    return sigmoid(x) * (1.0 - sigmoid(x))