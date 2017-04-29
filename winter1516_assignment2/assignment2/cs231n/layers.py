import numpy as np


def affine_forward(x, w, b):
    """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
  examples, where each example x[i] has shape (d_1, ..., d_k). We will
  reshape each input into a vector of dimension D = d_1 * ... * d_k, and
  then transform it to an output vector of dimension M.

  Inputs:
  - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
  - w: A numpy array of weights, of shape (D, M)
  - b: A numpy array of biases, of shape (M,)

  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
    out = None
    #############################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You     #
    # will need to reshape the input into rows.                                 #
    #############################################################################
    D = np.product(x.shape[1:])
    N = x.shape[0]
    x1 = x.reshape((N, D))

    out = np.dot(x1, w) + b
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
    x, w, b = cache
    dx, dw, db = None, None, None
    #############################################################################
    # TODO: Implement the affine backward pass.                                 #
    #############################################################################
    N = x.shape[0]
    db = np.sum(dout, axis=0)

    D = np.product(x.shape[1:])

    x1 = x.reshape((N, D))
    dw = np.dot(x1.T, dout)

    M = w.shape[1]
    D_expanded = x.shape[1:]
    w_new_shape = D_expanded + (M,)
    w1 = w.reshape(w_new_shape)
    dx = np.dot(dout, w.T)
    dx = dx.reshape(x.shape)
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
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
    out = None
    #############################################################################
    # TODO: Implement the ReLU forward pass.                                    #
    #############################################################################
    out = np.copy(x)
    out[out <= 0] = 0
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
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
    dx, x = None, cache
    #############################################################################
    # TODO: Implement the ReLU backward pass.                                   #
    #############################################################################
    x[x <= 0] = 0
    x[x > 0] = 1
    dx = dout * x
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
  Forward pass for batch normalization.

  During training the sample mean and (uncorrected) sample variance are
  computed from minibatch statistics and used to normalize the incoming data.
  During training we also keep an exponentially decaying running mean of the mean
  and variance of each feature, and these averages are used to normalize data
  at test-time.

  At each timestep we update the running averages for mean and variance using
  an exponential decay based on the momentum parameter:

  running_mean = momentum * running_mean + (1 - momentum) * sample_mean
  running_var = momentum * running_var + (1 - momentum) * sample_var

  Note that the batch normalization paper suggests a different test-time
  behavior: they compute sample mean and variance for each feature using a
  large number of training images rather than using a running average. For
  this implementation we have chosen to use running averages instead since
  they do not require an additional estimation step; the torch7 implementation
  of batch normalization also uses running averages.

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
        #############################################################################
        # TODO: Implement the training-time forward pass for batch normalization.   #
        # Use minibatch statistics to compute the mean and variance, use these      #
        # statistics to normalize the incoming data, and scale and shift the        #
        # normalized data using gamma and beta.                                     #
        #                                                                           #
        # You should store the output in the variable out. Any intermediates that   #
        # you need for the backward pass should be stored in the cache variable.    #
        #                                                                           #
        # You should also use your computed sample mean and variance together with  #
        # the momentum variable to update the running mean and running variance,    #
        # storing your result in the running_mean and running_var variables.        #
        #############################################################################
        u = np.mean(x, axis=0)
        numerator = x - u
        v = np.var(x, axis=0)
        v_plus_eps = v + eps
        denominator = np.sqrt(v_plus_eps)
        inv_denominator = 1. / denominator
        xhat = numerator * inv_denominator
        gamma_mul_xhat = gamma * xhat
        y = gamma_mul_xhat + beta
        out = y
        cache = x, u, numerator, v, v_plus_eps, denominator, inv_denominator, xhat, gamma_mul_xhat, y, gamma, beta

        running_mean = momentum * running_mean + (1 - momentum) * u
        running_var = momentum * running_var + (1 - momentum) * v
        bn_param['running_mean'] = running_mean
        bn_param['running_var'] = running_var
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
    elif mode == 'test':
        #############################################################################
        # TODO: Implement the test-time forward pass for batch normalization. Use   #
        # the running mean and variance to normalize the incoming data, then scale  #
        # and shift the normalized data using gamma and beta. Store the result in   #
        # the out variable.                                                         #
        #############################################################################
        u = bn_param['running_mean']
        numerator = x - u
        v = bn_param['running_var']
        v_plus_eps = v + eps
        denominator = np.sqrt(v_plus_eps)
        inv_denominator = 1. / denominator
        xhat = numerator * inv_denominator
        gamma_mul_xhat = gamma * xhat
        y = gamma_mul_xhat + beta
        out = y
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
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
    x, u, numerator, v, v_plus_eps, denominator, inv_denominator, xhat, gamma_mul_xhat, y, gamma, beta = cache

    #############################################################################
    # TODO: Implement the backward pass for batch normalization. Store the      #
    # results in the dx, dgamma, and dbeta variables.                           #
    #############################################################################
    dx = np.zeros(dout.shape)
    N = dout.shape[0]
    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout * xhat, axis=0)

    dxhat = dout * gamma

    dinv_den = np.sum(dxhat * numerator, axis=0)
    dsqrt_v_plus_eps = dinv_den * (-1) * (denominator ** (-2))
    dv_plus_eps = dsqrt_v_plus_eps * 0.5 * (v_plus_eps ** (-0.5))
    dv = 2. / N * (x - u) * dv_plus_eps
    dx += dv
    du1 = -np.sum(dv, axis=0)
    dux = np.full(dout.shape, 1. / N)
    du1_x = du1 * dux
    dx += du1_x

    dnumerator = dxhat * inv_denominator
    dx += dnumerator
    du2 = -np.sum(dnumerator, axis=0)
    du2_x = du2 * dux
    dx += du2_x

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
  Performs the forward pass for (inverted) dropout.

  Inputs:
  - x: Input data, of any shape
  - dropout_param: A dictionary with the following keys:
    - p: Dropout parameter. We drop each neuron output with probability p.
    - mode: 'test' or 'train'. If the mode is train, then perform dropout;
      if the mode is test, then just return the input.
    - seed: Seed for the random number generator. Passing seed makes this
      function deterministic, which is needed for gradient checking but not in
      real networks.

  Outputs:
  - out: Array of the same shape as x.
  - cache: A tuple (dropout_param, mask). In training mode, mask is the dropout
    mask that was used to multiply the input; in test mode, mask is None.
  """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    mask = np.random.random(x.shape)
    mask[mask < p] = 0
    mask[mask >= p] = 1

    # mask=(np.random.random(x.shape) > p) / (1 - p)


    if mode == 'train':
        ###########################################################################
        # TODO: Implement the training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                            #
        ###########################################################################
        out = x * mask
        ###########################################################################
        #                            END OF YOUR CODE                             #
        ###########################################################################
    elif mode == 'test':
        ###########################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.       #
        ###########################################################################
        out = x * (1 - p)
        # out = x
        ###########################################################################
        #                            END OF YOUR CODE                             #
        ###########################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
  Perform the backward pass for (inverted) dropout.

  Inputs:
  - dout: Upstream derivatives, of any shape
  - cache: (dropout_param, mask) from dropout_forward.
  """
    dropout_param, mask = cache
    mode = dropout_param['mode']
    dx = None
    if mode == 'train':
        ###########################################################################
        # TODO: Implement the training phase backward pass for inverted dropout.  #
        ###########################################################################
        dx = np.ones(mask.shape) * mask
        dx *= dout
        ###########################################################################
        #                            END OF YOUR CODE                             #
        ###########################################################################
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
    out = None
    #############################################################################
    # TODO: Implement the convolutional forward pass.                           #
    # Hint: you can use the function np.pad for padding.                        #
    #############################################################################
    pad = conv_param['pad']
    stride = conv_param['stride']
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    x = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant', constant_values=0)
    WOUT = (W - WW + 2 * pad) / stride + 1
    HOUT = (H - HH + 2 * pad) / stride + 1
    t = np.zeros((N, WW * HH * C, WOUT * HOUT))
    for i in range(0, N):
        x_i = x[i, :, :, :]
        # for each image
        l = 0
        for j in range(0, HOUT):
            for k in range(0, WOUT):
                # strech out filter into a col vector
                slide_filter = x_i[:, j * stride:j * stride + HH, k * stride:k * stride + WW]
                slide_filter_col = slide_filter.reshape((WW * HH * C))
                # there are WOUT * HOUT such col vector
                t[i, :, l] = slide_filter_col
                l += 1
    w_reshape = w.reshape((F, C * HH * WW))
    out = np.zeros((N, F, WOUT * HOUT))
    for i in range(0, N):
        out[i] = w_reshape.dot(t[i]) + b.reshape((F, 1))
    out = out.reshape((N, F, HOUT, WOUT))
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
    dx, dw, db = None, None, None
    #############################################################################
    # TODO: Implement the convolutional backward pass.                          #
    #############################################################################
    # db
    db1 = np.sum(dout, axis=0)
    db2 = np.sum(db1, axis=1)
    db = np.sum(db2, axis=1)

    # dx
    # x is after padding
    x, w, b, conv_param = cache
    N = x.shape[0]
    C = w.shape[1]
    F = dout.shape[1]
    HH = w.shape[2]
    WW = w.shape[3]
    HOUT = dout.shape[2]
    WOUT = dout.shape[3]
    stride = conv_param['stride']
    # get t from x
    t = np.zeros((N, WW * HH * C, WOUT * HOUT))
    for i in range(0, N):
        x_i = x[i, :, :, :]
        # for each image
        l = 0
        for j in range(0, HOUT):
            for k in range(0, WOUT):
                # strech out filter into a col vector
                slide_filter = x_i[:, j * stride:j * stride + HH, k * stride:k * stride + WW]
                slide_filter_col = slide_filter.reshape((WW * HH * C))
                # there are WOUT * HOUT such col vector
                t[i, :, l] = slide_filter_col
                l += 1
    # dout,(N=4,F=2,HOUT=5,WOUT=5)
    # dw,(F=2, C=3, HH=3, WW=3)
    # t,(N=4,(WW * HH * C),(HOUT * WOUT))
    dout = dout.reshape(N, F, (HOUT * WOUT))
    t = np.transpose(t, axes=(0, 2, 1))
    dw = np.zeros(w.shape)
    for i in range(0, N):
        t_i = t[i, :, :]
        dout_i = dout[i, :, :]
        # dout_i:(F,(HOUT * WOUT)) after reshaping, t_i:((HOUT * WOUT),(WW * HH * C)) after transposing
        dw_i = dout_i.dot(t_i)
        dw_i = dw_i.reshape((F, C, HH, WW))
        dw += dw_i

    # dx
    # dx:(N,C,HOUT,WOUT),dout:(N,F,(HOUT * WOUT)),w:(F=2,(C*HH*WW))
    w = w.reshape((F, (C * HH * WW)))
    # dout:(N,(HOUT * WOUT),F)
    dout = np.transpose(dout, axes=(0, 2, 1))
    # dt:(N, (HOUT * WOUT), (C*HH*WW))
    dt = dout.dot(w)
    # dt:(N, (C*HH*WW),(HOUT * WOUT))
    dt = np.transpose(dt, axes=(0, 2, 1))
    dx = np.zeros(x.shape)
    # recover dx from dt
    for i in range(0, N):
        x_i = dx[i, :, :, :]
        l = 0
        for j in range(0, HOUT):
            for k in range(0, WOUT):
                slide_filter_col = dt[i, :, l]
                slide_filter = slide_filter_col.reshape((C, HH, WW))
                # += not =, overlapping is needed
                x_i[:, j * stride:j * stride + HH, k * stride:k * stride + WW] += slide_filter
                l += 1
    # delete padding
    dx = dx[:, :, 1:-1, 1:-1]

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
    out = None
    #############################################################################
    # TODO: Implement the max pooling forward pass                    #
    #############################################################################
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']
    N, C, H, W = x.shape

    HOUT = (H - pool_height) / stride + 1
    WOUT = (W - pool_width) / stride + 1
    out = np.zeros((N, C, HOUT, WOUT))
    for i in range(0, N):
        x_i = x[i, :, :, :]
        for j in range(0, HOUT):
            for k in range(0, WOUT):
                slide_filter = x_i[:, j * stride:j * stride + pool_height, k * stride:k * stride + pool_width]
                max_entry = np.max(slide_filter, axis=(1, 2))
                out[i, :, j, k] = max_entry
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
    dx = None
    #############################################################################
    # TODO: Implement the max pooling backward pass                             #
    #############################################################################
    x, pool_param = cache
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']
    N, C, H, W = x.shape
    HOUT = (H - pool_height) / stride + 1
    WOUT = (W - pool_width) / stride + 1
    dx = np.zeros(x.shape)
    for i in range(0, N):
        x_i = x[i, :, :, :]
        for j in range(0, HOUT):
            for k in range(0, WOUT):
                slide_filter = x_i[:, j * stride:j * stride + pool_height, k * stride:k * stride + pool_width]
                # now 2 dimensional
                slide_filter_streched = slide_filter.reshape((C, pool_height * pool_width))
                indexes = np.argmax(slide_filter_streched, axis=1)
                # compute max entry's relative h,w in a filter
                for t in range(0, len(indexes)):
                    index = indexes[t]
                    h = index / pool_width
                    w = index % pool_width
                    # recover the absolute position in dx
                    # the gradient is 1, so the final value is just upstream derivates by chain rule
                    dx[i][t][j * stride + h][k * stride + w] = dout[i][t][j][k]
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
  Computes the forward pass for spatial batch normalization.

  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features

  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
  """
    out, cache = None, None

    #############################################################################
    # TODO: Implement the forward pass for spatial batch normalization.         #
    #                                                                           #
    # HINT: You can implement spatial batch normalization using the vanilla     #
    # version of batch normalization defined above. Your implementation should  #
    # be very short; ours is less than five lines.                              #
    #############################################################################
    N, C, H, W = x.shape
    # x:(N,C,H,W)->(C,N,H,W)
    x_NC_transposed = np.transpose(x, axes=(1,0,2,3))
    # stretch (H,W) to (H*W)
    x_NC_transposed_reshaped = x_NC_transposed.reshape(C,N,H*W)
    # result holder
    out = np.zeros((C,N,H*W))
    cache=[]
    # call previous batchnorm forward per depth
    for i in range(0, C):
      out[i],cachei = batchnorm_forward(x_NC_transposed_reshaped[i], gamma[i], beta[i], bn_param)
      cache.append(cachei)
    # recover result in demanded form
    out = np.transpose(out.reshape(C,N,H,W),axes=(1,0,2,3))
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
  Computes the backward pass for spatial batch normalization.

  Inputs:
  - dout: Upstream derivatives, of shape (N, C, H, W)
  - cache: Values from the forward pass

  Returns a tuple of:
  - dx: Gradient with respect to inputs, of shape (N, C, H, W)
  - dgamma: Gradient with respect to scale parameter, of shape (C,)
  - dbeta: Gradient with respect to shift parameter, of shape (C,)
  """
    dx, dgamma, dbeta = None, None, None

    #############################################################################
    # TODO: Implement the backward pass for spatial batch normalization.        #
    #                                                                           #
    # HINT: You can implement spatial batch normalization using the vanilla     #
    # version of batch normalization defined above. Your implementation should  #
    # be very short; ours is less than five lines.                              #
    #############################################################################
    N, C, H, W = dout.shape
    # dout:(N,C,H,W)->(C,N,H,W)
    dout_NC_transposed = np.transpose(dout,axes=(1,0,2,3))
    # stretch (H,W) to (H*W)
    dout_transposed_reshaped = dout_NC_transposed.reshape((C,N,H*W))
    # result holders
    dx = np.zeros((C,N,H*W))
    dgamma = np.zeros((C,))
    dbeta = np.zeros((C,))
    # call previous batchnorm backward per depth
    for i in range(0,C):
      dxi, dgammai, dbetai = batchnorm_backward(dout_transposed_reshaped[i],cache[i])
      dx[i] = dxi
      # note dgammai and dbetai are (H*W), but we want each depth slice has only one gamma and beta
      # so we just sum over the 2d matrix to obtain a single value
      dgamma[i] = np.sum(dgammai)
      dbeta[i] = np.sum(dbetai)
    # recover result in demanded form
    dx = np.transpose(dx.reshape((C,N,H,W)),axes=(1,0,2,3))
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
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
    probs = np.exp(x - np.max(x, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    N = x.shape[0]
    loss = -np.sum(np.log(probs[np.arange(N), y])) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx