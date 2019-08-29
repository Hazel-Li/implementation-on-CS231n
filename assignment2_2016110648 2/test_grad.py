
"""
Test correctness of neural nets by checking gradients
"""

import numpy as np
from nn.layers import Conv, MaxPool, Linear, Relu
from nn.cnn import CNN
from check_gradient import *


def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


check_conv = False
check_pool_back =False
check_conv_back = False
check_cnn_back = True

# check convolution is correct
if check_conv:

    x_shape = (2, 3, 4, 4)
    w_shape = (3, 3, 4, 4)
    x = np.linspace(-0.1, 0.5, num=np.prod(x_shape)).reshape(x_shape)
    w = np.linspace(-0.2, 0.3, num=np.prod(w_shape)).reshape(w_shape)
    b = np.linspace(-0.1, 0.2, num=3)

    conv = Conv(3, 3, 4, 4, 2, 1)
    conv.params['w']['param'] = w
    conv.params['b']['param'] = b
    out = conv(x)
    # The correct output
    correct_out = np.array([[[[-0.08759809, -0.10987781],
                               [-0.18387192, -0.2109216 ]],
                              [[ 0.21027089,  0.21661097],
                               [ 0.22847626,  0.23004637]],
                              [[ 0.50813986,  0.54309974],
                               [ 0.64082444,  0.67101435]]],
                             [[[-0.98053589, -1.03143541],
                               [-1.19128892, -1.24695841]],
                              [[ 0.69108355,  0.66880383],
                               [ 0.59480972,  0.56776003]],
                              [[ 2.36270298,  2.36904306],
                               [ 2.38090835,  2.38247847]]]])

    print('Testing conv_forward')
    print('difference: ', rel_error(out, correct_out))


# check pooling backpropagation is correct
if check_pool_back:
    np.random.seed(231)
    x = np.random.randn(3, 2, 8, 8)
    dout = np.random.randn(3, 2, 4, 4)

    pool = MaxPool(kernel_size=2, stride=2, padding=0)
    out = pool(x)

    dx = pool.backward(dout, x)

    dx_num = eval_numerical_gradient_array(pool, x, dout)

    # Your error should be around 1e-12
    print('Testing pooling backward:')
    print('dx error: ', rel_error(dx, dx_num))

# check convolutional backpropogation is correct:
if check_conv_back:
    x = np.random.randn(2, 3, 16, 16)
    w = np.random.randn(3, 3, 3, 3)
    b = np.random.randn(3, 1)
    dout = np.random.randn(2, 3, 14, 14)
    conv = Conv(in_channels=3, out_channels=3, height=3, width=3, stride=1, padding=0)
    conv.params['b']['param'] = b
    conv.params['w']['param'] = w # 此行为后添加的
    out = conv(x)
    dx = conv.backward(dout, x)

    dx_num = eval_numerical_gradient_array(conv, x, dout)

    params = conv.params


    def fw(v):
        tmp = params['w']['param']
        params['w']['param'] = v
        f_w = conv(x)
        params['w']['param'] = tmp
        return f_w


    dw = params['w']['grad']
    dw_num = eval_numerical_gradient_array(fw, w, dout)

    db = params['b']['grad']


    def fb(v):
        tmp = params['b']['param']
        params['b']['param'] = v
        f_b = conv(x)
        params['b']['param'] = tmp
        return f_b


    db_num = eval_numerical_gradient_array(fb, b, dout)

    print('Testing conv')
    print('dx error: ', rel_error(dx_num, dx))
    print('dw error: ', rel_error(dw_num, dw))
    print('db error: ', rel_error(db_num, db))


# TODO: write script to check the backpropagation on the whole CNN is correct
if check_cnn_back:
    inputs=2
    input_dim=(3,16,16)
    hidden_units=10
    num_classes=10
    num_filters=3
    filter_size=3
    pool_size=2

    np.random.seed(231)
    X = np.random.randn(inputs, *input_dim)
    y = np.random.randint(num_classes, size=inputs)

    model = CNN(input_dim, num_filters, filter_size, pool_size, hidden_units, num_classes)
    loss, score = model.oracle(X, y)
    print('loss:', loss)
    a=['w1','w2','w3']
    b=['b1','b2','b3']

    for param_name in sorted(a):
        f = lambda _: model.oracle(X, y)[0]
        param_grad_num = eval_numerical_gradient(f, model.param_groups['w'][param_name]['param'], verbose=False, h=0.00001)
        e = rel_error(param_grad_num, model.param_groups['w'][param_name]['grad'])
        print('%s relative error: %e' % (param_name, rel_error(param_grad_num, model.param_groups['w'][param_name]['grad'])))

    for param_name in sorted(b):
        f = lambda _: model.oracle(X, y)[0]
        param_grad_num = eval_numerical_gradient(f, model.param_groups['b'][param_name]['param'], verbose=False, h=0.00001)
        e = rel_error(param_grad_num, model.param_groups['b'][param_name]['grad'])
        print('%s relative error: %e' % (param_name, rel_error(param_grad_num, model.param_groups['b'][param_name]['grad'])))
