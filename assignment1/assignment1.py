import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.linalg import norm


def train(wd, n_hidden, n_iters, learning_rate, momentum_mul, do_early_stopping=False, minibatch_size=10):
    """
    a simple multilayer neural net for multiclass classification
    :param wd: weight_decay 
    :param n_hidden: number of units in hidden layer
    :param n_iters: number of sgd iteration
    :param learning_rate: 
    :param momentum_mul: velocity damping factor
    :param do_early_stopping: True if early_stopping is done, where we simply report the best past solution
    :param minibatch_size: size of minibatch in sgd
    :return: classification loss on the datasets
    """
    data_file = loadmat('data.mat', squeeze_me=True, struct_as_record=False)
    data = data_file['data']

    data_train = {'X': data.training.inputs, 'y': data.training.targets}
    data_valid = {'X': data.validation.inputs, 'y': data.validation.targets}
    data_test = {'X': data.test.inputs, 'y': data.test.targets}
    n_train = data_train['X'].shape[1]

    # initialize model
    params = initial_model(n_hidden)
    theta = model2theta(params)

    test_gradient(params, data_train, wd)
    # initialize velocity
    v = 0

    loss_train = []
    loss_valid = []
    best = {}

    if do_early_stopping:
        best['theta'] = 0
        best['loss_valid'] = np.inf
        best['iter'] = -1

    for t in range(n_iters + 1):
        batch_start = (t * minibatch_size) % n_train
        data_batch = {
            'X': data_train['X'][:, batch_start:batch_start + minibatch_size],
            'y': data_train['y'][:, batch_start:batch_start + minibatch_size]
        }

        # classical momentum
        loss, grad = eval_obj_grad(theta2model(theta), data_batch, wd)

        grad_vec = model2theta(grad)

        v = momentum_mul * v - grad_vec

        theta += learning_rate * v

        # todo Nesterov's accelerated method

        params = theta2model(theta)

        loss = eval_obj(params, data_train, wd)
        loss_train.append(loss)
        loss = eval_obj(params, data_valid, wd)
        loss_valid.append(loss)

        if do_early_stopping and loss_valid[-1] < best['loss_valid']:
            best['theta'] = theta.copy()
            best['loss_valid'] = loss_valid[-1]
            best['iter'] = t

        if t % (max(1, n_iters // 10)) == 0:
            print('After %d iterations, ||theta|| %.3e, training loss is %.2e, and validation loss is %.2e\n' % (
            t, norm(theta), loss_train[-1],
            loss_valid[-1]))

    test_gradient(params, data_train, wd)

    plt.close()
    plt.figure()

    plt.plot(loss_train, label='training loss')
    plt.plot(loss_valid, label='validation loss')
    plt.legend(loc='best')
    plt.show()

    if do_early_stopping:
        print("Early stopping: validation loss: %.3e,  was lowest after %d iterations" % (
        best['loss_valid'], best['iter']))
        theta = best['theta']

    params = theta2model(theta)
    # examine performance
    datasets = [data_train, data_valid, data_test]

    acc = [accuracy(params, x) for x in datasets]

    classification_loss = [eval_obj(params, x, 0) for x in datasets]

    print("Accuracy: training %.3e, validation %.3e, testing %.3e" % (acc[0], acc[1], acc[2]))
    info = {
        'loss_train': classification_loss[0],
        'loss_valid': classification_loss[1],
        'loss_test': classification_loss[2]
    }
    return info


def eval_obj(params, data, wd):
    # W_hid, b_hid, W_out, b_out = params['W_hid'], params['b_hid'], params['W_out'], params['b_out']
    W_hid, W_out = params['W_hid'], params['W_out']

    # todo implement the forward propagation
    loss = 0
    return loss


def eval_obj_grad(params, data, wd):
    """
    compute loss and gradient of model
    :param params: 
                    W_hid
                    W_out
    :param data:

    """

    # todo implement the forward propagation

    loss = 0

    # todo implement the backward prapagation
    n_hidden = 100
    grad_W_out = np.zeros(10, n_hidden)
    grad_W_hid = np.zeros(n_hidden, 256)

    grad = {'W_out': grad_W_out,
            'W_hid': grad_W_hid,
            }

    return loss, grad


def initial_model(n_hid):
    n_params = (256 + 10) * n_hid
    as_row_vector = np.cos(np.arange(n_params))
    params = {}
    params['W_hid'] = as_row_vector[:256 * n_hid].reshape((n_hid, 256)) * 0.1
    params['W_out'] = as_row_vector[256 * n_hid:].reshape((10, n_hid)) * 0.1
    return params


def test_gradient(params, data, wd):
    loss, analytic_grad = eval_obj_grad(params, data, wd)

    num_checks = 100
    theta = model2theta(params)
    grad_ana = model2theta(analytic_grad)

    delta = 1e-4
    threshold = 1e-5

    for i in range(num_checks):
        ind = (i * 1299283) % theta.size
        grad_ind_ana = grad_ana[ind]

        theta1 = theta.copy()
        theta1[ind] += delta
        l1 = eval_obj(theta2model(theta1), data, wd)

        theta2 = theta.copy()
        theta2[ind] -= delta
        l2 = eval_obj(theta2model(theta2), data, wd)

        grad_ind_fin = (l1 - l2) / (2 * delta)
        diff = abs(grad_ind_ana - grad_ind_fin)
        if diff < threshold:
            continue
        if diff / (abs(grad_ind_ana) + abs(grad_ind_fin)) < threshold:
            continue
        raise AssertionError('%d-th: l %.3e, l1 %.3e, l2 %.3e, analytic %.3e, fd %.3e, diff %.3e\n'
                             % (i, loss, l1, l2, grad_ind_ana, grad_ind_fin, diff))
    print('Gradient test passed')


def model2theta(params):
    """
    convert model parameters into vector form
    :param params: 
    :return: 
    """
    theta = np.concatenate((params['W_out'].flatten(), params['W_hid'].flatten()))
    return theta


def theta2model(theta):
    """
    convert vector form into model parameters
    :param theta: 
    :return: 
    """
    n_hid = theta.size // (256 + 10)
    params = {}
    params['W_out'] = np.reshape(theta[:n_hid * 10], (10, n_hid))
    params['W_hid'] = np.reshape(theta[n_hid * 10:], (n_hid, 256))
    return params


def accuracy(params, data):
    W_hid, W_out = params['W_hid'], params['W_out']

    # indices of class label
    # class_indices = np.nonzero(data['y'])
    index_transpose = np.nonzero(data['y'].T)
    true_label = index_transpose[1]
    # forward propagation
    a_hidden = W_hid.dot(data['X'])
    h_hidden = sigmoid(a_hidden)

    a_out = W_out.dot(h_hidden)

    pred = a_out.argmax(axis=0)

    return np.mean(pred == true_label)


def log_sum_exp(x):
    """
    compute log(sum(exp(a), 0)), should return a n-dim vector
    :param x: p*n matrix
    """
    # todo implement the log column sum of exp(x)
    return 0


def sigmoid(input):
    return 1 / (1 + np.exp(-input))
