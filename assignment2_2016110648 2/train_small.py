from __future__ import division, print_function, absolute_import
import numpy as np
from nn.optimizer import SGD
from nn.utils import accuracy
from dataset import get_cifar10_data
from nn.cnn import CNN


def train(model, X_train, y_train, X_val, y_val, batch_size, n_epochs, lr=1e-2,
          lr_decay=0.8, verbose=True, print_level=100):
    n_train = X_train.shape[0]
    iterations_per_epoch = max(n_train // batch_size, 1)
    n_iterations = n_epochs * iterations_per_epoch
    epoch = 0
    
    loss_hist = []

    # Define optimizer and set parameters
    opt_params = {'lr': 1e-3}
    sgd = SGD(model.param_groups, **opt_params)

    for t in range(n_iterations):
        batch_mask = np.random.choice(n_train, batch_size)
        X_batch = X_val[batch_mask]
        y_batch = y_val[batch_mask]        
        # Evaluate function value and gradient
        loss, score = model.oracle(X_batch, y_batch)
        loss_hist.append(loss)

        # Perform stochastic gradient descent

        sgd.step()

        # Maybe print training loss
        if verbose and t % print_level == 0:
            train_acc = accuracy(score, y_batch)
            print('(Iteration %d / %d , epoch %d) loss: %f, train_accu: %f' % (
                t + 1, n_iterations, epoch, loss_hist[-1], train_acc))

        # At the end of every epoch, adjust the learning rate.
        epoch_end = (t + 1) % iterations_per_epoch == 0
        if epoch_end:
            epoch += 1
            opt_params['lr'] *= lr_decay


if __name__ == '__main__':
    model = CNN()
    data = get_cifar10_data()
    num_train = 100
    data = {
        'X_train': data['X_train'][:num_train],
        'y_train': data['y_train'][:num_train],
        'X_val': data['X_val'],
        'y_val': data['y_val'],
    }
    X_train, y_train, X_val, y_val = data['X_train'], data['y_train'], data['X_val'], data['y_val']

    train(model, X_train, y_train, X_val, y_val, batch_size=50, n_epochs=50, print_level=1)
