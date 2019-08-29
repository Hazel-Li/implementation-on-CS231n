import math
import timeit
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from time import time
from dataset import load_cifar10

train_normal=False
train_improve=True

def get_cifar10_data(n_train=49000, n_val=1000, n_test=10000, subtract_mean=True):

    X_train, y_train, X_test, y_test = load_cifar10()

    # Subsample the data
    mask = list(range(n_train, n_train + n_val))
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = list(range(n_train))
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = list(range(n_test))
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    if subtract_mean:
        mean_image = np.mean(X_train, axis=0)
        X_train -= mean_image
        X_val -= mean_image
        X_test -= mean_image

    return X_train, y_train, X_val, y_val, X_test, y_test


X_train, y_train, X_val, y_val, X_test, y_test = get_cifar10_data()
print('Train data shape: ', X_train.shape)
print('Train labels shape: ', y_train.shape)
print('Validation data shape: ', X_val.shape)
print('Validation labels shape: ', y_val.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)


class Split(object):
    def __init__(self, x, y, batch_size, shuffle=True):
        self.x = x 
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        
    def __iter__(self):
        N = self.x.shape[0]
        B = self.batch_size
        idx = np.arange(N)
        if self.shuffle:
            np.random.shuffle(idx)
        return iter((self.x[i:i+B], self.y[i:i+B]) for i in range(0, N, B))
    
train_dset = Split(X_train, y_train, batch_size=64, shuffle=True)
val_dset = Split(X_val, y_val, batch_size=64, shuffle=False)
test_dset = Split(X_test, y_test, batch_size=64)


class CNN(tf.keras.Model):
    def __init__(self, channel, hidden_units, num_classes):
        super().__init__()
        initializer = tf.variance_scaling_initializer(scale=0.01)
        self.conv = tf.layers.Conv2D(filters=channel, kernel_size=(7, 7), strides=(1, 1),padding='valid', activation=tf.nn.relu, use_bias=True)
        self.pool = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        self.fc1 = tf.layers.Dense(hidden_units, activation=tf.nn.relu, kernel_initializer=initializer)
        self.fc2 = tf.layers.Dense(num_classes, kernel_initializer=initializer)
        
    def call(self, x, training=None):
        x = self.conv(x)
        x = self.pool(x)
        x = tf.layers.flatten(x)
        x = self.fc1(x)
        scores = self.fc2(x)
        return scores


def train(model_init, optimizer_init, num_epochs=1, GPUorNot=False):
    
    tf.reset_default_graph()
    
    if GPUorNot:
        device = '/device:GPU:0'
    else:
        device = '/cpu:0'
        
    with tf.device(device):
        x = tf.placeholder(tf.float32, [None, 32, 32, 3])
        y = tf.placeholder(tf.int32, [None])
        is_training = tf.placeholder(tf.bool, name='is_training')
        scores = model_init(x, is_training)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=scores)
        loss = tf.reduce_mean(loss)
        
        optimizer = optimizer_init()
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss)
            
    fig_loss = np.zeros([766])
            
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        t = 0
        for epoch in range(num_epochs):
            print('Starting epoch %d' % epoch)
            for x_np, y_np in train_dset:
                feed_dict = {x: x_np, y: y_np}
                loss_np, _ = sess.run([loss, train_op], feed_dict=feed_dict)
                fig_loss[t] = loss_np
                
                num_correct, num_samples = 0, 0
                for x_batch, y_batch in val_dset:
                    feed_dict = {x: x_batch, is_training: 0}
                    scores_np = sess.run(scores, feed_dict=feed_dict)
                    y_pred = scores_np.argmax(axis=1)
                    num_samples += x_batch.shape[0]
                    num_correct += (y_pred == y_batch).sum()
                    acc = float(num_correct) / num_samples
                    
                if t % 100 == 0:
                    print('Iteration %d, loss = %.4f, accuracy = %.2f%%' % (t, loss_np, 100 * acc))
                t += 1
                
if train_normal:

    learning_rate = 3e-3
    channel, hidden_units, num_classes = 32, 1024, 10

    def model_init(inputs, is_training):
        return CNN(channel, hidden_units, num_classes)(inputs)

    def optimizer_init():
        return tf.train.GradientDescentOptimizer(learning_rate)

    train(model_init, optimizer_init, GPUorNot = True)


class CNN_improve(tf.keras.Model):
    def __init__(self, channel1, channel2, hidden_units, num_classes):
        super().__init__()
        initializer = tf.variance_scaling_initializer(scale=0.01)
        bias_initializer = tf.initializers.random_normal(0,0.1)
        self.conv1 = tf.layers.Conv2D(filters=channel1, kernel_size=(3, 3), strides=(1, 1),padding='same', activation=tf.nn.relu, use_bias=True)
        self.pool1 = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')
        self.norm1 = tf.layers.BatchNormalization()
        self.conv2 = tf.layers.Conv2D(filters=channel2, kernel_size=(3, 3), strides=(1, 1),padding='same', activation=tf.nn.relu, use_bias=True)
        self.pool2 = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')
        self.norm2 = tf.layers.BatchNormalization()
        self.fc1 = tf.layers.Dense(hidden_units, activation=tf.nn.relu,use_bias= True, kernel_initializer=initializer,bias_initializer=bias_initializer)
        self.fc2 = tf.layers.Dense(num_classes, use_bias= True, kernel_initializer=initializer,bias_initializer=bias_initializer)
        
    def call(self, x, training=None):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.norm1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.norm2(x)
        x = tf.layers.flatten(x)
        x = self.fc1(x)
        scores = self.fc2(x)
        return scores


if train_improve:

    learning_rate = 3e-3
    channel1, channel2, hidden_units, num_classes = 32, 64, 1024, 10

    def model_init(inputs, is_training):
        return CNN_improve(channel1, channel2, hidden_units, num_classes)(inputs)

    def optimizer_init():
        return tf.train.GradientDescentOptimizer(learning_rate)

    train(model_init, optimizer_init, GPUorNot = True)


