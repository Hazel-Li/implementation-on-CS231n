
# coding: utf-8

# In[509]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import loadmat
from scipy.linalg import norm


# In[555]:


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
        #loss, grad = eval_obj_grad(theta2model(theta), data_batch, wd)
        #grad_vec = model2theta(grad)
        #v = momentum_mul * v - grad_vec
        #theta += learning_rate * v

        #todo Nesterov's accelerated method
        theta_nes = theta + momentum_mul * v
        loss, grad = eval_obj_grad(theta2model(theta_nes), data_batch, wd)
        grad_vec = model2theta(grad)
        v = momentum_mul * v - grad_vec
        theta += learning_rate * v
        
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
    
    loss = 0

    z_hid = W_hid.dot(data['X'])
    a_hid = sigmoid(z_hid)
    z_out = W_out.dot(a_hid)
    y_predict = np.exp(z_out - log_sum_exp(z_out))

    cross_entropy = 0
    cross_entropy += - np.log(y_predict) * data['y']  
    
    weight_decay=0
    weight_decay=wd/2*(np.square(np.linalg.norm(W_out))+np.square(np.linalg.norm(W_hid)))

    loss =  np.sum(cross_entropy)/data['y'].shape[1] + weight_decay /data['X'].shape[1]

    return loss


def eval_obj_grad(params, data, wd):
    """
    compute loss and gradient of model
    :param params: 
                    W_hid
                    W_out
    :param data:

    """
    W_hid, W_out = params['W_hid'], params['W_out']
    
    loss = 0

    z_hid = W_hid.dot(data['X'])
    a_hid = sigmoid(z_hid)
    z_out = W_out.dot(a_hid)
    y_predict = np.exp(z_out - log_sum_exp(z_out))

    cross_entropy = 0
    cross_entropy += - np.log(y_predict) * data['y']  
    
    weight_decay=0
    weight_decay=wd/2*(np.square(np.linalg.norm(W_out))+np.square(np.linalg.norm(W_hid)))

    loss =  np.sum(cross_entropy)/data['y'].shape[1] + weight_decay /data['X'].shape[1]
    
    # todo implement the backward prapagation
    n_hidden=100
    grad_W_out = np.zeros((10, n_hidden))
    grad_W_hid = np.zeros((n_hidden, 256))

    error_out = (y_predict - data['y']) / data['y'].shape[1]
    dw2 = np.dot(error_out, a_hid.T)
    grad_W_out = dw2 + wd * W_out / data['X'].shape[1]

    error_hid = a_hid * (1 - a_hid) * np.dot(W_out.T, error_out)
    dw1 = np.dot(error_hid, data['X'].T) 
    grad_W_hid = dw1 + wd * W_hid / data['X'].shape[1]
    
    grad = {'W_out': grad_W_out,
            'W_hid': grad_W_hid,}

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
    lse_x= np.log(np.sum(np.exp(x), axis = 0))
    
    return lse_x

def sigmoid(input):
    return 1 / (1 + np.exp(-input))


# In[548]:


train(0, 10, 70, 0.005, 0, False, 4)


# In[549]:


learning_array=[0.002, 0.01, 0.05, 0.2, 1.0, 5.0]
df1 = pd.DataFrame(columns=["loss_train",'loss_valid','loss_test'])


# In[550]:


for lr in learning_array:
    info=train(0, 10, 70, lr, 0, False, 4)
    loss_info=pd.DataFrame([[info['loss_train'],info['loss_valid'],info['loss_test']]],columns=["loss_train",'loss_valid','loss_test'])
    df1=df1.append(loss_info,ignore_index=True) 


# In[551]:


df1.index=['0.002', '0.01', '0.05', '0.2', '1.0', '5.0']
print("The best performance on validation data is under the learning rate 1.0")
df1


# In[552]:


learning_array2=[0.01, 0.05, 0.2, 1.0, 5.0]
momentum_array=[0,0.5,0.9]
n_hidden =10
n_iters = 100
df2 = pd.DataFrame(columns=["loss_train",'loss_valid','loss_test'])
for lr1 in learning_array2:
    for mm in momentum_array:
        info=train(0, n_hidden, n_iters, lr1, mm, do_early_stopping=False, minibatch_size=10)
        loss_info=pd.DataFrame([[info['loss_train'],info['loss_valid'],info['loss_test']]],columns=["loss_train",'loss_valid','loss_test'])
        df2=df2.append(loss_info,ignore_index=True) 


# In[553]:


df2.index=['l1m1', 'l1m2', 'l1m3', 'l2m1', 'l2m2', 'l2m3','l3m1', 'l3m2', 'l3m3','l4m1', 'l4m2', 'l4m3','l5m1', 'l5m2', 'l5m3']
print("The best performance on validation data is under the learning rate 1.0 and the momentum mul 0.5")
df2


# In[554]:


learning_array2=[0.01, 0.05, 0.2, 1.0, 5.0]
momentum_array=[0,0.5,0.9]
n_hidden =10
n_iters = 100
df3 = pd.DataFrame(columns=["loss_train",'loss_valid','loss_test'])
for lr1 in learning_array2:
    for mm in momentum_array:
        info=train(0, n_hidden, n_iters, lr1, mm, do_early_stopping=False, minibatch_size=10)
        loss_info=pd.DataFrame([[info['loss_train'],info['loss_valid'],info['loss_test']]],columns=["loss_train",'loss_valid','loss_test'])
        df3=df3.append(loss_info,ignore_index=True) 


# In[506]:


df3.index=['l1m1', 'l1m2', 'l1m3', 'l2m1', 'l2m2', 'l2m3','l3m1', 'l3m2', 'l3m3','l4m1', 'l4m2', 'l4m3','l5m1', 'l5m2', 'l5m3']
print("As for Nesterov Accelerated Method, the best performance on validation data is under the learning rate 1.0 and the momentum mul 0.5.The result shows that the best parameter is the same and Nesterov gives a better performance.")
df3


# In[507]:


print("Part3: Given that nesterov method has a better performance, we choose l3m3 and set momentum as 0.9.")


# In[511]:


n_hidden = 200
n_iters = 1000
learning_rate=0.2
wd_array=[0, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
df4 = pd.DataFrame(columns=["loss_train",'loss_valid','loss_test'])
for wd in wd_array:
    info=train(wd, n_hidden, n_iters, learning_rate, 0.9, do_early_stopping=True, minibatch_size=10)
    loss_info=pd.DataFrame([[info['loss_train'],info['loss_valid'],info['loss_test']]],columns=["loss_train",'loss_valid','loss_test'])
    df4=df4.append(loss_info,ignore_index=True) 


# In[514]:


df4.index=['0', '1e-4', '1e-3', '1e-2', '1e-1', '1', '10']
print("The best performance is under weight decay 1e-2.")
df4


# In[515]:


n_hidden = [10,50,100,200,300]
n_iters = 1000
learning_rate=0.2
earlystopping=[True,False]
wd = 0
df5 = pd.DataFrame(columns=["loss_train",'loss_valid','loss_test'])
for nh in n_hidden:
    for el in earlystopping:
        info=train(wd, nh, n_iters, learning_rate, 0.9, el, minibatch_size=10)
        loss_info=pd.DataFrame([[info['loss_train'],info['loss_valid'],info['loss_test']]],columns=["loss_train",'loss_valid','loss_test'])
        df5=df5.append(loss_info,ignore_index=True) 


# In[518]:


df5.index=['10_on','10_off','50_on','50_off','100_on','100_off','200_on','200_off','300_on','300_off']
print("The best performance is when using 100 hidden units and early stopping is on.")
df5


# In[521]:


print('Part3 Q3 Testing: Here we use model selection to find out the best parameters. First, let us take a look at Classical Momentum. We apply the best valid performance parameters given by part3: learning_rate=1.0 and momentum_mul=0.5.')


# In[522]:


n_hidden = [10,50,100,200,300]
n_iters = 1000
learning_rate=1.0
momentum_mul=0.5
weight_decay=[0, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
earlystopping=[True,False]
df6 = pd.DataFrame(columns=["loss_train",'loss_valid','loss_test'])
for el in earlystopping:
    for wd in weight_decay:
        for nh in n_hidden:
            info=train(wd, nh, n_iters, learning_rate, momentum_mul, el, minibatch_size=10)
            loss_info=pd.DataFrame([[info['loss_train'],info['loss_valid'],info['loss_test']]],columns=["loss_train",'loss_valid','loss_test'])
            df6=df6.append(loss_info,ignore_index=True) 


# In[538]:


mean_off=df6[(df6.index>=35)]['loss_valid'].mean()
mean_on=df6[(df6.index<35)]['loss_valid'].mean()
mean_on,mean_off


# In[539]:


print("Firstly, we should build our model with early stopping.")


# In[541]:


mean_nh1=df6[(df6.index<7)]['loss_valid'].mean()
mean_nh2=df6[(df6.index>=7)&(df6.index<14)]['loss_valid'].mean()
mean_nh3=df6[(df6.index>=14)&(df6.index<21)]['loss_valid'].mean()
mean_nh4=df6[(df6.index>=21)&(df6.index<28)]['loss_valid'].mean()
mean_nh5=df6[(df6.index>=28)&(df6.index<35)]['loss_valid'].mean()
mean_nh1,mean_nh2,mean_nh3,mean_nh4,mean_nh5


# In[542]:


print("Secondly, we should build our model with 50 hidden units")


# In[543]:


df6[(df6.index>=7)&(df6.index<14)]['loss_valid']


# In[545]:


print("Thirdly, we should build our model with weight decay le-1.")


# In[546]:


print('Then, let us take a look at Nesterov. We apply the best valid performance parameters given by part3: learning_rate=1.0 and momentum_mul=0.5.')


# In[556]:


n_hidden = [10,50,100,200,300]
n_iters = 1000
learning_rate=1.0
momentum_mul=0.5
weight_decay=[0, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
earlystopping=[True,False]
df7 = pd.DataFrame(columns=["loss_train",'loss_valid','loss_test'])
for el in earlystopping:
    for wd in weight_decay:
        for nh in n_hidden:
            info=train(wd, nh, n_iters, learning_rate, momentum_mul, el, minibatch_size=10)
            loss_info=pd.DataFrame([[info['loss_train'],info['loss_valid'],info['loss_test']]],columns=["loss_train",'loss_valid','loss_test'])
            df7=df7.append(loss_info,ignore_index=True) 


# In[557]:


mean_off=df7[(df7.index>=35)]['loss_valid'].mean()
mean_on=df7[(df7.index<35)]['loss_valid'].mean()
mean_on,mean_off


# In[558]:


print("Firstly, we should build our model with early stopping.")


# In[559]:


mean_nh1=df7[(df7.index<7)]['loss_valid'].mean()
mean_nh2=df7[(df7.index>=7)&(df7.index<14)]['loss_valid'].mean()
mean_nh3=df7[(df7.index>=14)&(df7.index<21)]['loss_valid'].mean()
mean_nh4=df7[(df7.index>=21)&(df7.index<28)]['loss_valid'].mean()
mean_nh5=df7[(df7.index>=28)&(df7.index<35)]['loss_valid'].mean()
mean_nh1,mean_nh2,mean_nh3,mean_nh4,mean_nh5


# In[560]:


print("Secondly, we should build our model with 50 hidden units")


# In[561]:


df7[(df7.index>=7)&(df7.index<14)]['loss_valid']


# In[562]:


print("Thirdly, we should build our model with weight decay le-1.")


# In[563]:


print('And Nesterov has an overall better performance than classical momentum.')


# In[565]:


train(1e-1, 50, 1000, 1.0, 0.5, True, minibatch_size=10)

