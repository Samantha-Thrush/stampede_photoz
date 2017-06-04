'''
If you are using a GPU, write the following in ~/.theanorc.
[global]
device=gpu
floatX=float32
[blas]
ldflags=-lopenblas
[cuda]
root=/opt/apps/cuda/7.0
[nvcc]
fastmath=True
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
import time
import cPickle as pickle

import numpy as np

import theano
import theano.tensor as T

import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from lasagne.objectives import squared_error
from lasagne.objectives import categorical_crossentropy
from lasagne.nonlinearities import leaky_rectify
from lasagne.init import Orthogonal, Constant
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import BatchIterator
from lasagne.nonlinearities import softmax

from sklearn import neighbors
from sklearn.cross_validation import train_test_split
from sklearn.utils import check_random_state

from astropy.io import fits
from astropy import wcs

import bmc

X_init = np.load("/work/04323/thrush/data/train/X.npy")
y_init = np.load("/work/04323/thrush/data/train/y.npy")

X_small = X_init[:70000] #Added to cut down run time
y_small = y_init[:70000] #This too

def renormalize(array): 
    return (array - array.min()) /(array.max() - array.min())

for i in range(5):
    X_small[:,i,:,:] = renormalize(X_small[:,i,:,:])

y_small = renormalize(y_small)

print("X_init type = {}, y_init type = {}".format(type(X_small), type(y_small)))

X, X_test, y, y_test = train_test_split(X_small, y_small, test_size = 0.2, random_state=24)

print("X.shape = {}, X.min = {}, X.max = {}".format(X.shape, X.min(), X.max()))

print("y.shape = {}, y.min = {}, y.max = {}".format(y.shape, y.min(), y.max()))

#def renormalize(array):
#    return (array - array.min()) / (array.max() - array.min())

#for i in range(5):
#    X[:, i, :, :] = renormalize(X[:, i, :, :])

X = X.astype(np.float32)
    
y = y.astype(np.float32)

print("X.shape = {}, X.min = {}, X.max = {}".format(X.shape, X.min(), X.max()))
print("y.shape = {}, y.min = {}, y.max = {}".format(y.shape, y.min(), y.max()))

def compute_PCA(array):

    nimages0, nchannels0, height0, width0 = array.shape
    rolled = np.transpose(array, (0, 2, 3, 1))
    # transpose from N x channels x height x width  to  N x height x width x channels
    nimages1, height1, width1, nchannels1 = rolled.shape
    # check shapes
    assert nimages0 == nimages1
    assert nchannels0 == nchannels1
    assert height0 == height1
    assert width0 == width1
    # flatten
    reshaped = rolled.reshape(nimages1 * height1 * width1, nchannels1)
    
    from sklearn.decomposition import PCA
    
    pca = PCA()
    pca.fit(reshaped)
    
    cov = pca.get_covariance()
    
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    
    return eigenvalues, eigenvectors


class AugmentedBatchIterator(BatchIterator):
    
    def __init__(self, batch_size, crop_size=8, testing=False):
        super(AugmentedBatchIterator, self).__init__(batch_size)
        self.crop_size = crop_size
        self.testing = testing

    def transform(self, Xb, yb):

        Xb, yb = super(AugmentedBatchIterator, self).transform(Xb, yb)
        batch_size, nchannels, width, height = Xb.shape
        
        if self.testing:
            if self.crop_size % 2 == 0:
                right = left = self.crop_size // 2
            else:
                right = self.crop_size // 2
                left = self.crop_size // 2 + 1
            X_new = Xb[:, :, right: -left, right: -left]
            return X_new, yb

        eigenvalues, eigenvectors = compute_PCA(Xb)

        # Flip half of the images horizontally at random
        indices = np.random.choice(batch_size, batch_size // 2, replace=False)        
        Xb[indices] = Xb[indices, :, :, ::-1]

        # Crop images
        X_new = np.zeros(
            (batch_size, nchannels, width - self.crop_size, height - self.crop_size),
            dtype=np.float32
        )

        for i in range(batch_size):
            # Choose x, y pixel posiitions at random
            px, py = np.random.choice(self.crop_size, size=2)
                
            sx = slice(px, px + width - self.crop_size)
            sy = slice(py, py + height - self.crop_size)
            
            # Rotate 0, 90, 180, or 270 degrees at random
            nrotate = np.random.choice(4)
            
            # add random color perturbation
            alpha = np.random.normal(loc=0.0, scale=0.5, size=5)
            noise = np.dot(eigenvectors, np.transpose(alpha * eigenvalues))
            
            for j in range(nchannels):
                X_new[i, j] = np.rot90(Xb[i, j, sx, sy] + noise[j], k=nrotate)
                
        return X_new, yb


class SaveParams(object):

    def __init__(self, name):
        self.name = name

    def __call__(self, nn, train_history):
        if train_history[-1]["valid_loss_best"]:
            nn.save_params_to("{}.params".format(self.name))
            with open("{}.history".format(self.name), "wb") as f:
                pickle.dump(train_history, f)

class UpdateLearningRate(object):

    def __init__(self, start=0.001, stop=0.0001):
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = np.float32(self.ls[epoch - 1])
        getattr(nn, "update_learning_rate").set_value(new_value)

class TrainSplit(object):

    def __init__(self, eval_size):
        self.eval_size = eval_size

    def __call__(self, X, y, net):
        if self.eval_size:
            X_train, y_train = X[:-self.eval_size], y[:-self.eval_size]
            X_valid, y_valid = X[-self.eval_size:], y[-self.eval_size:]
        else:
            X_train, y_train = X, y
            X_valid, y_valid = _sldict(X, slice(len(y), None)), y[len(y):]

        return X_train, X_valid, y_train, y_valid

net = NeuralNet(
    layers=[
        ('input', layers.InputLayer),

        ('conv11', layers.Conv2DLayer),
        ('pool1', layers.MaxPool2DLayer),

        ('conv21', layers.Conv2DLayer),
        ('conv22', layers.Conv2DLayer),
        ('pool2', layers.MaxPool2DLayer),

        ('conv31', layers.Conv2DLayer),
        ('conv32', layers.Conv2DLayer),
        ('pool3', layers.MaxPool2DLayer),

        ('hidden4', layers.DenseLayer),
        ('dropout4', layers.DropoutLayer),
        
        ('hidden5', layers.DenseLayer),
        ('dropout5', layers.DropoutLayer),

        ('output', layers.DenseLayer),
        ],
    input_shape=(None, 5, 44, 44),
    
    conv11_num_filters=32, conv11_filter_size=(5, 5), 
    pool1_pool_size=(2, 2),

    conv21_num_filters=64, conv21_filter_size=(3, 3),
    conv22_num_filters=64, conv22_filter_size=(3, 3),
    pool2_pool_size=(2, 2),

    conv31_num_filters=128, conv31_filter_size=(3, 3),
    conv32_num_filters=128, conv32_filter_size=(3, 3),
    pool3_pool_size=(2, 2),

    hidden4_num_units=2048,
    dropout4_p=0.5,
    
    hidden5_num_units=2048,
    dropout5_p=0.5,

    output_num_units=1,
    output_nonlinearity=None,

    update_learning_rate=0.0001,
    update_momentum=0.9,

    objective_loss_function=squared_error,
    regression=True,
    max_epochs=1000,
    batch_iterator_train=AugmentedBatchIterator(batch_size=128, crop_size=4),
    batch_iterator_test=AugmentedBatchIterator(batch_size=128, crop_size=4, testing=True),
    
    on_epoch_finished=[SaveParams("net")],

    verbose=2,
    )


net.fit(X, y)

train_loss = [row['train_loss'] for row in net.train_history_]
valid_loss = [row['valid_loss'] for row in net.train_history_]

np.save("/work/04323/thrush/train_loss.npy", train_loss)
np.save("/work/04323/thrush/valid_loss.npy", valid_loss)
            
best_valid_loss = min([row['valid_loss'] for row in net.train_history_])
print("Best valid loss: {}".format(best_valid_loss))

#for i in range(5):
#    X_test[:, i, :, :] = (X_test[:,i,:,:] - X[:,i,:,:].min())/(X[:,i,:,:].max() - X[:,i,:,:].min())#renormalize(X_test[:, i, :, :])

X_test = X_test.astype(np.float32)
#y_test = (y_test - y.min())/(y.max() - y.min())     #renormalize(y_test).astype(np.float32)

y_test = y_test.astype(np.float32)
y_pred = net.predict(X_test)

np.save("/work/04323/thrush/sdss_convnet_pred.npy", y_pred)
np.save("/work/04323/thrush/y_test.npy",y_test)

from sklearn.metrics import mean_squared_error
print("Mean squared error:")
print(mean_squared_error(y_test, y_pred))

from sklearn.metrics import r2_score
print("R2 score:")
print(r2_score(y_test, y_pred))

from sklearn.metrics import median_absolute_error
print("Median absolute error:")
print(median_absolute_error(y_test, y_pred))
print("Testing set done.")
