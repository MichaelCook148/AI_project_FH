# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 12:08:06 2019

@author: Callon
"""
import sys
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp
tf.disable_eager_execution()
import bokeh
import bokeh.io
import bokeh.plotting
import bokeh.models
from IPython.display import display, HTML
bokeh.io.output_notebook(hide_banner=True)

tfd = tfp.distributions
psd_kernels = tfp.positive_semidefinite_kernels

np.random.seed(42)
tf.set_random_seed(42)


from csvReadIn import FeatureVectors


'''
Prepare to train model by splitting features into training and testing sets
and setting a random seed for reproducibility

'''
featuresGen2 = FeatureVectors("2019-11-12", True)
featuresTrain1, yTrain1 = featuresGen2.generateFeatureVectors();

featuresGen3 = FeatureVectors("2019-11-11", True)
featuresTrain2, yTrain2 = featuresGen3.generateFeatureVectors();

featuresGen4 = FeatureVectors("2019-11-10", True)
featuresTrain3, yTrain3 = featuresGen4.generateFeatureVectors();

featuresGen5 = FeatureVectors("2019-11-09", True)
featuresTrain4, yTrain4 = featuresGen5.generateFeatureVectors();

featuresGen6 = FeatureVectors("2019-11-03", True)
featuresTrain5, yTrain5 = featuresGen6.generateFeatureVectors();

featuresGen7 = FeatureVectors("2019-11-08", True)
featuresTrain6, yTrain6 = featuresGen7.generateFeatureVectors();

featuresGen8 = FeatureVectors("2019-11-07", True)
featuresTrain7, yTrain7 = featuresGen8.generateFeatureVectors();

featuresGen9 = FeatureVectors("2019-11-06", True)
featuresTrain8, yTrain8 = featuresGen9.generateFeatureVectors();

featuresGen10 = FeatureVectors("2019-11-05", True)
featuresTrain9, yTrain9 = featuresGen10.generateFeatureVectors();

featuresGen11= FeatureVectors("2019-11-04", True)
featuresTrain10, yTrain10 = featuresGen11.generateFeatureVectors();

featuresGen12= FeatureVectors("2019-11-13", True)
featuresTrain11, yTrain11 = featuresGen12.generateFeatureVectors();

print("Test Data")
featureGen = FeatureVectors("2019-11-14", True)
featuresTest, yTest = featureGen.generateFeatureVectors()

#Need to extract player names from feature vectors
test_players = []
train_players1 = []
train_players2 = []
train_players3 = []
train_players4 = []
train_players5 = []
train_players6 = []
train_players7 = []
train_players8 = []
train_players9 = []
train_players10 = []
train_players11 = []
for row in featuresTest:
    test_players.append(row[0])
    del row[0]

for row in featuresTrain1:
    train_players1.append(row[0])
    del row[0]
    
for row in featuresTrain2:
    train_players2.append(row[0])
    del row[0]
    
for row in featuresTrain3:
    train_players3.append(row[0])
    del row[0]
    
for row in featuresTrain4:
    train_players4.append(row[0])
    del row[0]
    
for row in featuresTrain5:
    train_players5.append(row[0])
    del row[0]

for row in featuresTrain6:
    train_players6.append(row[0])
    del row[0]

for row in featuresTrain7:
    train_players7.append(row[0])
    del row[0]
    
for row in featuresTrain8:
    train_players8.append(row[0])
    del row[0]
    
for row in featuresTrain9:
    train_players9.append(row[0])
    del row[0]
    
for row in featuresTrain10:
    train_players10.append(row[0])
    del row[0]
    
for row in featuresTrain11:
    train_players11.append(row[0])
    del row[0]


featuresTrain1 = featuresTrain1[:len(yTrain1)]
featuresTrain2 = featuresTrain2[:len(yTrain2)]
featuresTrain3 = featuresTrain3[:len(yTrain3)]
featuresTrain4 = featuresTrain4[:len(yTrain4)]
featuresTrain5 = featuresTrain5[:len(yTrain5)]
featuresTrain6 = featuresTrain6[:len(yTrain6)]
featuresTrain7 = featuresTrain7[:len(yTrain7)]
featuresTrain8 = featuresTrain8[:len(yTrain8)]
featuresTrain9 = featuresTrain9[:len(yTrain9)]
featuresTrain10 = featuresTrain10[:len(yTrain10)]
featuresTrain11 = featuresTrain11[:len(yTrain11)]


np_xTrain1 = np.array(featuresTrain1)
np_xTrain2 = np.array(featuresTrain2)
np_xTrain3 = np.array(featuresTrain3)
np_xTrain4 = np.array(featuresTrain4)
np_xTrain5 = np.array(featuresTrain5)
np_xTrain6 = np.array(featuresTrain6)
np_xTrain7 = np.array(featuresTrain7)
np_xTrain8 = np.array(featuresTrain8)
np_xTrain9 = np.array(featuresTrain9)
np_xTrain10 = np.array(featuresTrain10)
np_xTrain11 = np.array(featuresTrain11)
np_xTest = np.array(featuresTest)

np_yTrain1 = np.array(yTrain1)
np_yTrain2 = np.array(yTrain2)
np_yTrain3 = np.array(yTrain3)
np_yTrain4 = np.array(yTrain4)
np_yTrain5 = np.array(yTrain5)
np_yTrain6 = np.array(yTrain6)
np_yTrain7 = np.array(yTrain7)
np_yTrain8 = np.array(yTrain8)
np_yTrain9 = np.array(yTrain9)
np_yTrain10 = np.array(yTrain10)
np_yTrain11 = np.array(yTrain11)
np_yTest = np.array(yTest)


np_yTrain1 = np_yTrain1.ravel()
np_yTrain2 = np_yTrain2.ravel()
np_yTrain3 = np_yTrain3.ravel()
np_yTrain4 = np_yTrain4.ravel()
np_yTrain5 = np_yTrain5.ravel()
np_yTrain6 = np_yTrain6.ravel()
np_yTrain7 = np_yTrain7.ravel()
np_yTrain8 = np_yTrain8.ravel()
np_yTrain9 = np_yTrain9.ravel()
np_yTrain10 = np_yTrain10.ravel()
np_yTrain11 = np_yTrain11.ravel()
np_yTest = np_yTest.ravel()

np_xTrain = np.concatenate((np_xTrain1, np_xTrain2, np_xTrain3, np_xTrain4, np_xTrain5, np_xTrain6, np_xTrain7, np_xTrain8, np_xTrain9, np_xTrain10, np_xTrain11))
np_yTrain = np.concatenate((np_yTrain1, np_yTrain2, np_yTrain3, np_yTrain4, np_yTrain5, np_yTrain6, np_yTrain7, np_yTrain8, np_yTrain9, np_yTrain10, np_yTrain11))

np_xTrain1 = np_xTrain
np_xTest1 = np_xTest

np_xTrain = np.reshape(np_xTrain, (np_xTrain.shape[0],1, np_xTrain.shape[1]))
np_xTest = np.reshape(np_xTest, (np_xTest.shape[0], 1, np_xTest.shape[1]))

print(np_xTrain.shape)
print(np_xTest.shape)


np_xTrain = np_xTrain.astype(float)
np_yTrain = np_yTrain.astype(float)
np_xTest = np_xTest.astype(float)

'''
Build Model
'''

# Define mean function which is the means of observations
observations_mean = tf.constant(
    [np.mean(np_xTrain)], dtype=tf.float64)
mean_fn = lambda _: observations_mean

observation_noise_variance = 0
# Define the kernel with trainable parameters. 
# Note we transform some of the trainable variables to ensure
#  they stay positive.

# Use float64 because this means that the kernel matrix will have 
#  less numerical issues when computing the Cholesky decomposition

# Smooth kernel hyperparameters
smooth_amplitude = tf.exp(tf.Variable(
    np.float64(3)), name='smooth_amplitude')
smooth_length_scale = tf.exp(tf.Variable(
    np.float64(3)), name='smooth_length_scale')
# Smooth kernel
smooth_kernel = psd_kernels.ExponentiatedQuadratic(
    amplitude=smooth_amplitude, 
    length_scale=smooth_length_scale)

# Local periodic kernel hyperparameters
periodic_amplitude = tf.exp(tf.Variable(
    np.float64(0)), name='periodic_amplitude')
periodic_length_scale = tf.exp(tf.Variable(
    np.float64(1)), name='periodic_length_scale')
periodic_period = tf.exp(tf.Variable(
    np.float64(0)), name='periodic_period')
periodic_local_length_scale = tf.exp(tf.Variable(
    np.float64(2)), name='periodic_local_length_scale')
# Local periodic kernel
local_periodic_kernel = (
    psd_kernels.ExpSinSquared(
        amplitude=periodic_amplitude, 
        length_scale=periodic_length_scale,
        period=periodic_period) * 
    psd_kernels.ExponentiatedQuadratic(
        length_scale=periodic_local_length_scale))

# Short-medium term irregularities kernel hyperparameters
irregular_amplitude = tf.exp(tf.Variable(
    np.float64(0)), name='irregular_amplitude')
irregular_length_scale = tf.exp(tf.Variable(
    np.float64(0)), name='irregular_length_scale')
irregular_scale_mixture = tf.exp(tf.Variable(
    np.float64(0)), name='irregular_scale_mixture')
# Short-medium term irregularities kernel
irregular_kernel = psd_kernels.RationalQuadratic(
    amplitude=irregular_amplitude,
    length_scale=irregular_length_scale,
    scale_mixture_rate=irregular_scale_mixture)


kernel = (smooth_kernel + local_periodic_kernel + irregular_kernel)


# Start session
session = tf.Session()
session.run(tf.global_variables_initializer())


#
print(mean_fn(np_xTrain))
print(mean_fn(np_xTest))

np_xTrain = np_xTrain1
np_xTest = np_xTest1
np_xTrain = np_xTrain.astype(float)
np_yTrain = np_yTrain.astype(float)
np_xTest = np_xTest.astype(float)


# Posterior GP using fitted kernel and observed data
gp_posterior_predict = tfd.GaussianProcessRegressionModel(
    mean_fn=mean_fn,
    kernel=kernel,
    index_points=np_xTest,
    observation_index_points=np_xTrain,
    observations=np_yTrain,
    observation_noise_variance=observation_noise_variance)

# Posterior mean and standard deviation
posterior_mean_predict = gp_posterior_predict.mean()
posterior_std_predict = gp_posterior_predict.stddev()

m = session.run(posterior_std_predict)
print(m)