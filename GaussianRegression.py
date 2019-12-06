# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 12:08:06 2019

@author: Callon
"""
import math
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.keras.backend import set_session
import matplotlib.pyplot as plt

#define necessary tensorflow variables to be used later
tfd = tfp.distributions
W = tf.Variable(10)
sess = tf.compat.v1.Session()
set_session(sess)
psd_kernels = tfp.positive_semidefinite_kernels

from csvReadIn import FeatureVectors


'''
Prepare to train model by splitting features into training and testing sets

'''
#Use 10 previous game nights for the testing data
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

#Ensure that training sets and their y value sets are the same length
#There are some instances where some players who may be present in the training
#set may not have any last game stats and are therefore not included
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

#Transform training and test sets to numpy arrays
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

#unravel y val sets so that they're just 1-D numpy arrays
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

#Concatenate the training x and y val sets into two large training sets
np_xTrain = np.concatenate((np_xTrain1, np_xTrain2, np_xTrain3, np_xTrain4, np_xTrain5, np_xTrain6, np_xTrain7, np_xTrain8, np_xTrain9, np_xTrain10, np_xTrain11))
np_yTrain = np.concatenate((np_yTrain1, np_yTrain2, np_yTrain3, np_yTrain4, np_yTrain5, np_yTrain6, np_yTrain7, np_yTrain8, np_yTrain9, np_yTrain10, np_yTrain11))

print(np_xTrain.shape)
print(np_xTest.shape)

#transform train and test sets in type np.float64 to be able to work with model
np_xTrain = np_xTrain.astype(float)
np_yTrain = np_yTrain.astype(float)
np_xTest = np_xTest.astype(float)
np_yTest =  np_yTest.astype(float)


# Begin tensorflow session
# Model will be defined, trained, and tested in here
with sess.as_default():
    
    #Special kernel class to be used in model
    #Based on code found at:
    #https://gist.github.com/jburnim/83c67aca9e4f361b5a768f3fa951ddf1
    #This is required because it is stated that the model we will be using a 
    #layer with @property, which yields a PositiveSemidefiniteKernel instance at: 
    #https://rdrr.io/github/rstudio/tfprobability/man/layer_variational_gaussian_process.html
    class RBFKernelFn(tf.keras.layers.Layer):
        def __init__(self):
            super(RBFKernelFn, self).__init__()

            
            self._amplitude = self.add_weight(
                    initializer=tf.constant_initializer(0),
                    dtype=np_xTrain.dtype,
                    name='amplitude')
            
            self._length_scale = self.add_weight(
                    initializer=tf.constant_initializer(0),
                    dtype=np_xTrain.dtype,
                    name='length_scale')
            
        def call(self, x):
            # Never called -- this is just a layer so it can hold variables
            # in a way Keras understands.
            #print(dtype)
            return x
        
        @property
        def kernel(self):

            return tfp.math.psd_kernels.ExponentiatedQuadratic(amplitude=tf.nn.softplus(0.1 * self._amplitude),
                                                                            length_scale=tf.nn.softplus(5. * self._length_scale)
                                                                            ) 
            
    # For numeric stability, set the default floating-point dtype to float64
    tf.keras.backend.set_floatx('float64')
    
    #The initialization, compilation, fitting, and testing of the following model
    #heavily uses code written at:
    #https://medium.com/tensorflow/regression-with-probabilistic-layers-in-tensorflow-probability-e46ff5d37baf
    #See the section "Bonus: Tabula Rasa"
    
    # Create trainable inducing point locations and variational parameters.
    num_inducing_points = 40

    inducing_index_points = tf.Variable(
            np.linspace(-10., 10., num_inducing_points)[..., np.newaxis],
                dtype=np_xTrain.dtype, name='inducing_index_points')
    # Build model.
    model = tf.keras.Sequential([
            tf.keras.layers.Dense(1, kernel_initializer='ones', use_bias=False),
            tfp.layers.VariationalGaussianProcess(
                    num_inducing_points = num_inducing_points,
                    kernel_provider=RBFKernelFn(),
                    event_shape=[1],
                    inducing_index_points_initializer=tf.constant_initializer(
                            np.linspace(-50, 200, num=num_inducing_points,
                                        dtype=np_xTrain.dtype)[..., np.newaxis]),
                            unconstrained_observation_noise_variance_initializer=(
                                    tf.constant_initializer(
                                            np.log(np.expm1(1.)).astype(np_xTrain.dtype))),
                                    ),
    ])
    
    # Do inference.
    batch_size = 374
    loss = lambda y, rv_y: rv_y.variational_loss(
    y, kl_weight=np.array(batch_size, np_xTrain.dtype) / np_xTrain.shape[0])
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.01), loss=loss, metrics=['accuracy'])
    history = model.fit(np_xTrain, np_yTrain, batch_size=batch_size, validation_data=(np_xTest, np_yTest), epochs=1000)
    
    test = model.evaluate(np_xTest, np_yTest)
    print(test)
    
    yhat = model.predict(np_xTest)
    
    y = yhat.ravel()
    y = y.astype(int)
    #Print out predicted players from best to worst followed by actual players from best to worst
    sort_predict = sorted(y, reverse=True)
    sort_players = [x for _,x in sorted(zip(y,test_players), reverse=True)]

    for player, predict in zip(sort_players, sort_predict):
        print(player + ": " + str(predict))
    
    y_sort = sorted(np_yTest.ravel(), reverse=True)
    real_players = [x for _, x in sorted(zip(np_yTest, test_players), reverse=True)]

    for player, true in zip(real_players, y_sort):
        print(player + ": " + str(true))
    
    
    # Plot the accuracy and the loss on the training and testing set 
    # Print graph for training accuracy
    plt.plot(history.history['accuracy'])
    plt.title('model accuracy - training')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.show()
    #Print graph for testing accuracy
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy - testing')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.show()
    # Print graph with both training and testing loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # Calculate different types of error values on predictions vs. gold standards
    #Taken partially from Mike's code for his linear regression model
    absolute_total_diff =0
    total_diff = 0
    
    count = 0
    for predict in y:
        total_diff = total_diff + ((np_yTest[count]-predict) * (np_yTest[count]-predict)) #sum of actual - predict
        absolute_total_diff = absolute_total_diff + abs(np_yTest[count]-predict)
    
    avg_diff = total_diff/len(y)
    root_avg_diff = math.sqrt(avg_diff)
    abs_avg_diff = absolute_total_diff/len(y)
    print()
    print("Squared Error = ", total_diff)
    print("Mean Squared Error = ", avg_diff)
    print("Root Mean Squared Error = ", root_avg_diff)
    print("Absolute Error = ", absolute_total_diff)
    print("Mean Absolute Error = ", abs_avg_diff)
    print()
    
