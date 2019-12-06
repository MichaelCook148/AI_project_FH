# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 16:02:55 2019

@author: michael
"""

import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import math
#import tensorflow_probability as tfp
#tf.disable_eager_execution()


#tfd = tfp.distributions
#psd_kernels = tfp.positive_semidefinite_kernels

np.random.seed(101)
tf.set_random_seed(101)

#trying to find which center to play
#picking 4 Centers that play on the test day
center_list = []
#data is stored as [player_name, location on game day, acutal value, prediction value, [x data set]]
center_list.append(["Aleksander Barkov", -1, -50, -50, []])   #Florida
center_list.append(["Steven Stamkos", -1, -50, -50, []])    #Tampa Bay
center_list.append(["Nazem Kadri", -1, -50, -50, []])  #Colorado
center_list.append(["Brett Howden", -1, -50, -50, []])  #New York R

center_list.append(["Sebastian Aho", -1, -50, -50, []])   #Carolina
center_list.append(["Jack Eichel", -1, -50, -50, []])    #Buffalo
center_list.append(["Adam Lowry", -1, -50, -50, []])  #Winnipeg
center_list.append(["Derek Stepan", -1, -50, -50, []])  #Arizona

center_list.append(["Mikko Koivu", -1, -50, -50, []])   #Minnesota
center_list.append(["Connor McDavid", -1, -50, -50, []])    #Edmonton
center_list.append(["Joe Pavelski", -1, -50, -50, []])  #Dallas
center_list.append(["Bo Horvat", -1, -50, -50, []])  #Vancover

center_list.append(["Logan Couture", -1, -50, -50, []])   #San Jose
center_list.append(["Ryan Getzlaf", -1, -50, -50, []])    #Anaheim
center_list.append(["Dylan Larkin", -1, -50, -50, []])  #Detroit
center_list.append(["Anze Kopitar", -1, -50, -50, []])  #LA



for play in center_list:
    print(play[0], " is in spot ", play[1])

#Going to need to make a new read in value
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

featuresGen6 = FeatureVectors("2019-11-13", True)
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

j=0
#going to take the first 100 players of that night and compare there values. going to get there names first
top_100 = []

for row in featuresTest:
    test_players.append(row[0])
    if j<=100:
        top_100.append([row[0], -50, -50])
    for play in center_list:
        if row[0] == play[0]:
            play[1] = j
    del row[0]
    j = j+ 1

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
np_yTest = np_yTest.ravel()

np_xTrain = np.concatenate((np_xTrain1, np_xTrain2, np_xTrain3, np_xTrain4, np_xTrain5, np_xTrain6, np_xTrain7, np_xTrain8, np_xTrain9, np_xTrain10))
np_yTrain = np.concatenate((np_yTrain1, np_yTrain2, np_yTrain3, np_yTrain4, np_yTrain5, np_yTrain6, np_yTrain7, np_yTrain8, np_yTrain9, np_yTrain10))
np_yTrain = np.reshape(np_yTrain, (np_yTrain.shape[0], 1))
np_yTest = np.reshape(np_yTest, (np_yTest.shape[0],1))
#np.append(np_yTrain[1], [8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8])
#print(np_xTrain.shape)
#print(np_yTrain.shape)
#print()
#print(np_xTrain[1])
#print(np_yTrain[1])
#i = 0
#print(np_yTest.shape)
#print(np_xTest[1])
#x_unit_test = (np_xTest[61])
#for tt in x_unit_test:
    #print(tt)
#x_unit_test = np.reshape(x_unit_test, (1,16))
#print(x_unit_test)
#y_unit_test = np.reshape([8], (1,1))
for center in center_list:  
    i =0
    for play in range(len(np_yTest)):
        if i== center[1]:
            center[2] = np_yTest[play][0]
            center[4] = np_xTest[play]
        i = i +1


for play in center_list:
    print(play[0], " is in spot ", play[1], " with actual value ", play[2])



learning_rate = 0.01
t_epochs = 100
disp_step = 50

num_samp = np_xTrain.shape[0]

X0 = tf.placeholder("float")
X1 = tf.placeholder("float")
X2 = tf.placeholder("float")
X3 = tf.placeholder("float")
X4 = tf.placeholder("float")
X5 = tf.placeholder("float")
X6 = tf.placeholder("float")
X7 = tf.placeholder("float")
X8 = tf.placeholder("float")
X9 = tf.placeholder("float")
X10 = tf.placeholder("float")
X11 = tf.placeholder("float")
X12 = tf.placeholder("float")
X13 = tf.placeholder("float")
X14 = tf.placeholder("float")
X15 = tf.placeholder("float")

Y = tf.placeholder("float")

rng = np.random
W0 = tf.Variable(rng.random(), name = "weight0")
W1 = tf.Variable(rng.random(), name = "weight1")
W2 = tf.Variable(rng.random(), name = "weight2")
W3 = tf.Variable(rng.random(), name = "weight3")
W4 = tf.Variable(rng.random(), name = "weight4")
W5 = tf.Variable(rng.random(), name = "weight5")
W6 = tf.Variable(rng.random(), name = "weight6")
W7 = tf.Variable(rng.random(), name = "weight7")
W8 = tf.Variable(rng.random(), name = "weight8")
W9 = tf.Variable(rng.random(), name = "weight9")
W10 = tf.Variable(rng.random(), name = "weight10")
W11 = tf.Variable(rng.random(), name = "weight11")
W12 = tf.Variable(rng.random(), name = "weight12")
W13 = tf.Variable(rng.random(), name = "weight13")
W14 = tf.Variable(rng.random(), name = "weight14")
W15 = tf.Variable(rng.random(), name = "weight15")


b = tf.Variable(rng.random(), name = "bias")

pred = tf.add_n([tf.multiply(X0,W0), tf.multiply(X1,W1) , tf.multiply(X2,W2) , tf.multiply(X3,W3) , tf.multiply(X4,W4) ,tf.multiply(X5,W5) , tf.multiply(X6,W6) , tf.multiply(X7,W7) , tf.multiply(X8,W8) , tf.multiply(X9,W9) , tf.multiply(X10,W10) ,tf.multiply(X11,W11) ,tf.multiply(X12,W12) ,tf.multiply(X13,W13) ,tf.multiply(X14,W14) , tf.multiply(X15,W15) ,b])

cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*num_samp)
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    
    sess.run(init)
    
    for epoch in range(t_epochs):
        print("epoch # ", epoch)
        for(x,y) in zip(np_xTrain, np_yTrain):
            sess.run(optimizer, feed_dict={X0: x[0], X1: x[1],X2: x[2],X3: x[3],X4: x[4],X5: x[5],X6: x[6],X7: x[7],X8: x[8],X9: x[9],X10: x[10],X11: x[11],X12: x[12],X13: x[13],X14: x[14],X15: x[15], Y: y})
            
        #if(epoch +1) % disp_step == 0:
            

    print("train step done")
    
    
    
    
    
    for play in center_list:
        total = 0
        total = sess.run(W0)*play[4][0] + sess.run(W1)*play[4][1] + sess.run(W2)*play[4][2] + sess.run(W3)*play[4][3] + sess.run(W4)*play[4][4] + sess.run(W5)*play[4][5] +sess.run(W6)*play[4][6] +sess.run(W7)*play[4][7] +sess.run(W8)*play[4][8] + sess.run(W9)*play[4][9] + sess.run(W10)*play[4][10] + sess.run(W11)*play[4][11] + sess.run(W12)*play[4][12] + sess.run(W13)*play[4][13] + sess.run(W14)*play[4][14] + sess.run(W15)*play[4][15]  +sess.run(b)
        play[3]= total 
        print(play[0], " with prediction ", play[3]," with actual value ", play[2])
    
    absolute_total_diff =0
    total_diff = 0
    for i in range(100):
        top_100[i][1]= np_yTest[i][0]
        top_100[i][2] = sess.run(W0)*np_xTest[i][0] + sess.run(W1)*np_xTest[i][1] + sess.run(W2)*np_xTest[i][2] + sess.run(W3)*np_xTest[i][3] + sess.run(W4)*np_xTest[i][4] + sess.run(W5)*np_xTest[i][5] +sess.run(W6)*np_xTest[i][6] +sess.run(W7)*np_xTest[i][7] +sess.run(W8)*np_xTest[i][8] + sess.run(W9)*np_xTest[i][9] + sess.run(W10)*np_xTest[i][10] + sess.run(W11)*np_xTest[i][11] + sess.run(W12)*np_xTest[i][12] + sess.run(W13)*np_xTest[i][13] + sess.run(W14)*np_xTest[i][14] + sess.run(W15)*np_xTest[i][15]  +sess.run(b)
        total_diff = total_diff + ((top_100[i][1]-top_100[i][2]) * (top_100[i][1]-top_100[i][2])) #sum of actual - predict
        absolute_total_diff = absolute_total_diff + abs(top_100[i][1]-top_100[i][2])
        
    avg_diff = total_diff/100
    root_avg_diff = math.sqrt(avg_diff)
    abs_avg_diff = absolute_total_diff/100
    print()
    print("Squared Error = ", total_diff)
    print("Mean Squared Error = ", avg_diff)
    print("Root Mean Squared Error = ", root_avg_diff)
    print("Absolute Error = ", absolute_total_diff)
    print("Mean Absolute Error = ", abs_avg_diff)
    print()
    
    #from csvReadIn
    '''
    New format:
            [Name, G(season), A(season), +/-(season), PIM(season), PPP(season), SOG(season), 
            BLK(season), G(last game), A(last game), +/-(last game), PIM(last game), PPP(last game), SOG(last game), 
            BLK(last game), Team RK, Opponent RK]
    '''
    print("Breakdown of weights as follows:")
    print("Goals (Season) Weight = ", sess.run(W0))
    print("Assists (Season) Weight = ", sess.run(W1))
    print("+/- (Season) Weight = ", sess.run(W2))
    print("PIM (Season) Weight = ", sess.run(W3))
    print("PPP (Season) Weight = ", sess.run(W4))
    print("SOG (Season) Weight = ", sess.run(W5))
    print("BLK (Season) Weight = ", sess.run(W6))
    print("Goals (Last Game) Weight = ", sess.run(W7))
    print("Assists (Last Game) Weight = ", sess.run(W8))
    print("+/- (Last Game) Weight = ", sess.run(W9))
    print("PIM (Last Game) Weight = ", sess.run(W10))
    print("PPP (Last Game) Weight = ", sess.run(W11))
    print("SOG (Last Game) Weight = ", sess.run(W12))
    print("BLK (Last Game) Weight = ", sess.run(W13))
    print("Team Rank Weight = ", sess.run(W14))
    print("Opponent Rank Weight = ", sess.run(W15))
    print("Basic Bias = ", sess.run(b))

print("done")