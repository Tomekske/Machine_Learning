import tensorflow as tf
import numpy as np
import math
import csv
import time
from sklearn import cross_validation

def ProgramExecutionTime(start):
    '''Output program execution time
    
    Args:
        start (time): Program start time
    '''

    currentTime = time.time() - start

    if currentTime < 59:
        print(f'Execution time: {currentTime:0.2f} sec ')
    else:
        currentTime = currentTime / 60
        print(f'Execution time: {currentTime:0.2f} min')

def Hypothesis(W,B,X):
    '''Calculates the hypothesis using the lineair regression formula 'X * W + B'
    
    Args:
        W (numpy array): Weight of the equation (slope)
        b (numpy array): Bias of the equation (offset)
        X (numpy array): Original x-values

    Return:
        (Tensor): Y-value for fiven X-value
    '''

    return W[0] * X[0] + W[1] * X[1] + W[2] * X[2] + B


def TransposeData(dataSet):
    '''Transpose the data sets [data, features]
    
    Args:
        dataSet (numpy array): Array of data points
    Return:
        (numpy array): Returns parsed data
    '''

    # Create multi demonsional array of the training data
    dataArray = np.array([dataSet])

    return tf.cast(np.transpose(dataArray), tf.float32)
# Program start time
startTime = time.time()

# Clears the default graph stack and resets the global default graph.
tf.reset_default_graph()

# Gets an existing variable with these parameters 
W = tf.get_variable("weights", shape=[3,1], initializer = tf.zeros_initializer)
B = tf.get_variable("bias", shape=[1], initializer = tf.zeros_initializer)

# X-values
dataset = [3.0, 1180.0, 64.0]

# Restores variables
restoreSaver = tf.train.Saver()

with tf.Session() as sess:
    # Restore variables from disk.
    restoreSaver.restore(sess, "Linear_Regression_Multi_Variable/Model/")

    # Evaluate existing variables
    weights = W.eval()
    bias = B.eval()

    print(f'weight: {weights}')
    print(f'Bias: {bias}')

    # Calculate Y-values for given X-values
    price = Hypothesis(weights,bias,dataset)

    print(f'PRED: {price}')

# print program execution time
ProgramExecutionTime(startTime)
