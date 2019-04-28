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

# Program start time
startTime = time.time()

# Clears the default graph stack and resets the global default graph.
tf.reset_default_graph()

# Gets an existing variable with these parameters 
W = tf.get_variable("weights", shape=[1,1], initializer = tf.zeros_initializer)
B = tf.get_variable("bias", shape=[1], initializer = tf.zeros_initializer)

# Restores variables
restoreSaver = tf.train.Saver()

with tf.Session() as sess:
    # Restore variables from disk.
    restoreSaver.restore(sess, "Linear_Regression_One_Variable/Model/")

    # Evaluate existing variables
    weights = W.eval()
    bias = B.eval()

    print(f'weight: {weights}')
    print(f'Bias: {bias}')

# print program execution time
ProgramExecutionTime(startTime)