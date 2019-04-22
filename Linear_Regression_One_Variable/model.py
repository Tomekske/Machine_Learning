import tensorflow as tf
import numpy as np
import math
import csv
import time
from sklearn import cross_validation

def csvContent(filename):
    '''Get content of the csv file
    
    Args:
        filename (string): Path to the csv file

    Return: 
        (list): Returns a list of a converted dictionary with data columns
    '''

    csvFile = open(filename)

    return list(csv.DictReader(csvFile))

def HeaderData(csvList,header):
    '''Gets the column of a certain header
    
    Args:
        csvList (list): Converted dictionary list containing all the columns and rows
        header (string): Header containing the associated data

    Return:
        (numpy array): Returns a numpy array containing the data
    '''

    data = []

    for row in csvList:
        data.append(float(row[header]))

    return np.asarray(data)

def ProgramExecutionTime(start):
    '''Output program execution time
    
    Args:
        start (time): Program start time
    '''

    currentTime = time.time() - start

    if currentTime < 60:
        print(f'Execution time: {currentTime:0.2f} sec ')
    else:
        currentTime = currentTime / 60
        print(f'Execution time: {currentTime:0.2f} min')

def Hypothesis(X,W,b):
    '''Calculates the hypothesis using the lineair regression formula 'X * W + B'
    
    Args:
        X (numpy array): Original x-values
        W (numpy array): Weight of the equation (slope)
        b (numpy array): Bias of the equation (offset)
    Return:
        (Tensor): Y-value for fiven X-value
    '''

    return tf.matmul(X, W) + b    

def CostFunction(hx, Y):
    '''Calculates the cost function of the linear equation '1/n * sum((pred - y)^2)' 
    
    Args:
        hx (numpy array): Predicted Y-values
        Y (numpy array): Actual Y-values
    Return:
        (Tensor): Cost function value
    '''

    return  tf.reduce_mean(tf.square(hx - Y))
#Program start time
startTime = time.time()

#Retrieve all csv column data
csvList = csvContent('Linear_Regression_One_Variable/Dataset/simple.csv')

#Get column data
featuresX = HeaderData(csvList, 'x')
featuresY = HeaderData(csvList, 'y')

#Split train data into train and test data
trainX, testX, trainY, testY = cross_validation.train_test_split(featuresX,featuresY, test_size=0.7, random_state=42)

#A placeholder is simply a variable that we will assign data to at a later date. 
#It allows us to create our operations and build our computation graph, without needing the data.
X = tf.placeholder(tf.float32, [None, 1])
Y = tf.placeholder(tf.float32,  [None, 1])

#Create and initialize variables with zeros
W = tf.Variable(tf.zeros([1,1]), name = "weights")
B = tf.Variable(tf.zeros([1]), name = "bias")

#Calculate the hypothesis with random values
pred = Hypothesis(X,W,B)
#Calculate the cost function with the prediction of the random values
cost = CostFunction(pred, Y)

#print program execution time
ProgramExecutionTime(startTime)
