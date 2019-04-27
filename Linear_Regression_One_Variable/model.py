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

def Hypothesis(W,B,X):
    '''Calculates the hypothesis using the lineair regression formula 'X * W + B'
    
    Args:
        W (numpy array): Weight of the equation (slope)
        b (numpy array): Bias of the equation (offset)
        X (numpy array): Original x-values

    Return:
        (Tensor): Y-value for fiven X-value
    '''

    return tf.matmul(X, W) + B    

def CostFunction(hx, Y):
    '''Calculates the cost function of the linear equation '1/n * sum((pred - y)^2)' 
    
    Args:
        hx (numpy array): Predicted Y-values
        Y (numpy array): Actual Y-values
    Return:
        (Tensor): Cost function value
    '''

    return  tf.reduce_mean(tf.square(hx - Y))

def SplitData(featuresX, featuresY, testbatch):
    '''Splits data into a training and test data set
    
    Args:
        featuresX (numpy array): Array of X-features
        featuresY (numpy array): Array of Y-features
        testbatch (float): Test batch ratio rate
    Return:
        (numpy array): Returns XY-training and XY-test sets
    '''

    trainX, testX, trainY, testY = cross_validation.train_test_split(featuresX,featuresY, test_size=testbatch, random_state=42)

    #Create a multidemensional array containing training data
    training = np.array([trainX, trainY])
    #Create a multidemensional array containing test data
    test =  np.array([testX, testY])

    return TransposeData(training, test)

def TransposeData(trainingSet, testSet):
    '''Transpose the data sets [data, features]
    
    Args:
        trainingSet (numpy array): Array of training sets
        testSet (numpy array): Array of test sets
    Return:
        (numpy array): Returns XY-training and XY-test sets
    '''

    #Split traing data into XY-arrays
    trainX,trainY = trainingSet
    #Split test data into XY-arrays
    testX,testY = testSet

    #Create multi demonsional array of the training data
    trainX = np.array([trainX])
    trainY = np.array([trainY])

    #Create multi demonsional array of the test data
    testX = np.array([testX])
    testY = np.array([testY])

    return np.transpose(trainX), np.transpose(testX), np.transpose(trainY), np.transpose(testY)

#Program start time
startTime = time.time()

#Retrieve all csv column data
csvList = csvContent('Linear_Regression_One_Variable/Dataset/simple.csv')

#Get column data
featuresX = HeaderData(csvList, 'x')
featuresY = HeaderData(csvList, 'y')

#Split data into training and test data (30% training data and 70% test data)
trainX, testX, trainY, testY = SplitData(featuresX, featuresY, 0.3)

#A placeholder is simply a variable that we will assign data to at a later date. 
#It allows us to create our operations and build our computation graph, without needing the data.
X = tf.placeholder(tf.float32, [None, 1])
Y = tf.placeholder(tf.float32,  [None, 1])

#Create and initialize variables with zeros
W = tf.Variable(tf.zeros([1,1]), name = "weights")
B = tf.Variable(tf.zeros([1]), name = "bias")

#Calculate the hypothesis with random values
pred = Hypothesis(W,B,X)
#Calculate the cost function with the prediction of the random values
cost = CostFunction(pred, Y)

#Minimise cost function parameters
learningRate = 0.000000001
epochs = 60

#Optimise cost function
optimizer = tf.train.GradientDescentOptimizer(learningRate).minimize(cost)
init = tf.initialize_all_variables()

#Save model
saveModel = "Linear_Regression_One_Variable/Model/"
trainSaver = tf.train.Saver()

with tf.Session() as sesh:
    sesh.run(init)

    # loop-over all epochs
    for epoch in range(epochs):
        #Minimise cost function
        sesh.run(optimizer, feed_dict= { X: trainX, Y: trainY})

        #Calculate cost, weight and bias
        c =  sesh.run(cost, feed_dict = {X: trainX, Y: trainY})
        w = sesh.run(W)
        b = sesh.run(B)

        print(f'epoch: {epoch} c: {c:.6f} w: {w} b: {b}')

    #Get weight and bias
    weight = sesh.run(W)
    bias = sesh.run(B)

    #Assign variables
    W.assign(weight).op.run()
    B.assign(bias).op.run()

    #Save the trained model
    trainSaver.save(sess = sesh, save_path= saveModel)

    print(f'Result: Weight = {weight}\nBias = {bias}')

#print program execution time
ProgramExecutionTime(startTime)
