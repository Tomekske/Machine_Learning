import tensorflow as tf
import numpy as np
import math
import csv
import time
import datetime
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

    if currentTime < 59:
        print(f'Execution time: {currentTime:0.2f} sec ')
    else:
        currentTime = currentTime / 60
        print(f'Execution time: {currentTime:0.2f} min')

def SplitData(dataset, testbatch):
    '''Splits data into a training and test data set
    
    Args:
        dataset (numpy array): Array of X-features
        testbatch (float): Test batch ratio rate
    Return:
        (numpy array): Returns XY-training and XY-test sets
    '''

    return TransposeData(*cross_validation.train_test_split(*dataset, test_size=testbatch, random_state=42))

def TransposeData(*dataset):
    '''Transpose the data sets [data, features]
    
    Args:
        dataset (numpy array): Array of the full dataset
    Return:
        (numpy array): Returns XY-training and XY-test sets
    '''

    testList = []
    trainList = []
    yTestList = []
    yTrainList = []

    # Slice test data from the dataset
    testSetList = list(dataset)[::2]
    # Slice training data from the dataset
    trainSetList = list(dataset)[1:][::2]

    # Split data set accordingly 
    for test,train in zip(testSetList[:-1],trainSetList[:-1]):
        testList.append(np.array(test))
        trainList.append(np.array(train))
    
    # Obtain only Y-values from the dataset
    for test,train in zip(testSetList[-1:],trainSetList[-1:]):
        yTestList.append(np.array(test))
        yTrainList.append(np.array(train))

    testSet = np.asarray(testList)
    trainSet = np.asarray(trainList)

    yTestSet = np.asarray(yTestList)
    yTrainSet = np.asarray(yTrainList)

    return np.transpose(testSet), np.transpose(trainSet), np.transpose(yTestSet), np.transpose(yTrainSet)

def CalculateHouseAge(yearBuild):
    '''Calculate the house's age
    
    Args:
        yearBuild (float): The year the house was build
    '''

    agesList = []

    # Obtain the current year
    currentYear = datetime.datetime.now().strftime("%Y")
    
    # Calculate the current age of the house
    for year in yearBuild:
        agesList.append(float(currentYear) - float(year))

    return np.asarray(agesList)

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


# Program start time
startTime = time.time()
year = datetime.datetime.now().strftime("%Y")

# Retrieve all csv column data
csvList = csvContent('Linear_Regression_Multi_Variable/Dataset/housePrices.csv')

# Get column data
featureBedrooms = HeaderData(csvList, 'bedrooms')
featureArea = HeaderData(csvList, 'sqft_living')
featureBuild = HeaderData(csvList, 'yr_built')
featurePrices = HeaderData(csvList, 'price')
featureAge= CalculateHouseAge(featureBuild)

dataset = []

# Append the features to the dataset list
dataset.append(featureBedrooms)
dataset.append(featureArea)
dataset.append(featureAge)
dataset.append(featurePrices)

# Split the data in to training and test data
testSet, trainSet, yTestSet, yTrainSet = SplitData(dataset, 0.01)

# A placeholder is simply a variable that we will assign data to at a later date. 
# It allows us to create our operations and build our computation graph, without needing the data.
X = tf.placeholder(tf.float32, [None, 3])
Y = tf.placeholder(tf.float32,  [None, 1])

# Create and initialize variables with zeros
W = tf.Variable(tf.zeros([3,1]), name = "weights")
B = tf.Variable(tf.zeros([1]), name = "bias")

# Calculate the hypothesis with random values
pred = Hypothesis(W,B,X)
# Calculate the cost function with the prediction of the random values
cost = CostFunction(pred, Y)

# Minimise cost function parameters
learningRate = 0.0000000001
epochs = 10000

# Optimise cost function
optimizer = tf.train.GradientDescentOptimizer(learningRate).minimize(cost)

# An Op that initializes global variables in the graph
init = tf.global_variables_initializer()

# Model save location
saveModel = "Linear_Regression_Multi_Variable/Model/"

# Saves variables
trainSaver = tf.train.Saver()

with tf.Session() as sesh:
    sesh.run(init)

    for epoch in range(epochs):
        # Minimise cost function
        sesh.run(optimizer, feed_dict= { X: trainSet, Y: yTrainSet})
        
        # Calculate cost, weight and bias
        c =  sesh.run(cost, feed_dict = {X: trainSet, Y: yTrainSet})
        w = sesh.run(W)
        b = sesh.run(B)

        print(f'epoch: {epoch} c: {c:.6f} w: {w} b: {b}')

    # Get weight and bias
    weight = sesh.run(W)
    bias = sesh.run(B)

    # Assign variables
    W.assign(weight).op.run()
    B.assign(bias).op.run()

    # Save the trained model
    trainSaver.save(sess = sesh, save_path= saveModel)

    print(f'Weight = {weight}\nBias = {bias}')

# print program execution time
ProgramExecutionTime(startTime)
