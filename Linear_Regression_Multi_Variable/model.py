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

    # Slice test data from the dataset
    testSetList = list(dataset)[::2]
    # Slice training data from the dataset
    trainSetList = list(dataset)[1:][::2]

    # Split data set accordingly 
    for test,train in zip(testSetList,trainSetList):
        testList.append(np.array(test))
        trainList.append(np.array(train))

    # Convert list to a numpy array
    testSet = np.asarray(testList)
    trainSet = np.asarray(trainList)

    return testSet, trainSet

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
testSet, trainSet = SplitData(dataset, 0.3)

# print program execution time
ProgramExecutionTime(startTime)
