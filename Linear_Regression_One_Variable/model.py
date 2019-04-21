import tensorflow as tf
import numpy as np
import math
import csv
import time

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
        (numpy array) Returns a numpy array containing the data
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

#Program start time
startTime = time.time()

#Retrieve all csv column data
csvList = csvContent('Linear_Regression_One_Variable/Dataset/simple.csv')

#Get column data
originalX = HeaderData(csvList, 'x')
originalY = HeaderData(csvList, 'y')

#print program execution time
ProgramExecutionTime(startTime)