import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import time
import random

from sklearn import linear_model as lm
from sklearn import decomposition
from sklearn import neighbors

from PIL import Image


def getLabelImage(adress):
    image = mpimg.imread(adress)

    image = image.tolist()
    labels = []

    for y in image:
        column = []
        for x in y:
            if (x[0] > 253) and (x[1] < 2) and (x[1] < 2):
                column.append(True)
            else:
                column.append(False)
            
        labels.append(column)
    
    return np.array(labels)
    
def getImage(adress):
    image = mpimg.imread(adress)
    return np.array(image,dtype ='float64')
    
#take one image in array form and array of labels and convert it into a form that can be passed to the sklearn classification algorithms
def makeTrainingSubset(labelImageArray, imageArray,radius):
    print('Creating data subset...')
    time1 = time.clock()
    x,y = labelImageArray.shape
    if (imageArray.shape != (x,y,3)):
        raise Exception 
    
    labels = []
    trainingSet = []
    for i in range(0,x):
        print(i/x *100,'%complete',sep='')
        for j in range(0,y):
            trainingSet.append(getSubset(imageArray,i,j,radius).flatten())
            labels.append(labelImageArray[i,j])
            
    time2 = time.clock()
    print('Runtime:',time2-time1)
    print()
    return (np.array(trainingSet,dtype='float64'),np.array(labelImageArray).flatten())
            

#take a list of adresses of (label,image) tuples and retrive the images then convert them into a usable dataset
def makeTrainingSet(adresses,radius):
    data = zeros((0,radius^2*12))
    for adress in adresses:
        labelAdress,imageAdress = adress
        label = getLabelImage(labelAdress)
        image = getImage(imageAdress)
        X,Y,a = image.shape
        print('Image Dimensions:',image.shape)
        print('Adress:',imageAdress)
        
        data.append(makeTrainingSubset(label,image,radius))
        
    pass
    

#create an object that can tranform the data to reduce the dimensionality while preseving most of the variance    
def learnPCA(data,components):
    print('Learning data...')
    
    time1 = time.clock()
    pca = decomposition.PCA()
    #pca.set_params(kernel=)
    time2 = time.clock()
    
    pca = pca.fit(data)
    
    print('Runtime',time2-time1)
    print()
    
    return pca

def makeClassifier(trainingData):
    print('Making Classifier...')
    time1 = time.clock()
    
    x,y = trainingData
    classifier = lm.BayesianRidge()
    classifier.fit(x,y)
    
    time2 = time.clock()
    print('Classification time:',time2-time1)
    print()
    
    return classifier

#retrive a subset of an image centered around a point    
def getSubset(array,x,y,radius):
    maxX,maxY,a = array.shape
    
    returnArray = np.zeros((2*radius,2*radius,3),dtype='float64')
    
    #print(returnArray.shape)
    
    for i in range(x-radius,x+radius):
        for j in range(y-radius,y+radius):
            if (i < 0 or i >= maxX or j < 0 or j >= maxY):
                returnArray[i+radius-x,j+radius-y] = [0,0,0]
            else:
                returnArray[i+radius-x,j+radius-y] = array[i,j]
    
    return returnArray
    
#Remove all but a certain fraction of the data
def thinData(data,ratio):
    x,y = data
    print('Initial data:',x.shape,y.shape)
    x = x.tolist()
    y = y.tolist()
    length = len(y)
    
    newX = []
    newY = []
    
    for i in range(0,length):
        if random.random() < ratio:
            newX.append(x[i])
            newY.append(y[i])
    
    newX = np.array(newX)
    newY = np.array(newY)
    
    print('New data:',newX.shape,newY.shape)
    print()
    
    return (newX,newY)





