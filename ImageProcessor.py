import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import ImageProcessingFunctions as ipf
import time
from PIL import Image

#retrive the image and the labeled image
labels = ipf.getLabelImage('/Users/curtishuebner/Documents/Development/CDHImageProccessing/nasa forest fire images copy/1.jpg')
image = ipf.getImage('/Users/curtishuebner/Documents/Development/CDHImageProccessing/nasa forest fire images/1.jpg')

X,Y,a = image.shape

print('Image Dimensions:',image.shape)
print()

data = ipf.makeTrainingSubset(labels,image,1)

thinData = ipf.thinData(data,1)

"""pca = ipf.learnPCA(thinData[0],40)

thinData = (pca.transform(thinData[0]),thinData[1])"""


clf = ipf.makeClassifier(thinData)

#retrive predictions
print('Predicting Data...')

time1 = time.clock()
predictions = clf.predict(data[0])
time2 = time.clock()

print('Prediction time:',time2-time1)

#convert the 1-D array of predictions into something that can be plotted
heatMap = []

for i in range(0,X):
    heatMap.append([])
    for j in range(0,Y):
        heatMap[i].append(predictions[i*Y+j])
        
a = 5 % 3
        
heatMap = np.array(heatMap)

print('Heatmap shape:', heatMap.shape)

#Display the results of the data
plt.imshow(heatMap)
plt.show()
"""dim = 115

array = np.zeros((dim,dim,3))
x,y,a = array.shape

for i in range(0,x):
    for j in range(0,y):
        array[i][j] = [i+j,i+j,i+j]


print(ipf.getSubset(array,112,112,3))"""
        