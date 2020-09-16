import numpy as np
import matplotlib.pyplot as plt
import cv2
import kMeansFunctions as km
import videoProcessingFunctions as vp

X = np.array([[2,8],[9,7],[1,9],[4,1],[0,8],[6,2],[5,0],[5,1],[10,10],[8,10],[0,7],[1,6],[9,9],[8,8],[5,2]])
k = 3
maxIter = 5

centroids = km.initializeCentroids(X, k)
for i in range(maxIter):
    closest = km.findClosestCentroids(X, centroids)
    
    colors = ['red','green','blue']
    for i in range(k):
        Xtemp = np.transpose(np.multiply(np.transpose(X),np.equal(closest,i)))
        Xtemp = Xtemp[~np.all(Xtemp == 0, axis=1)]
        plt.scatter(Xtemp[:,0],Xtemp[:,1],color=colors[i],marker='o')
    plt.scatter(centroids[:,0],centroids[:,1],color='black',marker='+',s=100)
    plt.show()
    
    centroids = km.updateCentroids(X, k, closest)

################################################################################
#ground truth clusters [0 1 0 2 0 2 2 2 1 1 0 0 1 1 2]
#ground truth centroids [[5.  1.2]
#                        [8.8 8.8]
#                        [0.8 7.6]]
################################################################################
