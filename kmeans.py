import numpy as np
import random as r

def initializeCentroids (X,k):
    centroids = np.empty((k,X.shape[1]),np.dtype('float32'))
    indices = r.sample(range(X.shape[0]),k)
    for i in range(k):
        centroids[i,:] = X[indices[i],:]
    return centroids

def findClosestCentroids (X,centroids):
    k = centroids.shape[0]
    distances = np.empty((X.shape[0],k),np.dtype('float32'))
    for i in range(k):
        distances[:,i] = np.sum(np.power(X-centroids[i,:], 2), axis=1)
    closest = np.argmin(distances, axis=1)
    return closest

def updateCentroids (X,k,closest):
    centroids = np.empty((k,X.shape[1]),np.dtype('float32'))
    for i in range(k):
        inCluster = np.equal(closest,i)
        if np.sum(inCluster) != 0:
            centroids[i,:] = np.sum(np.transpose(np.multiply(np.transpose(X),inCluster)), axis=0)/np.sum(inCluster)
        else:
            centroids[i,:] = X[r.sample(range(X.shape[0]),1),:]
    return centroids
