import numpy as np
import random as r

#Scales the features between 0 and 1 along the row (axis=1) or column (axis=0)
def scale_features (X,axis):
    epsilon = 0.0001
    if axis == 0:
        return (X-X.min(0))/((X-X.min(0)).max(0)+epsilon)
    elif axis == 1:
        return (X-X.min(1).reshape((X.shape[0],1)))/((X-X.min(1).reshape((X.shape[0],1))).max(1).reshape((X.shape[0],1))+epsilon)

#Add a feature to the data of the time in the video (represented by row index in the tensor)
def add_position_feature (X,weight):
    return np.append(X, weight/X.shape[0]*np.arange(X.shape[0]).reshape(X.shape[0],1), 1)

def initialize_centroids (X,k):
    centroids = np.empty((k,X.shape[1]),np.dtype('float32'))
    indices = r.sample(range(X.shape[0]),k)
    for i in range(k):
        centroids[i,:] = X[indices[i],:]
    return centroids

def find_closest_centroids (X,centroids):
    k = centroids.shape[0]
    distances = np.empty((X.shape[0],k),np.dtype('float32'))
    for i in range(k):
        distances[:,i] = np.sum(np.power(X-centroids[i,:], 2), axis=1)
    closest_centroids = np.argmin(distances, axis=1)
    return closest_centroids

def update_centroids (X,k,closest):
    centroids = np.empty((k,X.shape[1]),np.dtype('float32'))
    for i in range(k):
        in_cluster = np.equal(closest,i)
        if np.sum(in_cluster) != 0:
            centroids[i,:] = np.sum(np.transpose(np.multiply(np.transpose(X),in_cluster)), axis=0)/np.sum(in_cluster)
        else:
            centroids[i,:] = X[r.sample(range(X.shape[0]),1),:]
    return centroids

def find_closest_points (X,centroids):
    k = centroids.shape[0]
    distances = np.empty((X.shape[0],k),np.dtype('float32'))
    for i in range(k):
        distances[:,i] = np.sum(np.power(X-centroids[i,:], 2), axis=1)
    closest_points = np.argmin(distances, axis=0)
    return closest_points
