import numpy as np
import random as r
import torch

device = torch.device('cuda')
dtype = torch.float32

#Scales the features between 0 and 1 along the row (axis=1) or column (axis=0)
def scale_features (X,axis):
    epsilon = torch.tensor(0.0001,dtype=torch.float32)
    if axis == 0:
        return (X-X.min(0).values)/((X-X.min(0).values).max(0).values+epsilon)
    elif axis == 1:
        return (X-X.min(1).values.reshape((X.shape[0],1)))/((X-X.min(1).values.reshape((X.shape[0],1))).max(1).values.reshape((X.shape[0],1))+epsilon)

#Add a feature to the data of the time in the video (represented by row index in the tensor)
def add_position_feature (X,weight):
    return torch.cat((X,weight*(torch.arange(X.shape[0],dtype=torch.float32).to(device).reshape((X.shape[0],1))/X.shape[0])),dim=1)

#Initialize the centroids to random data points return the initialized centroids
def initialize_centroids (X,k):
    centroids = torch.empty((k,X.shape[1]), device=device, dtype=dtype)
    indices = r.sample(range(X.shape[0]),k)
    for i in range(k):
        centroids[i,:] = X[indices[i],:]
    return centroids

#Return a tensor of centroid indices that each data point is closest to (which cluster they belong to)
def find_closest_centroids (X,centroids):
    k = centroids.shape[0]
    distances = torch.empty((X.shape[0],k), device=device, dtype=dtype)
    for i in range(k):
        distances[:,i] = torch.sum(torch.pow(X-centroids[i,:], 2).to(device), dim=1).to(device)
    closest_centroids = torch.argmin(distances, dim=1).to(device)
    return closest_centroids

#Update the centroids to the mean position of all data points in their cluster
def update_centroids (X,k,closest):
    centroids = torch.empty((k,X.shape[1]), device=device, dtype=dtype)
    for i in range(k):
        in_cluster = torch.where(closest==i,1,0).to(device)
        if torch.sum(in_cluster).to(device) != 0:
            centroids[i,:] = torch.sum(torch.transpose(torch.multiply(torch.transpose(X,0,1).to(device),in_cluster).to(device),0,1).to(device), dim=0).to(device)/torch.sum(in_cluster).to(device)
        else:
            centroids[i,:] = X[r.sample(range(X.shape[0]),1),:]
    return centroids

#Get the data point in every cluster that is closest to the centroid
def find_closest_points (X,centroids):
    k = centroids.shape[0]
    distances = torch.empty((X.shape[0],k), device=device, dtype=dtype)
    for i in range(k):
        distances[:,i] = torch.sum(torch.pow(X-centroids[i,:], 2).to(device), dim=1).to(device)
    closest_points = torch.argmin(distances, dim=0)
    return closest_points
