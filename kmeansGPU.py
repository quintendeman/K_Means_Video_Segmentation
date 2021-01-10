import numpy as np
import random as r
import torch

device = torch.device('cuda')
dtype = torch.float32

def initialize_centroids (X,k):
    centroids = torch.empty((k,X.shape[1]), device=device, dtype=dtype)
    indices = r.sample(range(X.shape[0]),k)
    for i in range(k):
        centroids[i,:] = X[indices[i],:]
    return centroids

def find_closest_centroids (X,centroids):
    k = centroids.shape[0]
    distances = torch.empty((X.shape[0],k), device=device, dtype=dtype)
    for i in range(k):
        distances[:,i] = torch.sum(torch.pow(X-centroids[i,:], 2).to(device), dim=1).to(device)
    closest_centroids = torch.argmin(distances, dim=1).to(device)
    return closest_centroids

def update_centroids (X,k,closest):
    centroids = torch.empty((k,X.shape[1]), device=device, dtype=dtype)
    for i in range(k):
        in_cluster = torch.where(closest==i,1,0).to(device)
        if torch.sum(in_cluster).to(device) != 0:
            centroids[i,:] = torch.sum(torch.transpose(torch.multiply(torch.transpose(X,0,1).to(device),in_cluster).to(device),0,1).to(device), dim=0).to(device)/torch.sum(in_cluster).to(device)
        else:
            centroids[i,:] = X[r.sample(range(X.shape[0]),1),:]
    return centroids

def find_closest_points (X,centroids):
    k = centroids.shape[0]
    distances = torch.empty((X.shape[0],k), device=device, dtype=dtype)
    for i in range(k):
        distances[:,i] = torch.sum(torch.pow(X-centroids[i,:], 2).to(device), dim=1).to(device)
    closest_points = torch.argmin(distances, dim=0)
    return closest_points
