import numpy as np
import cv2
import kMeansFunctions as km
import videoProcessingFunctions as vp

path = 'clip1.mp4'
resolution = (100,150)
framekeep = 10

print('Processing video...')
videoArray = vp.videoToArray(path,resolution,framekeep)
print('Done.')

X = videoArray
k = 10
maxIter = 10

print('Performing K-means...')
centroids = km.initializeCentroids(X, k)
for i in range(maxIter):
    closest = km.findClosestCentroids(X, centroids)
    centroids = km.updateCentroids(X, k, closest)
print('Done.')

for i in range(k):
    image = centroids[i,:].reshape(resolution[0],resolution[1],3)
    cv2.imshow('centroid '+str(i), image)
