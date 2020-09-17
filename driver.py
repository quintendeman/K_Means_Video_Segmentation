import numpy as np
import cv2
import kmeans as km
import videoprocessing as vp

#Video processing
path = 'resources\\clip3.mp4'
target_resolution = (100,150)
target_fps = 4

print('Processing video...')
video_array = vp.video_to_array(path,target_resolution,target_fps)
print('Done.')

#K-means algorithm
X = video_array
k = 20
max_iter = 20

print('Performing K-means...')
centroids = km.initialize_centroids(X, k)
for i in range(max_iter):
    print('iteration'+str(i+1)+' out of '+str(max_iter))
    closest_centroids = km.find_closest_centroids(X, centroids)
    centroids = km.update_centroids(X, k, closest_centroids)
print('Done.')

#display centroids
for i in range(k):
    image = centroids[i,:].reshape(target_resolution[0],target_resolution[1],3)
    #cv2.imshow('centroid '+str(i), image)

#display thumbnails
closest_points = km.find_closest_points(X, centroids)
for i in range(k):
    image = X[closest_points[i],:].reshape(target_resolution[0],target_resolution[1],3)
    cv2.imshow('thumbnail '+str(i), image)
