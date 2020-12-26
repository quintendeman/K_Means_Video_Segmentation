import numpy as np
import cv2
from PIL import Image
import kmeans as km

#Image processing
image = Image.open('image1.jpg')
image_array = np.asarray(image)
height = image_array.shape[0]
width = image_array.shape[1]
image_array = image_array.reshape(height*width,3)

#K-means algorithm
X = image_array
k = 3
max_iter = 10

print('Performing K-means...')
centroids = km.initialize_centroids(X, k)
for i in range(max_iter):
    print('iteration '+str(i+1)+' out of '+str(max_iter))
    closest_centroids = km.find_closest_centroids(X, centroids)
    centroids = km.update_centroids(X, k, closest_centroids)
print('Done.')

#Image construction
closest_centroids = km.find_closest_centroids(X, centroids)
new_image = np.zeros((height*width,3))
for i in range(new_image.shape[0]):
    new_image[i] = centroids[closest_centroids[i]]
    
new_image = new_image.reshape(height,width,3)
cv2.imshow('output', new_image/256)
