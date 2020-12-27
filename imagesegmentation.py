import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import kmeans as km

#Image processing
image = Image.open('image1.jpg')
image_array = np.asarray(image)
height = image_array.shape[0]
width = image_array.shape[1]
image_array = image_array.reshape(height*width,3)/image_array.max()

#K-means algorithm
coor_weight = 1/3
x_coor = np.transpose(np.tile(np.arange(width,dtype='float32'),height)).reshape(height*width,1)/width*coor_weight
y_coor = np.transpose(np.repeat(np.arange(height,dtype='float32'),width)).reshape(height*width,1)/height*coor_weight
X = np.append(np.append(image_array,x_coor,1),y_coor,1)
k = 4
max_iter = 5

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
    new_image[i] = centroids[closest_centroids[i]][0:3]
    
new_image = new_image.reshape(height,width,3)
plt.imshow(new_image)
plt.show()
