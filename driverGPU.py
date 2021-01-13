import torch
import matplotlib.pyplot as plt
import cv2
import kmeansGPU as km
import videoprocessing as vp

def main():
    #Video processing
    path = 'resources\\clip1.mp4'
    target_resolution = (200,300)
    target_fps = 1

    print('Loading video...')
    video_array = vp.video_to_array(path,target_resolution,target_fps)
    print('Done.')

    #Data initialization
    time_weight = 100
    
    X = torch.from_numpy(video_array).to(torch.device('cuda'))
    #X = km.scale_features(X,0)
    X = km.scale_features(X,1)
    X = km.add_position_feature(X,time_weight)

    #K-means algorithm
    k = 10
    max_iter = 20
    
    print('Performing K-means...')
    centroids = km.initialize_centroids(X, k)
    for i in range(max_iter):
        print('iteration '+str(i+1)+' out of '+str(max_iter))
        closest_centroids = km.find_closest_centroids(X, centroids)
        centroids = km.update_centroids(X, k, closest_centroids)
    print('Done.')

    print(closest_centroids)

    #display centroids and thumbnails
    closest_points = km.find_closest_points(X, centroids)
    X = X[:,0:X.shape[1]-1]
    centroids = centroids[:,0:centroids.shape[1]-1]
    for i in range(k):
        image = centroids[i,:].reshape(target_resolution[0],target_resolution[1],3)
        plt.imshow(cv2.cvtColor(image.cpu().numpy(),cv2.COLOR_BGR2RGB))
        plt.title('centroid '+str(i))
        plt.show()
        image = X[closest_points[i],:].reshape(target_resolution[0],target_resolution[1],3)
        plt.imshow(cv2.cvtColor(image.cpu().numpy(),cv2.COLOR_BGR2RGB))
        plt.title('thumbnail '+str(i))
        plt.show()

if __name__ == '__main__':
    main()
