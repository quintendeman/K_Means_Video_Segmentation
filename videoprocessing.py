import numpy as np
import cv2

#Returns the video at path as a numpy array with a certain resolution and fps
def video_to_array (path,resolution,fps):
    cap = cv2.VideoCapture(path)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameSkip = int(cap.get(cv2.CAP_PROP_FPS)/fps)
    videoArray = np.empty(((frameCount-1)//frameSkip+1,resolution[0]*resolution[1]*3),np.dtype('float32'))
    i = 0
    while i < frameCount:
        cap.set(cv2.CAP_PROP_POS_FRAMES,i)
        ret, frame = cap.read()
        frame = cv2.resize(frame, (resolution[1],resolution[0]))
        frame = frame.astype('float32')/255
        videoArray[int(i/frameSkip),:] = np.reshape(frame,resolution[0]*resolution[1]*3)
        i += frameSkip
    return videoArray

#Return an array of the images in the list of image numbers "images"
def get_images(path,images,fps):
    cap = cv2.VideoCapture(path)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameSkip = int(cap.get(cv2.CAP_PROP_FPS)/fps)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(height, width)
    imageArray = np.empty((len(images),height,width,3),np.dtype('float32'))
    for i in range(len(images)):
        cap.set(cv2.CAP_PROP_POS_FRAMES,frameSkip*images[i])
        ret, frame = cap.read()
        imageArray[i,:,:,:] = frame.astype('float32')/255
    return imageArray
    
