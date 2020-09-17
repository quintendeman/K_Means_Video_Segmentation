import numpy as np
import cv2

def video_to_array (path,resolution,fps):
    cap = cv2.VideoCapture(path)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    framekeep = int(cap.get(cv2.CAP_PROP_FPS)/fps)
    videoArray = np.empty(((frameCount-1)//framekeep+1,resolution[0]*resolution[1]*3),np.dtype('float32'))
    for i in range(frameCount):
        ret, frame = cap.read()
        if i%framekeep == 0:
            frame = cv2.resize(frame,(resolution[1],resolution[0]))
            frame = frame.astype('float32')/255
            videoArray[int(i/framekeep),:] = np.reshape(frame, resolution[0]*resolution[1]*3)
    return videoArray
