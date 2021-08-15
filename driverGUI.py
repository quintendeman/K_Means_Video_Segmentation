import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import numpy as np
import cv2
from PIL import Image, ImageTk
import kmeans as km
import videoprocessing as vp

def getLocalFile():
    filename = filedialog.askopenfile(initialdir="/", title="Select video file", filetypes=(("Mp4 files", "*.mp4"),))
    localPathLabel1["text"] = filename.name

def getPhotoImage(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = (image*255).astype(np.uint8)
    return ImageTk.PhotoImage(Image.fromarray(image))

#define the outer window
window = tk.Tk()
window.geometry('800x600')
window.title("Thumbnail Recommender & Scene Detector")

#define the notebook and both frames
notebook = ttk.Notebook(window)
notebook.pack(fill="both", expand=True)
frame1 = tk.Frame(notebook)
frame2 = tk.Frame(notebook)
frame1.pack(fill="both", expand=True)
frame2.pack(fill="both", expand=True)
notebook.add(frame1, text="Thumbnail Recommender")
notebook.add(frame2, text="Scene Detector")

#thumbnail recommender frame
leftFrame1 = tk.Frame(frame1, width=200)
leftFrame1.pack(side=tk.LEFT, fill=tk.Y, expand=False)
rightFrame1 = tk.Frame(frame1)
rightFrame1.pack(side=tk.LEFT, fill="both", expand=True)
controlFrame1 = tk.Frame(leftFrame1, width=200)
controlFrame1.pack(side=tk.TOP, expand=False)
thumbnailImages = []
currentImage = None
currentImageNumber = None

sourceSelectLabel1 = tk.Label(controlFrame1, text="Video Source:", width=15)
sourceSelectLabel1.grid(padx=2, pady=2, row=0, column=0)
sourceSelect1 = ttk.Combobox(controlFrame1, state="readonly", values=("clip1", "clip2", "clip3", "Use local file"), width=15)
sourceSelect1.current(0)
sourceSelect1.grid(padx=2, pady=2, row=0, column=1)
localPathLabel1 = tk.Label(controlFrame1, text="", width=15, fg="blue", anchor="w")
localPathLabel1.grid(padx=2, pady=2, row=1, column=0)
localPathButton1 = tk.Button(controlFrame1, text="Browse local files", command=getLocalFile, width=15)
localPathButton1.grid(padx=2, pady=2, row=1, column=1)
fpsLabel1 = tk.Label(controlFrame1, text="Analysis FPS:", width=15)
fpsLabel1.grid(padx=2, pady=2, row=2, column=0)
fpsSelect1 = ttk.Combobox(controlFrame1, state="readonly", values=(0.5, 1, 2, 3, 5), width=15)
fpsSelect1.current(1)
fpsSelect1.grid(padx=2, pady=2, row=2, column=1)
resLabel1 = tk.Label(controlFrame1, text="Analysis Resolution:", width=15)
resLabel1.grid(padx=2, pady=2, row=3, column=0)
resSelect1 = ttk.Combobox(controlFrame1, state="readonly", values=("50x50", "100x100", "200x200", "500x500"), width=15)
resSelect1.current(1)
resSelect1.grid(padx=2, pady=2, row=3, column=1)
thumbnailsLabel1 = tk.Label(controlFrame1, text="Thumbnails:", width=15)
thumbnailsLabel1.grid(padx=2, pady=2, row=4, column=0)
thumbnailSelect1 = ttk.Combobox(controlFrame1, state="readonly", values=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10), width=15)
thumbnailSelect1.current(4)
thumbnailSelect1.grid(padx=2, pady=2, row=4, column=1)
iterationsLabel1 = tk.Label(controlFrame1, text="Iterations:", width=15)
iterationsLabel1.grid(padx=2, pady=2, row=5, column=0)
iterationsSelect1 = ttk.Combobox(controlFrame1, state="readonly", values=(5, 10, 15, 20, 50, 100), width=15)
iterationsSelect1.current(1)
iterationsSelect1.grid(padx=2, pady=2, row=5, column=1)

def generateThumbnails():
    global thumbnailImages
    thumbnailImages = [];
    if (sourceSelect1.get() == "Use local file"):
        path = localPathLabel1["text"]
    else:
        path = "resources\\" + sourceSelect1.get() + ".mp4"
    target_fps = float(fpsSelect1.get())
    resolutions = resSelect1.get().split("x")
    target_resolution = (int(resolutions[0]), int(resolutions[1]))
    progress1.set(10)
    window.update()

    X = vp.video_to_array(path,target_resolution,target_fps)
    k = int(thumbnailSelect1.get())
    max_iter = int(iterationsSelect1.get())
    progress1.set(20)
    window.update()

    centroids = km.initialize_centroids(X, k)
    for i in range(max_iter):
        closest_centroids = km.find_closest_centroids(X, centroids)
        centroids = km.update_centroids(X, k, closest_centroids)
        progress1.set(20 + 80 * (i + 1) / max_iter)
        window.update()

    closest_points = km.find_closest_points(X, centroids)
    thumbnails = vp.get_images(path,closest_points,target_fps)
    for i in range(k):
        image = thumbnails[i,:,:]
        thumbnailImages.append(getPhotoImage(image))
    global currentImage
    currentImage = thumbnailImages[0]
    global currentImageNumber
    currentImageNumber = 0
    imageDisplay.configure(image=currentImage)
    imageLabel["text"] = "Thumbnail 1"

startButton1 = tk.Button(leftFrame1, text="Generate Thumbnails", command=generateThumbnails, width=31)
startButton1.pack(padx=2, pady=2, expand=False)
progress1 = tk.DoubleVar(value=0)
progressBar1 = ttk.Progressbar(leftFrame1, length=225, variable=progress1, mode="determinate")
progressBar1.pack(padx=2, pady=2, expand=False)

def nextImage():
    global thumbnailImages
    global currentImage
    global currentImageNumber
    if(currentImageNumber+1 < len(thumbnailImages)):
        currentImageNumber += 1
        currentImage = thumbnailImages[currentImageNumber]
        imageDisplay.configure(image=currentImage)
        imageLabel["text"] = "Thumbnail " + str(currentImageNumber+1)

def previousImage():
    global thumbnailImages
    global currentImage
    global currentImageNumber
    if(currentImageNumber > 0):
        currentImageNumber -= 1
        currentImage = thumbnailImages[currentImageNumber]
        imageDisplay.configure(image=currentImage)
        imageLabel["text"] = "Thumbnail " + str(currentImageNumber+1)

buttonFrame = tk.Frame(rightFrame1)
buttonFrame.pack(side=tk.TOP, fill=tk.X, expand=False)
previousImageButton = tk.Button(buttonFrame, text="Previous", command=previousImage, width=6)
nextImageButton = tk.Button(buttonFrame, text="Next", command=nextImage, width=6)
nextImageButton.pack(side=tk.RIGHT, expand=False)
previousImageButton.pack(side=tk.RIGHT, expand=False)
imageLabel = tk.Label(buttonFrame, text="No thumbnails")
imageLabel.pack(side=tk.LEFT)
imageDisplay = tk.Label(rightFrame1, image=currentImage, borderwidth=1, relief="solid")
imageDisplay.pack(side=tk.BOTTOM, fill="both", expand=True)

#thumbnail recommender frame
leftFrame2 = tk.Frame(frame2, width=200)
leftFrame2.pack(side=tk.LEFT, fill=tk.Y, expand=False)
rightFrame2 = tk.Frame(frame2)
rightFrame2.pack(side=tk.LEFT, fill="both", expand=True)
controlFrame2 = tk.Frame(leftFrame2, width=200)
controlFrame2.pack(side=tk.TOP, expand=False)

sourceSelectLabel2 = tk.Label(controlFrame2, text="Video Source:", width=15)
sourceSelectLabel2.grid(padx=2, pady=2, row=0, column=0)
sourceSelect2 = ttk.Combobox(controlFrame2, state="readonly", values=("clip1", "clip2", "clip3", "Use local file"), width=15)
sourceSelect2.current(0)
sourceSelect2.grid(padx=2, pady=2, row=0, column=1)
localPathLabel2 = tk.Label(controlFrame2, text="", width=15, fg="blue", anchor="w")
localPathLabel2.grid(padx=2, pady=2, row=1, column=0)
localPathButton2 = tk.Button(controlFrame2, text="Browse local files", command=getLocalFile, width=15)
localPathButton2.grid(padx=2, pady=2, row=1, column=1)
fpsLabel2 = tk.Label(controlFrame2, text="Analysis FPS:", width=15)
fpsLabel2.grid(padx=2, pady=2, row=2, column=0)
fpsSelect2 = ttk.Combobox(controlFrame2, state="readonly", values=(0.5, 1, 2, 3, 5), width=15)
fpsSelect2.current(1)
fpsSelect2.grid(padx=2, pady=2, row=2, column=1)
resLabel2 = tk.Label(controlFrame2, text="Analysis Resolution:", width=15)
resLabel2.grid(padx=2, pady=2, row=3, column=0)
resSelect2 = ttk.Combobox(controlFrame2, state="readonly", values=("50x50", "100x100", "200x200", "500x500"), width=15)
resSelect2.current(1)
resSelect2.grid(padx=2, pady=2, row=3, column=1)
scenesLabel2 = tk.Label(controlFrame2, text="Scenes:", width=15)
scenesLabel2.grid(padx=2, pady=2, row=4, column=0)
scenesSelect2 = ttk.Combobox(controlFrame2, state="readonly", values=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10), width=15)
scenesSelect2.current(4)
scenesSelect2.grid(padx=2, pady=2, row=4, column=1)
iterationsLabel2 = tk.Label(controlFrame2, text="Iterations:", width=15)
iterationsLabel2.grid(padx=2, pady=2, row=5, column=0)
iterationsSelect2 = ttk.Combobox(controlFrame2, state="readonly", values=(5, 10, 15, 20, 50, 100), width=15)
iterationsSelect2.current(1)
iterationsSelect2.grid(padx=2, pady=2, row=5, column=1)

def detectScenes():
    if (sourceSelect2.get() == "Use local file"):
        path = localPathLabel2["text"]
    else:
        path = "resources\\" + sourceSelect2.get() + ".mp4"
    target_fps = float(fpsSelect2.get())
    resolutions = resSelect2.get().split("x")
    target_resolution = (int(resolutions[0]), int(resolutions[1]))
    progress2.set(10)
    window.update()

    time_weight = 200
    X = vp.video_to_array(path,target_resolution,target_fps)
    X = km.scale_features(X,1)
    X = km.add_position_feature(X,time_weight)
    k = int(scenesSelect2.get())
    max_iter = int(iterationsSelect2.get())
    progress2.set(20)
    window.update()

    centroids = km.initialize_centroids(X, k)
    for i in range(max_iter):
        closest_centroids = km.find_closest_centroids(X, centroids)
        centroids = km.update_centroids(X, k, closest_centroids)
        progress2.set(20 + 80 * (i + 1) / max_iter)
        window.update()

    centroids = centroids[np.argsort(centroids[:,-1])]
    scenes = km.find_closest_centroids(X,centroids)
    print(scenes)
    for i in range(k):
        scenesOutput.insert(tk.END, "Scene "+ str(i+1) + ":\n")
        start = None
        for j in range(len(scenes)+1):
            if (start == None):
                if (j != len(scenes) and scenes[j] == i):
                    start = j
            else:
                if (j == len(scenes) or scenes[j] != i):
                    end = j
                    scenesOutput.insert(tk.END, str(start/target_fps) + "s - " + str(end/target_fps) + "s\n")
                    start = None

startButton2 = tk.Button(leftFrame2, text="Detect Scenes", command=detectScenes, width=31)
startButton2.pack(padx=2, pady=2, expand=False)
progress2 = tk.DoubleVar(value=0)
progressBar2 = ttk.Progressbar(leftFrame2, length=225, variable=progress2, mode="determinate")
progressBar2.pack(padx=2, pady=2, expand=False)

scenesOutput = tk.Text(rightFrame2)
scenesOutput.pack(fill="both", expand=True)

window.mainloop()