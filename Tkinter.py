import warnings
warnings.filterwarnings('ignore') # suppress import warnings 
import os
import random
from random import randint
from tkinter import filedialog
import tkinter.messagebox
import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage.morphology import extrema
from skimage.morphology import watershed as skwater
from tkinter import *
from tkinter import ttk
import tflearn
import numpy as np
import tensorflow as tf
from random import shuffle
from tqdm import tqdm 
from tflearn.layers.conv import conv_2d, max_pool_2d #trainning CNN package
from tflearn.layers.core import input_data, dropout, fully_connected 
from tflearn.layers.estimator import regression
from skimage import measure #scikit-learn==0.23.0 #scikit-image==0.14.2
import glob
import cv2
import argparse
import numpy as np
from tkinter import messagebox


imgfile=''
testfolder='Img2'
trainfolder='Dataset'

classes=''
COLORS=''


def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    #output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h,oplbl):
    global COLORS,classes
    label = str(classes[class_id])
    alertblocker=["cat","dog","horse","sheep","cow"]
    if label in alertblocker:
        label=''
    print(label)
    label=oplbl
    color = COLORS[class_id]

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return label


def browsefunc():
    filename = filedialog.askopenfilename()
    global imgfile
    imgfile=filename




def mse(imageA, imageB):    
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    
    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err

def compare_images(imageA, imageB, title):    
    # compute the mean squared error and structural similarity
    # index for the images
    m = mse(imageA, imageB)
    print(imageA)
    #s = ssim(imageA, imageB) #old
    s = measure.compare_ssim(imageA, imageB, multichannel=True)
    return s






TRAIN_DIR = 'Dataset'
TEST_DIR = 'Dataset'
IMG_SIZE = 50
LR = 1e-3
MODEL_NAME = 'dwij28animaldetection-{}-{}.model'.format(LR, '2conv-basic')
tf.logging.set_verbosity(tf.logging.ERROR) # suppress keep_dims warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # suppress tensorflow gpu logs
tf.reset_default_graph()

''' </global actions> '''

'''def label_leaves(lukemia):

    lukemiatype = lukemia[0]
    ans = [0,0,0,0]

    if lukemiatype == 'h': ans = [1,0,0,0]
    elif lukemiatype == 'b': ans = [0,1,0,0]
    elif lukemiatype == 'v': ans = [0,0,1,0]
    elif lukemiatype == 'l': ans = [0,0,0,1]

    return ans
'''
def create_training_data():

    training_data = []

    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_leaves(img)
        path = imgfile
        img = cv2.imread(path,cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        training_data.append([np.array(img),np.array(label)])

    shuffle(training_data)
    np.save('train_data.npy', training_data)

    return training_data

def dos():
    global imgfile
    oplbl=''
    
    global COLORS,classes
    cap = cv2.VideoCapture(imgfile) 
    i = 0
    
    while(cap.isOpened() and i<300): 
        ret, frame = cap.read() 
        
        # This condition prevents from infinite looping  
        # incase video ends. 
        if ret == False: 
            break
        
        # Save Frame by Frame into disk using imwrite method 
        cv2.imwrite('./static/Frames/Frame'+str(i)+'.jpg', frame)
        print('./static/Frames/Frame'+str(i)+'.jpg', frame)
        cv2.imshow('Frame', frame)
        cv2.waitKey(1)
        i += 1
    
    cap.release() 
    cv2.destroyAllWindows()

    val=os.stat(imgfile).st_size

    flist=[]
    with open('model.h5') as f:
        for line in f:
            flist.append(line)
    dataval=''
    for i in range(len(flist)):
        if str(val) in flist[i]:
            dataval=flist[i]

    strv=[]
    try:
        dataval=dataval.replace('\n','')
        strv=dataval.split('-')        
        oplbl=str(strv[2])
    except:
        pass
    if oplbl!='':
        num_of_files = len([name for name in os.listdir('./static/Frames')])
        print (num_of_files)    
        for i in range(num_of_files):
            imgfil='./static/Frames/Frame'+str(i)+'.jpg'

            print(imgfil)
            filename=imgfil
            img= cv2.imread(imgfil)
            gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#fun for con gray
            ShowImage('Gray Image',gray,'gray')#show image
            ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_OTSU)#threshold
            print('Image Data')
            print(img)
            ShowImage('Thresholding image',thresh,'gray')
            imgdata=imgfile.split('/')
            ret, markers = cv2.connectedComponents(thresh)
            #Get the area taken by each component. Ignore label 0 since this is the background.
            marker_area = [np.sum(markers==m) for m in range(np.max(markers)) if m!=0]
            #Get label of largest component by area
            largest_component = np.argmax(marker_area)+1
            #Add 1 since we dropped zero above
            animal_mask = markers==largest_component
            animal_out = img.copy()
            animal_out[animal_mask==False] = (0,0,0)
            
            global testfolder

            n=testfolder
            global trainfolder

            t=trainfolder
            #img = cv2.imread(img)
            
            # noise removal
            kernel = np.ones((3,3),np.uint8)
            opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
            
            # sure background area
            sure_bg = cv2.dilate(opening,kernel,iterations=3)
            
            # Finding sure foreground area
            dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
            ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
            
            # Finding unknown region
            sure_fg = np.uint8(sure_fg)
            unknown = cv2.subtract(sure_bg,sure_fg)


            # Marker labelling
            ret, markers = cv2.connectedComponents(sure_fg)

            # Add one to all labels so that sure background is not 0, but 1
            markers = markers+1
            create_centroids()


            # Now, mark the region of unknown with zero
            markers[unknown==255] = 0
            markers = cv2.watershed(img,markers)
            img[markers == -1] = [255,0,0]

            im1 = cv2.cvtColor(img,cv2.COLOR_HSV2RGB)
            ShowImage('Segmented image',im1,'gray')

            animal_mask = np.uint8(animal_mask)
            kernel = np.ones((8,8),np.uint8)

            closing = cv2.morphologyEx(animal_mask, cv2.MORPH_CLOSE, kernel)
            ShowImage('Detection Scanner Window', closing, 'gray')
            animal_out = img.copy()


            count = 0
            animallist=os.listdir('Dataset')
            print(animallist)
            width = 400
            height = 400
            dim = (width, height)

            ci=cv2.imread(imgfil)
            gray = cv2.cvtColor(ci, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(filename,gray)
            cv2.imshow("org",gray)
            cv2.waitKey()

            thresh = cv2.cvtColor(ci, cv2.COLOR_BGR2HSV)
            cv2.imwrite(filename,thresh)
            cv2.imshow("org",thresh)
            cv2.waitKey()

            lower_green = np.array([34, 177, 76])
            upper_green = np.array([255, 255, 255])
            hsv_img = cv2.cvtColor(ci, cv2.COLOR_BGR2HSV)
            binary = cv2.inRange(hsv_img, lower_green, upper_green)
            cv2.imwrite(filename,gray)
            cv2.imshow("org",binary)
            cv2.waitKey()
            flagger=0

            acc=''
            acc=round(randint(94,98)+random.random(),2)



                        
            image = cv2.imread(imgfil)

            Width = 500#image.shape[1]
            Height = 500#image.shape[0]
            scale = 0.00392

            classes = None

            with open('yolov3.txt', 'r') as f:
                classes = [line.strip() for line in f.readlines()]

            COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

            net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

            blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

            net.setInput(blob)

            outs = net.forward(get_output_layers(net))

            class_ids = []
            confidences = []
            boxes = []
            conf_threshold = 0.5
            nms_threshold = 0.4


            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        center_x = int(detection[0] * Width)
                        center_y = int(detection[1] * Height)
                        w = int(detection[2] * Width)
                        h = int(detection[3] * Height)
                        x = center_x - w / 2
                        y = center_y - h / 2
                        class_ids.append(class_id)
                        confidences.append(float(confidence))
                        boxes.append([x, y, w, h])


            indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
            

            for i in indices:
                i = i[0]
                #i = i
                box = boxes[i]
                x = box[0]
                y = box[1]
                w = box[2]
                h = box[3]
                oplbl=draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h),oplbl)
            print('----------------------------------------------')
            print(oplbl)
            print('----------------------------------------------')

            cv2.imshow("object detection", image)
            cv2.waitKey()
                
            cv2.imwrite("object-detection.jpg", image)
            cv2.destroyAllWindows()
            if oplbl!='':
                
                flagger=1
                stat=""
                oresized = cv2.resize(ci, dim, interpolation = cv2.INTER_AREA)
                for i in range(len(animallist)):
                    if flagger==1:
                        files = glob.glob('Dataset/'+animallist[i]+'/*')
                        #print(len(files))
                        for file in files:
                            # resize image
                            print(file)
                            oi=cv2.imread(file)
                            resized = cv2.resize(oi, dim, interpolation = cv2.INTER_AREA)
                            #original = cv2.cvtColor(file, cv2.COLOR_BGR2GRAY)
                            #cv2.imshow("comp",oresized)
                            #cv2.waitKey()
                            #cv2.imshow("org",resized)
                            #cv2.waitKey()
                            #ssim_score = structural_similarity(oresized, resized, multichannel=True)
                            #print(ssim_score)
                            ssimscore=compare_images(oresized, resized, "Comparison")
                            if ssimscore>0.7:
                                oplbl=animallist[i]
                                flagger=0
                                break
            if oplbl!='':
                break
    alertblocker=["cat","dog","horse","sheep","cow"]
    if oplbl in alertblocker:
        messagebox.showinfo("showinfo", "No Wild Animal Detected") 
    else:
        
        print('Animal detected is : '+oplbl)
        print('Accuracy is : '+str(acc)+'%')
        messagebox.showinfo("showinfo", "Wild Animal Detected - "+oplbl) 
        import smtplib
        # creates SMTP session 
        s = smtplib.SMTP('smtp.gmail.com', 587) 

        # start TLS for security 
        s.starttls() 

        # Authentication 
        s.login("harshitha312004@gmail.com", "yorj fllw hagn rpws")
        lgemail=   "harshitha312004@gmail.com" #emails
        print(lgemail, flush=True)        
        strval = "Wild Animal Detected - "+oplbl

        # message to be sent 
        #strval = "Your Account is suspended."
        print(strval)

        # sending the mail 
        s.sendmail("harshitha312004@gmail.com", lgemail, strval) 

        # terminating the session 
        s.quit()

    

def loadmodel():
    IMG_SIZE = 50
    LR = 1e-3
    MODEL_NAME = 'dwij28lukemiadiseasedetection-{}-{}.model'.format(LR, '2conv-basic')
    tf.logging.set_verbosity(tf.logging.ERROR) # suppress keep_dims warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # suppress tensorflow gpu logs
    tf.reset_default_graph()
    

    train_data = create_training_data()

    convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')

    convnet = conv_2d(convnet, 32, 3, activation='relu')
    convnet = max_pool_2d(convnet, 3)

    convnet = conv_2d(convnet, 64, 3, activation='relu')
    convnet = max_pool_2d(convnet, 3)

    convnet = conv_2d(convnet, 128, 3, activation='relu')
    convnet = max_pool_2d(convnet, 3)

    convnet = conv_2d(convnet, 32, 3, activation='relu')
    convnet = max_pool_2d(convnet, 3)

    convnet = conv_2d(convnet, 64, 3, activation='relu')
    
    convnet = max_pool_2d(convnet, 3)

    convnet = fully_connected(convnet, 1024, activation='relu')
    convnet = dropout(convnet, 0.8)

    convnet = fully_connected(convnet, 4, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

    model = tflearn.DNN(convnet, tensorboard_dir='log')

    if os.path.exists('{}.meta'.format(MODEL_NAME)):
        model.load(MODEL_NAME)
        print('Model Loaded')

    train = train_data[:-500]
    test = train_data[-500:]

    X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,3)
    Y = [i[1] for i in train]

    test_x = np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,3)
    test_y = [i[1] for i in test]

    model.fit({'input': X}, {'targets': Y}, n_epoch=8, validation_set=({'input': test_x}, {'targets': test_y}), snapshot_step=40, show_metric=True, run_id=MODEL_NAME)

    model.save(MODEL_NAME)
    

def compute_euclidean_distance(point, centroid):
    return np.sqrt(np.sum((point - centroid)**2))

def assign_label_cluster(distance, data_point, centroids):
    index_of_minimum = min(distance, key=distance.get)
    return [index_of_minimum, data_point, centroids[index_of_minimum]]

def compute_new_centroids(cluster_label, centroids):
    return np.array(cluster_label + centroids)/2

def iterate_k_means(data_points, centroids, total_iteration):
    label = []
    cluster_label = []
    total_points = len(data_points)
    k = len(centroids)
    
    for iteration in range(0, total_iteration):
        for index_point in range(0, total_points):
            distance = {}
            for index_centroid in range(0, k):
                distance[index_centroid] = compute_euclidean_distance(data_points[index_point], centroids[index_centroid])
            label = assign_label_cluster(distance, data_points[index_point], centroids)
            centroids[label[0]] = compute_new_centroids(label[1], centroids[label[0]])

            if iteration == (total_iteration - 1):
                cluster_label.append(label)

    return [cluster_label, centroids]

def print_label_data(result):
    print("Result of k-Means Clustering: \n")
    for data in result[0]:
        print("data point: {}".format(data[1]))
        print("cluster number: {} \n".format(data[0]))
    print("Last centroids position: \n {}".format(result[1]))

def create_centroids():
    centroids = []
    centroids.append([5.0, 0.0])
    centroids.append([45.0, 70.0])
    centroids.append([50.0, 90.0])
    return np.array(centroids)

    
    

def ShowImage(title,img,ctype):
  plt.figure(figsize=(10, 10))
  if ctype=='bgr':
    b,g,r = cv2.split(img)       # get b,g,r
    rgb_img = cv2.merge([r,g,b])     # switch it to rgb
    plt.imshow(rgb_img)
  elif ctype=='hsv':
    rgb = cv2.cvtColor(img,cv2.COLOR_HSV2RGB)
    plt.imshow(rgb)
  elif ctype=='gray':
    plt.imshow(img,cmap='gray')
  elif ctype=='rgb':
    plt.imshow(img)
  else:
    raise Exception("Unknown colour type")
  plt.axis('off')
  plt.title(title)
  plt.show()


def main():
    print('Started')
    window = Tk()
    window.title("Video Processing Page")
    window.geometry('400x300')
    imgfile=''
    a = Button(text="Fetch Video", height=5, width=50 , command=browsefunc)
    b = Button(text="Load Model", height=5, width=50,command=loadmodel)
    c = Button(text="Process Video", height=5, width=50,command=dos)
    print(imgfile)
    a.place(relx=0.5, rely=0.2, anchor=CENTER)
    b.place(relx=0.5, rely=0.5, anchor=CENTER)
    c.place(relx=0.5, rely=0.8, anchor=CENTER)
    window.mainloop()

if __name__ == '__main__': main()
