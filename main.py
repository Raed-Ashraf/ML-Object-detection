import cv2
import numpy as np

confThreshold = 0.4
nmsThreshold = 0.3

# open camera
cap = cv2.VideoCapture(1)

# Load Yolo
net = cv2.dnn.readNetFromDarknet('models\yolov3.cfg','models\yolov3.weights')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

classes = []
with open("models\coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# # use this if you want different colors
# np.random.seed(543210)
# colors = np.random.uniform(0, 255, size=(len(classes), 3))

# find object function
def findObjects(outputs,img):
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []

    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]

            if confidence > confThreshold:
                w,h = int(det[2]*wT), int(det[3]*hT)
                x,y = int((det[0]*wT)-w/2), int((det[1]*hT)-h/2)
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(bbox,confs,confThreshold,nmsThreshold)

    for i in indices:
        box = bbox[i]
        x,y,w,h = box[0], box[1], box[2], box[3]
        # if you want different colors use colors[classIds[i]] instead of (255,0,255)
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,255), 2)
        cv2.putText(img, f'{classes[classIds[i]].upper()} {int(confs[i]*100)}%',
                    (x , y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,255), 2)

while True:
    # take image and process it
    success, img = cap.read()

    # # use pre-downloaded image
    # img = cv2.imread('street.jpg')

    # make blob object to feed it to the model
    blob = cv2.dnn.blobFromImage(img, 1/255, (320,320), [0,0,0], 1, crop=False)
    net.setInput(blob)

    # getting the output layers from the network
    layerNames = net.getLayerNames()
    outputNames = [layerNames[i-1] for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(outputNames)

    # finding objects in the image using the model predictions and draw bounding boxes
    findObjects(outputs,img)

    # view the images with bounding boxes
    cv2.imshow('image', img)
    cv2.waitKey(1)


