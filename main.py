import cv2
import numpy as np

cap = cv2.VideoCapture(0)

classesFile = 'coco.names'
classes = []
with open (classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

modelConfiguation = 'yolov3.cfg'
modelWeights = 'yolov3.weights'

net = cv2.dnn.readNetFromDarknet(modelConfiguation, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def findObjects(outputs, img):
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []

    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > 0.5:
                w,h = int(det[2]*wT), int(det[3]*hT)
                x,y = int(det[0]*wT - w / 2), int(det[1]*hT - h / 2)
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(bbox, confs, 0.5, 0.3)
    
    for i in indices:
        i = i
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.putText(img, f'{classNames[classIds[i]].upper()} {int(confs[i] * 100)}%', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

while True:
    success, img = cap.read()

    blob = cv2.dnn.blobFromImage(img, 1/255, (320, 320), [0,0,0], 1, crop=False)
    net.setInput(blob)

    layerNames = net.getLayerNames()
    outputNames = [layerNames[i-1] for i in net.getUnconnectedOutLayers()]

    output = net.forward(outputNames)

    findObjects(output,img)

    cv2.imshow('Image', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break