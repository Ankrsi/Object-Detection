import cv2 as cv
import numpy as np
 
cap = cv.VideoCapture(0)
imgSize=320
confThreshold =0.5
nmsThreshold= 0.2

classesFile = "coco.names"
classNames=open(classesFile).read().strip().split('\n')

modelCfg = "yolov3.cfg"
modelWeights = "yolov3.weights"
net = cv.dnn.readNetFromDarknet(modelCfg, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
colors = np.random.randint(0, 255, size=(len(classNames), 3), dtype='uint8')

while True:
    success, img = cap.read()
    bbox = []
    classIds = []
    confs = []
    hT, wT, cT = img.shape
    blob = cv.dnn.blobFromImage(img, 1 / 255, (imgSize, imgSize), [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    layersNames = net.getLayerNames()
    outputNames = [(layersNames[i[0] - 1]) for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(outputNames)
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w,h = int(det[2]*wT) , int(det[3]*hT)
                x,y = int((det[0]*wT)-w/2) , int((det[1]*hT)-h/2)
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))

    indices = cv.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)

    for i in indices:
        i = i[0]
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        color = [int(c) for c in colors[classIds[i]]]
        text = "{}: {:.4f}".format(classNames[classIds[i]],confs[i])
        cv.rectangle(img, (x, y), (x+w,y+h), color, 2)
        cv.putText(img,text,(x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv.imshow('Image', img)
    if cv.waitKey(1) & 0xFF==ord('q'):
        break
cap.release()
cv.destroyAllWindows()