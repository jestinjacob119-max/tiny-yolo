# Usage example:  python3 object_detection_yolo.py --video=run.mp4
#                 python3 object_detection_yolo.py --image=bird.jpg
import time
import cv2 as cv
import argparse
import sys
import numpy as np
import os.path
from imutils.video import VideoStream
import imutils

# Initialize the parameters
confThreshold = 0.1  # Confidence threshold
nmsThreshold = 0.45  # Non-maximum suppression threshold
inpWidth = 416 # Width of network's input image
inpHeight = 416 # Height of network's input image
#usingPiCamera = True
usingPiCamera = False
frameSize = (416, 416)
saveVideo = False  # Set True to save the annotated video (slows things down, especially on a Pi)

parser = argparse.ArgumentParser(description='Object Detection using YOLO in OPENCV')
parser.add_argument('--image', help='Path to image file.')
parser.add_argument('--video', help='Path to video file')
args = parser.parse_args()

# Load names of classes
classesFile = "pothole.names"
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Give the configuration and weight files for the model and load the network using them.
modelConfiguration = "cfg/yolov3.cfg"
modelWeights = "cfg/yolov3.weights"

net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)


# Draw the predicted bounding box
def drawPred(frame, classId, conf, left, top, right, bottom):
    # Draw a bounding box.
    cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)

    label = '%.2f' % conf

    # Get the label for the class name and its confidence
    if classes:
        assert (classId < len(classes))
        label = '%s:%s' % (classes[classId], label)

    # Display the label at the top of the bounding box
    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv.rectangle(frame, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine),
                 (255, 255, 255), cv.FILLED)
    cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)


# Remove the bounding boxes with low confidence using non-maxima suppression
def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    # Newer OpenCV versions return a flat array (older ones return Nx1), so flatten to handle both
    for i in np.array(indices).flatten():
        left, top, width, height = boxes[i]
        drawPred(frame, classIds[i], confidences[i], left, top, left + width, top + height)


# Process inputs
winName = 'Deep learning object detection in OpenCV'
cv.namedWindow(winName, cv.WINDOW_NORMAL)

outputFile = "yolo_out_py.avi"
cap = None         # cv.VideoCapture, used for image/video files
vs = None          # VideoStream, used for the live camera
vid_writer = None  # created on the first frame when saveVideo is True
if (args.image):
    # Open the image file
    if not os.path.isfile(args.image):
        print("Input image file ", args.image, " doesn't exist")
        sys.exit(1)
    cap = cv.VideoCapture(args.image)
    outputFile = os.path.splitext(args.image)[0] + '_yolo_out_py.jpg'
elif (args.video):
    # Open the video file
    if not os.path.isfile(args.video):
        print("Input video file ", args.video, " doesn't exist")
        sys.exit(1)
    cap = cv.VideoCapture(args.video)
    outputFile = os.path.splitext(args.video)[0] + '_yolo_out_py.avi'
else:
    # Webcam input
    vs = VideoStream(src=0, usePiCamera=usingPiCamera, resolution=frameSize,
        framerate=32).start()
    time.sleep(2.0)
    

# Make sure a file input actually opened
if cap is not None and not cap.isOpened():
    print("Could not open the input file")
    sys.exit(1)

time4 = time.time()
start_time = int(round(time.monotonic() * 1000))
while cv.waitKey(1) < 0:
    
    time01=time.time()
    print('a',time01-time4)#-------------------------
    
    # get frame from the camera stream or from the image/video file
    if vs is not None:
        frame = vs.read()
        hasFrame = frame is not None
    else:
        hasFrame, frame = cap.read()
    
    time02=time.time()
    print('b',time02-time01)#-------------------------
    
    # Stop the program if reached end of video
    if not hasFrame:
        print("Done processing !!!")
        if args.image or vid_writer is not None:
            print("Output file is stored as ", outputFile)
        if args.image:
            cv.waitKey(3000)  # keep the annotated image on screen for a moment
        break

    # Create the video writer on the first frame, using the real frame size
    if saveVideo and not args.image and vid_writer is None:
        print(outputFile)
        vid_writer = cv.VideoWriter(outputFile, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30,
                                    (frame.shape[1], frame.shape[0]))

    # Create a 4D blob from a frame.
    blob = cv.dnn.blobFromImage(frame, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)
    
    time03=time.time()
    print('c',time03-time02)#-------------------------
    
    # Sets the input to the network
    net.setInput(blob)
    # Runs the forward pass to get output of the output layers
    outs = net.forward(net.getUnconnectedOutLayersNames())
    
    time1=time.time()
    print('d',time1-time03)#-------------------------
    
    # Remove the bounding boxes with low confidence
    postprocess(frame, outs)
    
    time2=time.time()
    print('e',time2-time1)#---------------------------
    
    # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
    t, _ = net.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
    cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
    
    time3=time.time()
    print('f',time3-time2)#---------------------------
    
    # Write the frame with the detection boxes
    if (args.image):
        cv.imwrite(outputFile, frame.astype(np.uint8))
    elif vid_writer is not None:
        vid_writer.write(frame.astype(np.uint8))
    cv.imshow(winName, frame)
    
    time4=time.time()
    print('g',time4-time3)#---------------------------
    
# Release resources
if vs is not None:
    vs.stop()
if cap is not None:
    cap.release()
if vid_writer is not None:
    vid_writer.release()
cv.destroyAllWindows()

current_time = int(round(time.monotonic() * 1000))

print("Total Processing time:- " + str(current_time - start_time))

