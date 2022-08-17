# ML-Object-detection
In this project we are going to learn how to run one of the most popular object detection algorithms YOLOv3.

## What is YOLO?
YOLO (You Only Look Once) is a family of deep learning models designed for fast object Detection.
There are many versions of YOLO, we are using YOLO version 3.
The first version proposed the general architecture, where the second version refined the design and made use of predefined anchor boxes to improve the bounding box proposal, and version three further refined the model architecture and training process.
It is based on the idea that:
" A single neural network predicts bounding boxes and class probabilities directly from full images in one evaluation. Since the whole detection pipeline is a single network, it can be optimized end-to-end directly on detection performance. "

## Object detecion steps in YOLOv3
*	The input is a batch of images of shape (m, 416, 416, 3).
*	YOLO v3 passes this image to a convolutional neural network (CNN).
*	The last dimensions of the output are flattened to get an output volume of (19, 19, 425):
    *	Here, each cell of a 19 x 19 grid returns 425 numbers.
    *	425 = 5 * 85, where 5 is the number of anchor boxes per grid.
    *	85 = 5 + 80, where 5 is (pc, bx, by, bh, bw) and 80 is the number of classes we want to detect.
*	The output is a list of bounding boxes along with the recognized classes. Each bounding box is represented by 6 numbers (pc, bx, by, bh, bw, c). If we expand c into an 80-dimensional vector, each bounding box is represented by 85 numbers.
*	Finally, we do the IoU (Intersection over Union) and Non-Max Suppression to avoid selecting overlapping boxes.

![](images/YOLOv3_output.PNG)
