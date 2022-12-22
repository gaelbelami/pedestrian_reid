
import time
import cv2
import numpy as np
from arguments import args


class MaskRCNN:
    def __init__(self):

        # Loading Mask RCNN
        self.net = cv2.dnn.readNetFromTensorflow("dnn/frozen_inference_graph_coco.pb", "dnn/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt")
        self.use_gpu_or_cpu()
    
        # Generate random colors(80 colors equals to detectable classes of the model, 3 is the number of channels)
        np.random.seed(2)
        self.colors = np.random.randint(0, 255, (90, 3))

        # Confidence thereshold
        self.detection_threshold = 0.5
        self.mask_threshold = 0.3
        self.classe = 0

        self.bboxes = []
        self.scores = []
        self.classes = ["person"]
        self.centers = []
        self.contours = []

        # Distances
        self.distances = []

        # Resize frame for a faster speed
        self.frame_resizing = 0.25
    
    def use_gpu_or_cpu(self):
        if args["use_gpu"]:
	        # set CUDA as the preferable backend and target
            print("[INFO] Setting preferable backend and target to CUDA...")
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

              
    def mask_rcnn_postprocess(self, bgr_frame):
        small_frame = cv2.resize(bgr_frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        blob = cv2.dnn.blobFromImage(small_frame, swapRB=True, crop=False)
        self.net.setInput(blob)

        start = time.time()
        # Detection results from Mask RCNN
        boxes, masks = self.net.forward(["detection_out_final", "detection_masks"])
        end = time.time()
        print("[Info] Mask R-CNN took {:.6f} seconds".format(end - start))

        frameH, frameW, _ = bgr_frame.shape
        numDetections = boxes.shape[2]

        self.bboxes = []
        self.scores = []
        self.classes = []
        self.centers = []
        self.contours = []
        self.maskes = []


        for i in range(numDetections):
            box = boxes[0, 0, i]
            class_id = box[1]
            score = box[2]

            # If class is not person or score for person is less than threshold, continue
            if class_id != self.classe:
                continue

            if score < self.detection_threshold:
                continue
            
            self.classes.append(class_id)
            self.scores.append(score)
            x1 = int(box[3] * frameW)
            y1 = int(box[4] * frameH)
            x2 = int(box[5] * frameW)
            y2 = int(box[6] * frameH)  

            x1 = max(0, min(x1, frameW - 1))
            y1 = max(0, min(y1, frameH - 1))
            x2 = max(0, min(x2, frameW - 1))
            y2 = max(0, min(y2, frameH - 1))      
            self.bboxes.append([x1, y1, x2, y2])

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            self.centers.append((cx, cy))
            # Contours
            # Get mask coordinated
            # Get the mask
            mask = masks[i, int(class_id)]
            roi = bgr_frame[y1 : y2, x1 : x2]
            roi_height, roi_width, _ = roi.shape
            mask = cv2.resize(mask, (roi_width, roi_height), interpolation=cv2.INTER_CUBIC)
            _, mask = cv2.threshold(mask, self.mask_threshold, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(np.array(mask, np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            self.contours.append(contours)
            mask = np.stack((mask,) * 3, axis= -1)
            mask = mask.astype('uint8')
            bg = 255 - mask * 255
            # mask_show = np.invert(bg)
            mask_img = roi * mask
            mask = mask_img + bg
            # cv2.imwrite("hog/" + str(index) + ".jpg", mask_img)
            self.maskes.append(mask)
        return self.bboxes, self.classes, self.contours, self.scores, self.centers, self.maskes


    # Point is inside
    def isInSide(self, point, box):
        # print(box[0] <= point[0] <= box[2] , box[1] <= point[1] <= box[3])
        return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]
