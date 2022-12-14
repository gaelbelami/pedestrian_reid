import os
import cv2
import time
import imutils
import numpy as np
from realsense.realsense_camera import *
from face_reco import FaceReco

face_reco = FaceReco()

box_size = 2

def person_reid():
    print("[INFO] Accessing video stream...")
    rs = RealsenseCamera()
    face_descriptors, index = face_reco.load_encoding_images()

    start = time.time()
    

    while(True):
        ret, bgr_frame, depth_frame, color_intrin, depth_intrin, aligned_depth_frame = rs.get_frame_stream()

        if ret:
            frame_orig = bgr_frame.copy()
        else:
            print("No camera was detected")
        
        height, width, _ = bgr_frame.shape
        image_np = np.array(frame_orig, 'uint8')
        img = imutils.resize(image_np, width=640)
        ratio = bgr_frame.shape[1] / img.shape[1]

        face_locations = face_reco.detect_known_faces(img)

        if len(face_locations) > 0:
            for face in face_locations:
                if face.confidence > 0.7:
                    l, t, r, b, c = face.rect.left(), face.rect.top(), face.rect.right(), face.rect.bottom(), face.confidence
                    l, t, r, b, c = int(l * ratio), int(t * ratio), int(r * ratio), int(b * ratio), int(c * ratio)
                    # face_descriptor = face_reco.face_encodings(img, face)
                    cv2.rectangle(bgr_frame, (l, t), (r, b), (0, 255, 255), box_size//4)
        cv2.imshow("Reid", bgr_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

person_reid()