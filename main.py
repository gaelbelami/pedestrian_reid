import os
import cv2
import time
import imutils
import numpy as np
from realsense.realsense_camera import *
from face_reco import FaceReco

face_reco = FaceReco()

box_thickness = 2
font = cv2.FONT_HERSHEY_COMPLEX_SMALL

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
                    face_descriptor = face_reco.face_encodings(img, face)
                    cv2.rectangle(bgr_frame, (l, t), (r, b), (0, 255, 255), box_thickness//4)

                    if face_locations is not None:
                        min_index = face_reco.compare_faces(face_descriptor, face_descriptors)
                        if min_index:
                            name_pred = index[min_index]
                            name_pred = str(name_pred)
                        else:
                            face_descriptors = np.concatenate((face_descriptors, face_descriptor), axis=0)
                            name_pred = face_reco.sub_faces(bgr_frame, index, (l, t, r, b))
                    else:
                        face_descriptors = face_descriptor
                        name_pred = face_reco.sub_faces(bgr_frame, index, (l, t, r, b))
                    print("name", name_pred)
                    name_pred = int(os.path.split(name_pred)[1].split('.')[0])
                    y3 = t - 15 if t - 15 > 15 else t + 15
                    cv2.putText(bgr_frame, "FaceID: " + str(name_pred), (l + 6, y3), font, 0.6, (0, 255, 255),  box_thickness)
                    
        cv2.imshow("Reid", bgr_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

person_reid()