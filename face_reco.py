import os
import cv2
import dlib
import math 
import numpy as np


class FaceReco:
    def __init__(self):
        self.face_detector = dlib.get_frontal_face_detector()
        self.pose_predictor_68_points = dlib.shape_predictor('pre_weights/shape_predictor_68_face_landmarks.dat')
        # self.pose_predictor_5_points = dlib.shape_predictor('pre_weights/shape_predictor_5_face_landmarks.dat')
        self.cnn_face_detector = dlib.cnn_face_detection_model_v1('pre_weights/mmod_human_face_detector.dat')
        self.face_encoder = dlib.face_recognition_model_v1('pre_weights/dlib_face_recognition_resnet_model_v1.dat')

        self.known_face_encodings = []
        self.face_IDs = []
        
        self.centers = []
        self.boxes = []

        self.width = 220
        self.height = 220

        # Resize frame for a faster speed
        self.frame_resizing = 0.25

    
    def face_locations(self, img, number_of_times_to_upsample=1):
        return self.cnn_face_detector(img, number_of_times_to_upsample)

    def load_encoding_images(self, ):
        index = {}
        idx = 0
        face_descriptors = None
        images_path = [os.path.join('dataset/face', f) for f in os.listdir('dataset/face')]
        
        for img_path in images_path:
            img = cv2.imread(img_path)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            image_np = np.array(rgb_img, 'uint8')
            face_location = self.face_locations(image_np)

            for face in face_location:
                face_descriptor = self.face_encodings(image_np, face)
                if face_descriptors is None:
                    face_descriptors = face_descriptor
                else:
                    face_descriptors = np.concatenate((face_descriptors, face_descriptor), axis=0)
                index[idx] = img_path
                idx += 1
            face_descriptors = np.array(face_descriptors)
            np.save('face_repr.npy', face_descriptors)

        return face_descriptors, index
    
    def compare_faces(self, face_descriptor, face_descriptors, threshold = 0.6):
        distances = np.linalg(face_descriptor - face_descriptors, axis=1)
        min_index = np.argmin(distances)
        if distances[min_index] <= threshold:
            return min_index
    

    def face_landmarks(self, image_np, face_locations):
        return [self.pose_predictor_68_points(image_np, face_location.rect) for face_location in face_locations]


    def face_encodings(self, image_np, face=None):
        points = self.pose_predictor_68_points(image_np, face.rect)
        face_encodings = self.face_encoder.compute_face_descriptor(image_np, points, num_jitters=2)
        face_encodings = [f for f in face_encodings]
        return face_encodings[np.newaxis, :]

    
    def sub_faces(self, bgr_frame, index, face):
        l, t, r, b = face
        name_pred = len(index) + 1
        sub_face = cv2.resize(bgr_frame[t:b, l:r], (self.width, self.height))
        FaceFileName = "dataset/Face/" + \
            str(name_pred) + ".jpg"
        cv2.imwrite(FaceFileName, sub_face)
        index[name_pred - 1] = FaceFileName
        name_pred = str(name_pred)
        return name_pred
    
    def detect_known_faces(self, image_np,  return_ori_result = False):      
        face_locations = self.face_locations(image_np)
        return  face_locations
    

    def is_close(self, bgr_frame, camera_coordinate1, camera_coordinate2):        
        if camera_coordinate1[0]*camera_coordinate1[1]*camera_coordinate1[2]*camera_coordinate2[0]*camera_coordinate2[1]*camera_coordinate2[2]==0:
            cv2.putText(bgr_frame, "Dist1to2:" + "Please insert points with depth", (40, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 255, 255])
        else:     
            distance =  math.sqrt((camera_coordinate2[0]-camera_coordinate1[0])**2 + (camera_coordinate2[1]-camera_coordinate1[1])**2 + (camera_coordinate2[2]-camera_coordinate1[2])**2)
            cv2.putText(bgr_frame, "Dist1to2: " +str(distance) + "m", (40, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 255, 255])
            return distance