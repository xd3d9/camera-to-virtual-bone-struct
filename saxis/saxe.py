import cv2
import mediapipe as mp
import time
import numpy as np
# Face mesh detection
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh


drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)



cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:


    while cap.isOpened():

        success, image = cap.read()

        start = time.time()

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        img_w = np.empty(image.shape)
        img_w.fill(0)
        image.flags.writeable = False

        results = face_mesh.process(image)

        image.flags.writeable = True

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


        if results.multi_face_landmarks:
          for face_landmarks in results.multi_face_landmarks:
                #print(face_landmarks)
                #print(face_landmarks.landmark.x)
                mp_drawing.draw_landmarks(
                    image=img_w,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec)
        cv2.imshow('zeros magary saxe   ', img_w)



        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()