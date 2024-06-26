import cv2
import mediapipe as mp
import time
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
mpHands=mp.solutions.hands
hands=mpHands.Hands()
mpDraw = mp.solutions.drawing_utils


cap = cv2.VideoCapture(0)
mpHands=mp.solutions.hands
hands=mpHands.Hands()
mpDraw = mp.solutions.drawing_utils


with mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:


    while cap.isOpened():

        success, image = cap.read()

        start = time.time()
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = face_mesh.process(image)
        resultsg = hands.process(image)
        image.flags.writeable = True

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


        if results.multi_face_landmarks:
          for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec)
        if resultsg.multi_hand_landmarks:
            for handLms in resultsg.multi_hand_landmarks: 
                for id, lm in enumerate(handLms.landmark):
                    h, w, c =image.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                mpDraw.draw_landmarks(image, handLms, mpHands.HAND_CONNECTIONS)
                #mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)  # real kamera part\
        end = time.time()
        totalTime = end - start

        fps = 1 / totalTime
        print("FPS: ", fps)
        cv2.imshow('zeros magary sxeulis nawilebi', image)


        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()