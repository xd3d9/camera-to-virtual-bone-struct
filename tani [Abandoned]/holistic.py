import cv2
import mediapipe as mp
import time
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mpHands=mp.solutions.hands
hands=mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

cap = cv2.VideoCapture(0)


with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as holistic:

    while cap.isOpened():

        success, image = cap.read()
       
        start = time.time()

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = holistic.process(image)
        resultss = hands.process(image)
        image.flags.writeable = True

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        img_w = np.empty(image.shape)
        img_w.fill(0)


        #print(results.pose_landmarks)

        if resultss.multi_hand_landmarks:
            for handLms in resultss.multi_hand_landmarks: 
                for id, lm in enumerate(handLms.landmark):
                    h, w, c =image.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                mpDraw.draw_landmarks(img_w, handLms, mpHands.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(img_w, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,landmark_drawing_spec=drawing_spec,connection_drawing_spec=drawing_spec)


        end = time.time()
        totalTime = end - start

        fps = 1 / totalTime
        print("FPS: ", fps)

        cv2.putText(img_w, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)

        cv2.imshow('MediaPipe Holistic', img_w)


        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()