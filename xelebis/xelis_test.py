import cv2
import mediapipe as mp
import time
import numpy as np

frameWidth = 1920
frameHeight = 1080
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
mpHands=mp.solutions.hands
hands=mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

while True:
    success, img = cap.read()
    img = cv2.flip(img,1)
    img_w = np.empty(img.shape)
    img_w.fill(0)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    #print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks: 
            for id, lm in enumerate(handLms.landmark):
                #print(id, lm)
                h, w, c =img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                #print(id, cx, cy)
                # if id == 4: #(To draw 4th point)
                #cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
            #mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)  # real kamera part


    cv2.imshow('Magari virtualuri xelis display', img)
    #cv2.imshow('Magari xelis display', img) # real kamera
    if cv2.waitKey(1)==27:
        break