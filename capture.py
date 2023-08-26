import pickle
import pandas as pd


import mediapipe as mp
import cv2
import numpy as np
import os
import csv

modell = None
modelr = None
try:
    with open('modell.pkl', 'rb') as f:
        modell = pickle.load(f)
except: pass
try:
    with open('modelr.pkl', 'rb') as f:
        modelr = pickle.load(f)
except: pass


angka = 145312

maxHands = 1
detectionCon = 0.5

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=maxHands, min_detection_confidence=detectionCon)

mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

handtype = 'l'
print("Capturing Left Hand")
handtypes2 = 'l'
while cap.isOpened():
    ret, frame = cap.read()
    
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False        

    resultsHand = hands.process(image)
    
    hand_row = 0
    result = ""
    if resultsHand.multi_hand_landmarks:

        handtypes = []
        for handLms in resultsHand.multi_handedness:
            handtypes.append(handLms.classification[0].label) #Left, Right 
            if handLms.classification[0].label == "Right":
                handtypes2 = 'l'
            elif handLms.classification[0].label == "Left":
                handtypes2 = 'r'
            #terbalik, Left = tg kanan, Right = tg kiri

        for i, handLms in enumerate(resultsHand.multi_hand_landmarks):
            handLmsList = handLms.landmark
            handLm0 = handLmsList[0]
            handLmNew = []
            for handLm in handLms.landmark:
                scale = handLm.z-handLm0.z
                if scale != 0:
                    handLmNew.append((handLm.x-handLm0.x)/scale)
                    handLmNew.append((handLm.y-handLm0.y)/scale)
                else:
                    handLmNew.append(1)
                    handLmNew.append(1)
                handLmNew.append(1)
            # hand_row = list(np.array([[landmark.x, landmark.y, landmark.z] for landmark in handLms.landmark]).flatten())
            hand_row = list(np.array(handLmNew).flatten())
            mp_drawing.draw_landmarks(image, handLms,
                                        mpHands.HAND_CONNECTIONS)
            # print(handtypes[i] == "Left")
            if handtypes[i] == "Right":
                print("Left") #flipped
                if modell != None:
                    df = pd.DataFrame([hand_row])
                    predicted = modell.predict(df)[0]
                    result+=str(predicted)
            elif handtypes[i] == "Left":
                print("Right") #flipped
                if modelr != None:
                    df = pd.DataFrame([hand_row])
                    predicted = modelr.predict(df)[0]
                    result+=str(predicted)
        # if model != None: print(result)
    if result != '': print(result)
    image.flags.writeable = True   
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    
    cv2.imshow('Raw Webcam Feed', image)

    key = cv2.waitKey(1) & 0xFF
    if key in [ord(x) for x in['1','2','3','4','5','6','7','8','9','0','q','n','l','r']]:
        if chr(key) == 'q':
            break
        elif chr(key) == 'l':
            handtype = 'l'
            print("Capturing Left Hand")
        elif chr(key) == 'r':
            handtype = 'r'
            print("Capturing Right Hand")
        elif chr(key) == 'n':
            create = input(f"Want to replace a new coords{handtype}.csv?(Y/[n]):")
            if create == "Y":
                landmarks = ['class']
                for val in range(1, 21+1):
                    landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val)]
                with open(f'coords{handtype}.csv', mode='w', newline='') as f:
                    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    csv_writer.writerow(landmarks)
                print(f"coords{handtype}.csv Created")
        elif hand_row != 0:
            if handtypes2 == handtype:
                hand_row.insert(0,chr(key))
                if os.path.exists(f'coords{handtype}.csv'):
                    with open(f'coords{handtype}.csv', mode='a', newline='') as f:
                        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                        csv_writer.writerow(hand_row)
                    print("Angka",chr(key),"disimpan")
                else:
                    print(f"Please Create a New coords{handtype}.csv first with pressing 'n'")
            else:
                print(f"Please use your other hand, currently capturing {handtype} hand")
cap.release()
cv2.destroyAllWindows()
