######################################
import pickle
import pandas as pd

import mediapipe as mp
import cv2
import numpy as np
import os
import csv
import math

class HandDetection:
    def __init__(self, model = "model.pkl", maxHands = 2, flip = False, show_image = True):
        with open("modell.pkl", "rb") as f:
            self.modell = pickle.load(f)
        with open("modelr.pkl", "rb") as f:
            self.modelr = pickle.load(f)
        self.detectionCon = 0.6

        self.flip = flip

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(max_num_hands=maxHands, min_detection_confidence=self.detectionCon)
        self.maxHands = maxHands

        self.mp_drawing = mp.solutions.drawing_utils

        self.cap = cv2.VideoCapture(0)

        self.recognizeTrigTarget = 0
        self.recognizeTrig = self.recognizeTrigTarget
        self.lastResult = 0

        self.boolShowSK = False
        self.lastvalue = 0

        self.wpixel = 0
        self.hpixel = 0
        self.startwPos = 0
        self.starthPos = 0
        self.image = None
        self.results = 0
        self.steering = 0
        self.gas = 0
        self.brake = 0
        self.averaging = [0,0]
        self.show_image = show_image

    def capIsOpened(self):
        return self.cap.isOpened()
    
    def detectHandNumber(self):
        
        ret, frame = self.cap.read()
        self.image = frame.copy()

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False        

        resultsHand = self.hands.process(image)
        
        hand_row = 0
        self.result = 0
        # result = ''
        self.steering = 0
        self.gas = 0
        self.brake = 0
        leftflag = 0
        rightflag = 0
        
        if resultsHand.multi_hand_landmarks:
            handtypes = []
            for handLms in resultsHand.multi_handedness:
                handtypes.append(handLms.classification[0].label) #Left, Right 
                #terbalik, Left = tg kanan, Right = tg kiri
            
            for i,handLms in enumerate(resultsHand.multi_hand_landmarks):
                handLmsList = handLms.landmark
                handLm5 = handLmsList[5]
                handLm17 = handLmsList[17]
                #imagine have a right triangle  a|\c
                #                                |_\
                #                                 b
                a = handLm5.y - handLm17.y
                b = handLm5.x - handLm17.x
                c = (a**2 + b**2)**(1/2) # pythagoras
                
                if self.show_image:
                    self.mp_drawing.draw_landmarks(image, handLms,
                                                self.mpHands.HAND_CONNECTIONS)
                if self.recognizeTrig == self.recognizeTrigTarget:
                    if handtypes[i] == "Right":
                        # print("Left") #flipped
                        leftflag = 1
                        
                        if self.modell != None:
                            handLm0 = handLmsList[0]
                
                            handLmNew = []
                            for handLm in handLms.landmark:
                                scale = (handLm.z-handLm0.z)
                                if scale != 0:
                                    handLmNew.append((handLm.x-handLm0.x)/scale)
                                    handLmNew.append((handLm.y-handLm0.y)/scale)
                                else:
                                    handLmNew.append(1)
                                    handLmNew.append(1)
                                handLmNew.append(1)
                            hand_row = list(np.array(handLmNew).flatten())
                            df = pd.DataFrame([hand_row])
                            predicted = self.modell.predict(df)[0]
                            self.results = predicted
                            self.brake = c # yup, simply measure the distance between 5 - 17, then set a threshold to trigger the brake
                            if self.show_image:
                                image = self.drawText(image, str(predicted), 2, handLm0.x)
                            
                    elif handtypes[i] == "Left":
                        # print("Right") #flipped
                        rightflag = 1

                        #we want to find the angle of a|\c, so that left : negative, right: positive
                        radian = math.atan(b/a)
                        degree = radian*180/math.pi
                        
                        #now we need to modify if a is negative
                        if a > 0:
                            if b < 0:
                                degree = 180 + degree #plus, because in quadran 4, the degree become negative
                            else:
                                degree = -180 + degree #plus, because in quadran 3, the degree become positive

                        self.averaging.append(degree)
                        del self.averaging[0]

                        self.steering = np.average(self.averaging)
                        self.gas = c # also for the gas
                    
                    
            if self.recognizeTrig < self.recognizeTrigTarget:
                self.recognizeTrig += 1
            else:
                self.recognizeTrig = 0
            # if self.flip: result = result[::-1]
        
            # print(result)
            # self.lastResult = result

        if self.show_image:
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        return [leftflag, self.results, self.brake], [rightflag, self.steering, self.gas], image
    
    def drawBoxPlace(self, image, hands, ratio = (3,4)):
        w = image.shape[1]
        h = image.shape[0]
        # print(w,h)

        wpixel = int(w/hands)
        hpixel = int(wpixel*ratio[1]/ratio[0])
        
        if hpixel > h: #check if the rect can't fit horizontally
            hpixel = h
            wpixel = int(hpixel*ratio[0]/ratio[1])
            startwPos = int((w-(wpixel*hands))/2)
            starthPos = 0
        else: #if the rect fit horizontally
            startwPos = 0
            starthPos = int((h - hpixel)/2)
        self.startwPos = startwPos
        self.starthPos = starthPos
        self.wpixel= wpixel
        self.hpixel= hpixel

        for x in range(hands):
            startPos = (startwPos, starthPos)
            endPos = (startwPos + wpixel, starthPos + hpixel)
            image = cv2.rectangle(image, startPos, endPos, (255,100,0), 2)
            startwPos+=wpixel

        return image
    
    def drawText(self, image, text, fontScale, xhand0):
        
        # text = str(kodeRahasia) # put in a variable, so that we can measure the size, then put the text
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = fontScale+1
        
        textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]
        
        # textX = (image.shape[1] - textsize[0]) / 2
        # textY = (image.shape[0] + textsize[1]) / 2

        w = image.shape[1]
        h = image.shape[0]

        xpos = 0
        for i in range(self.maxHands):
            # print(self.wpixel*(self.maxHands-i))
            # print(xhand0*w)
            if xhand0*w < self.wpixel*(i+1):
                xpos = i
                break
        # self.results[xpos] = text
        
        wpixel = self.wpixel
        # hpixel = self.hpixel
        
        # startPos = (self.startwPos, self.starthPos)

        # endPos = (self.startwPos + self.wpixel, self.starthPos + self.hpixel)
        textX = self.startwPos + (wpixel*xpos) + (wpixel/2) - (textsize[0]/2)
        textY = (h + textsize[1]) / 2

        image = cv2.putText(image, text, (int(textX)+4, int(textY)+4), font, 
                fontScale, (0, 0, 255), thickness, cv2.LINE_AA)
        image = cv2.putText(image, text, (int(textX), int(textY)), font, 
                fontScale, (255, 255, 255), thickness, cv2.LINE_AA)
        return image
    
    def waitKey(self):
        return chr(cv2.waitKey(1) & 0xFF)
    
    def close(self):
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    fontScale = 3 # ukuran font
    maxHands = 2 # 1 player
    flip = False
    show_image = False

    det = HandDetection(maxHands = maxHands, flip = flip, show_image = show_image)
    while det.cap.isOpened():
        left, right, image = det.detectHandNumber()
        print("left:",left)
        print("right:",right)
        if show_image:
            cv2.imshow("Detect", image)
            if det.waitKey() == 'q': break