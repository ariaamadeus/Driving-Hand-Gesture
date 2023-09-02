######################################
import pickle
import pandas as pd

import mediapipe as mp
import cv2
import numpy as np
import os
import csv
import math
import random

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
        self.results = {}
        self.averaging = [0,0,0,0,0,0,0,0,0]
        self.show_image = show_image

    def capIsOpened(self):
        return self.cap.isOpened()
    
    def detectHandNumber(self, shrink = 0, ofset = (0,0)):
               
        ret, image = self.cap.read()
        image = cv2.flip(image, int(self.flip))

        self.image = image.copy()
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        w = image.shape[1]
        h = image.shape[0]
        
        wshrink = int(shrink * w/100)
        hshrink = int(shrink * h/100)

        wsofset = int(ofset[0] * w/100)
        hsofset = int(ofset[1] * h/100)
        image = image[hshrink + hsofset:h-hshrink + hsofset, wshrink + wsofset:w-wshrink + wsofset]
        # image.flags.writeable = False 
        
        resultsHand = self.hands.process(image)
        
        hand_row = 0
        self.results = {}
        # result = ''
        
        if resultsHand.multi_hand_landmarks:
            newResultsHandDict = {}
            # for handLms in resultsHand.multi_hand_landmarks:
            #     newResultsHandDict[handLms.landmark[0].x] = handLms
            # newResultsHandList = sorted(newResultsHandDict)
            
            # print(resultsHand.multi_hand_landmarks[0][0])
            handtypes = []
            for handLms in resultsHand.multi_handedness:
                handtypes.append(handLms.classification[0].label) #Left, Right 
                #terbalik, Left = tg kanan, Right = tg kiri
            for i,handLms in enumerate(resultsHand.multi_hand_landmarks):
                handLmsList = handLms.landmark
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
                # handLm20 = handLmsList[20]
                # handLm19 = handLmsList[19]
                # scale = handLm20.z-handLm0.z

                # print((handLm20.x/(handLm20.z*scale))-(handLm19.x/(handLm19.z*scale))-(handLm0.x/(handLm0.z*scale)))
                # calculate = (handLm20.x/(handLm20.z*scale))-(handLm19.x/(handLm19.z*scale))-(handLm0.x/(handLm0.z*scale))
                # calculate = (handLm20.x/scale)-(handLm19.x/scale)-(handLm0.x/scale)
                # calculate = (handLm20.x-handLm0.x)/scale
                # print(handLm20.z)
                # print(calculate)
                # self.lastvalue = calculate

                # print((handLm20.y/handLm20.z)-(handLm19.y/handLm19.z)-(handLm0.y/handLm0.z))
                # print((handLm20.y*handLm20.z-handLm19.y*handLm19.z))
                hand_row = list(np.array(handLmNew).flatten())
                # hand_row = list(np.array([[landmark.x, landmark.y, landmark.z] for landmark in handLms.landmark]).flatten())
                # print(hand_row)
                if self.show_image:
                    self.mp_drawing.draw_landmarks(image, handLms,
                                                self.mpHands.HAND_CONNECTIONS)
                if self.recognizeTrig == self.recognizeTrigTarget:
                    
                    if handtypes[i] == "Right":
                        if self.flip:
                            # print("Left") #flipped
                            if self.modell != None:
                                df = pd.DataFrame([hand_row])
                                predicted = self.modell.predict(df)[0]
                                self.results[i] = [str(predicted),handLm0.x]
                        else:
                            # print("Right") #flipped
                            if self.modelr != None:
                                df = pd.DataFrame([hand_row])
                                predicted = self.modelr.predict(df)[0]
                                self.results[i] = [str(predicted),handLm0.x]

                    elif handtypes[i] == "Left":
                        if self.flip:
                            # print("Right") #flipped
                            
                            if self.modelr != None:
                                df = pd.DataFrame([hand_row])
                                predicted = self.modelr.predict(df)[0]
                                self.results[i] = [str(predicted),handLm0.x]
                            
                        else:
                            # print("Left") #flipped
                            if self.modell != None:
                                df = pd.DataFrame([hand_row])
                                predicted = self.modell.predict(df)[0]
                                self.results[i] = [str(predicted),handLm0.x]
                                
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

        return self.results, image

    def showSecretKey(self, image, kodeRahasia, fontScale):
        text = str(kodeRahasia) # put in a variable, so that we can measure the size, then put the text
        
        image = self.putText(image = image, text=text, fontScale=fontScale)
        
        return image
    
    def textSize(self, text, fontScale=None, font=None, thickness=None):
        if fontScale == None : fontScale = 3
        if font == None: font = cv2.FONT_HERSHEY_SIMPLEX
        if thickness == None: thickness = fontScale+1
        text = str(text)
        return cv2.getTextSize(text, font, fontScale, thickness)[0]

    def putText(self, image, text, pos = None, fontScale=None, font=None , color=None, color_shadow=None, thickness=None, lineType=None):
        if fontScale == None : fontScale = 3
        if font == None: font = cv2.FONT_HERSHEY_SIMPLEX
        if thickness == None: thickness = fontScale+1
        if color == None: color = (255, 255, 255)
        if color_shadow == None: color_shadow = (0, 0, 255)
        if lineType == None: lineType = cv2.LINE_AA

        textsize = self.textSize(text, font, fontScale, thickness)
        
        if pos == None: #put it in the center
            textX = (image.shape[1] - textsize[0]) / 2
            textY = (image.shape[0] + textsize[1]) / 2
        else:
            textX = pos[0]
            textY = pos[1]
        
        text = str(text)
        image = cv2.putText(image, text, (int(textX)+4, int(textY)+4), font, 
                fontScale, color_shadow, thickness, lineType)
        image = cv2.putText(image, text, (int(textX), int(textY)), font, 
                fontScale, color, thickness, lineType)

        return image
    
    def drawBoxPlace(self, image, hands, ofset, ratio = (3,4) ):
        
        imageShrink = image.copy()
        wshrink = imageShrink.shape[1]
        hshrink = imageShrink.shape[0]
        # print(w,h)
        image = self.image
        w = image.shape[1]
        h = image.shape[0]

        wdif = w-wshrink
        hdif = h-hshrink
        
        wpixel = int(w/hands) 
        hpixel = int(wpixel*ratio[1]/ratio[0])
        wsofset = int(ofset[0] * w/100)
        hsofset = int(ofset[1] * h/100)
        if hpixel > hshrink: #check if the rect can't fit horizontally
            hpixel = hshrink
            wpixel = int(hpixel*ratio[0]/ratio[1])
            startwPos = int(((wdif/2)+(wshrink-(wpixel*hands))/2) + wsofset)
            starthPos = int((hdif/2) + hsofset)
        else: #if the rect fit horizontally
            startwPos = int((wdif/2) + wsofset)
            starthPos = int(((hdif/2)+(hshrink - hpixel)/2) + hsofset)


        
        
        self.startwPos = startwPos
        self.starthPos = starthPos
        # self.wpixel = wpixel
        self.wpixel = 144
        self.hpixel = hpixel

        for x in range(hands):
            startPos = (startwPos, starthPos)
            endPos = (startwPos + wpixel, starthPos + hpixel)
            image = cv2.rectangle(image, startPos, endPos, (255,100,0), 2)
            startwPos+=wpixel

        return image, imageShrink
    
    def rndEquation(self, op):
        doubleMulti = random.randint(0,1)
        op1 = random.choice(op)
        
        if op1 in op[2:]:
            if doubleMulti:
                op2 = random.choice(op[2:])
            else:
                op2 = random.choice(op[:-2])
        else:
            op2 = random.choice(op[2:])
        num1 = random.randint(0,9)
        num2 = random.randint(0,9) if op1 != "/" else random.randint(1,9)
        num3 = random.randint(0,9) if op2 != "/" else random.randint(1,9)
        
        
        while True:
            if op1 == '/' and num1 % num2 != 0:
                num1 = random.randint(0,9)
                num2 = random.randint(1,9)
            else:
                break

        while True:
            if op2 == '/' and num2 % num3 != 0:
                num2 = random.randint(0,9) if op1 != "/" else random.randint(1,9)
                num3 = random.randint(1,9)
            else:
                break

        result = int(eval(f"num1 {op1} num2 {op2} num3"))

        return num1, op1, num2, op2, num3, result
    
    def otherOp(self, eq):
        import itertools

        # Define the target value and a list of numbers
        target_value = eq[5]  # Change this to your desired target value
        numbers = [eq[0], eq[2], eq[4]]  # Change these to your desired numbers

        # Generate all possible permutations of the operators (+, -, *, /)
        operators = ['+', '-', '*', '/']
        operator_combinations = list(itertools.product(operators, repeat=len(numbers) - 1))

        # Function to evaluate an expression
        def evaluate_expression(expression):
            try:
                result = eval(expression)
                return result
            except ZeroDivisionError:
                return None  # Ignore divisions by zero

        # Find combinations that result in the target value
        valid_combinations = []
        for operator_set in operator_combinations:
            expression = f"{numbers[0]}"
            for i, operator in enumerate(operator_set):
                expression += f" {operator} {numbers[i + 1]}"
            result = evaluate_expression(expression)
            if result is not None and result == target_value:
                ans = ''
                for x in expression:
                    if x in operators:
                        ans += x
                valid_combinations.append(ans)
        return valid_combinations
    
    def drawEquation(self, image, imageShrink, eq, ofset):
        w = image.shape[1]
        h = imageShrink.shape[0]
        
        wsofset = int(ofset[0] * w/100)
        hsofset = int(ofset[1] * h/100)

        textsize1 = self.textSize(eq[0])
        textsize2 = self.textSize(eq[2])
        textsize3 = self.textSize(eq[4])
        textsize4 = self.textSize('=')
        textsize5 = self.textSize(eq[5])
        wpixel = self.wpixel
        textX1 = self.startwPos - (textsize1[0]/2) + wsofset
        textX2 = self.startwPos + wpixel - (textsize2[0]/2) + wsofset
        textX3 = self.startwPos + wpixel*2 - (textsize3[0]/2) + wsofset
        textX4 = self.startwPos + wpixel*2.5 - (textsize4[0]/2) + wsofset
        textX5 = self.startwPos + wpixel*3.2 - (textsize5[0]/2) + wsofset
        textY = ((h + textsize1[1]) / 2) + hsofset 
        image = self.putText(image = image, text = eq[0], pos=(textX1,textY))
        image = self.putText(image = image, text = eq[2], pos=(textX2,textY))
        image = self.putText(image = image, text = eq[4], pos=(textX3,textY))
        image = self.putText(image = image, text = "=", pos=(textX4,textY))
        image = self.putText(image = image, text = eq[5], pos=(textX5,textY))
        
        return image
    
    def drawText(self, image, imageShrink, text, fontScale, xhand0, ofset):
        try:

            w = imageShrink.shape[1]
            h = imageShrink.shape[0]

            wsofset = int(ofset[0] * image.shape[1]/100)
            hsofset = int(ofset[1] * image.shape[1]/100)

            xpos = 0
            for i in range(self.maxHands):
                # print(self.wpixel*(self.maxHands-i))
                # print(xhand0*w)
                if xhand0*w < self.wpixel*(i+1):
                    xpos = i
                    break
            
            wpixel = self.wpixel
            # hpixel = self.hpixel
            
            # startPos = (self.startwPos, self.starthPos)

            # endPos = (self.startwPos + self.wpixel, self.starthPos + self.hpixel)
            textsize = self.textSize(text=text)

            textX = self.startwPos + (wpixel*xpos) + (wpixel/2) - (textsize[0]/2) + wsofset
            textY = ((h + textsize[1]) / 2) + hsofset
            self.putText(image=image, text=text, pos=(textX,textY))
            return image
        except Exception as e:
            print(e)
    
    def waitKey(self):
        return chr(cv2.waitKey(1) & 0xFF)
    
    def close(self):
        self.cap.release()
        cv2.destroyAllWindows()


mode = 1 #0: escape room, 1: math equation
if __name__ == "__main__":
    
    angka = '0001'
    kodeRahasia = 254414
    shrink = 25 # 0 -> 100
    ofset = (-10,0) # -50 -> 50
    ofsetbox = (0,20) # -50 -> 50
    maxHands = 2
    flip = True
    show_image = True

    det = HandDetection(maxHands = maxHands, flip = flip, show_image = show_image)
    if mode == 0:
        while det.cap.isOpened():
            results, image = det.detectHandNumber(shrink, ofset)
            
            image, imageShrink = det.drawBoxPlace(image, hands = maxHands)
            for key in results.keys():
                image = det.drawText(image, imageShrink, results[key][0], 2, results[key][1])
            

            cv2.namedWindow("Detect", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("Detect", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            if show_image:
                cv2.imshow("Detect", image)
                if det.waitKey() == 'q': break
    elif mode == 1:
        win = True
        lastframeWin = True
        op = ['+','-','*','/']
        while det.cap.isOpened():
            if win:
                if not lastframeWin:
                    print("reward")
                
                while True:
                    eq = det.rndEquation(op)
                    anss = det.otherOp(eq) # find the suitable ops
                    if len(anss) > 0:
                        break
                win = False
            results, image = det.detectHandNumber(shrink, ofsetbox)
            
            image, imageShrink = det.drawBoxPlace(image, hands = maxHands, ofset = ofsetbox)
            answer = ''
            xpos = 0
            for key in results.keys():
                if 1 <= int(results[key][0]) <= 4:
                    text = op[int(results[key][0])-1]
                    
                    answer += text
                    print(xpos, results[key][1])
                    if results[key][1] < xpos: 
                        print("masuk")
                        answer = answer[::-1]
                    xpos = results[key][1]
                    image = det.drawText(image, imageShrink, text, 2, results[key][1], ofset=ofset)
            if not lastframeWin:
                image = det.drawEquation(image, imageShrink, eq, ofset=ofset)
            else:
                image = det.drawEquation(image, imageShrink, ["LO", "", "AD", "", "I", "NG"], ofset=ofset)
            
            print(anss, answer)
            if answer in anss:
                win = True
                lastframeWin = True
            else:
                lastframeWin = False

            cv2.namedWindow("Detect", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("Detect", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            if show_image:
                cv2.imshow("Detect", image)
                if det.waitKey() == 'q': break