import cv2

import mediapipe as mp

class PoseDetection:
    def __init__(self):

        # Initialize mediapipe pose class.
        self.mp_pose = mp.solutions.pose

        # Setup the Pose function for images - independently for the images standalone processing.
        self.pose_image = self.mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.4)

        # Setup the Pose function for videos - for video processing.
        self.pose_video = self.mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.4,
                                min_tracking_confidence=0.4)

        # Initialize mediapipe drawing class - to draw the landmarks points.
        self.mp_drawing = mp.solutions.drawing_utils

    def detectPose(self, image, pose, draw=False, display=False):
        #x, y. Using basic math. Nanggung kalau pakai model classifier
        sikuri = (0,0)
        sikunan = (0,0)
        gelangri = (0,0)
        gelangnan = (0,0)

        original_image = image.copy()
        copy_image = image.copy()
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resultant = pose.process(image)
        result = ''
        if resultant.pose_landmarks and draw:
            coords = self.mp_drawing.draw_landmarks(image=copy_image, landmark_list=resultant.pose_landmarks, connections=self.mp_pose.POSE_CONNECTIONS, landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(255,255,255),
                                                                                thickness=1, circle_radius=1),
                                    connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0,100,255),
                                                                                thickness=1, circle_radius=1))

            pointsCheck = [False,False,False,False]
            
            if coords.get(13) != None:
                pointsCheck[0] = True
                sikuri = coords[13]
                cv2.circle(original_image, sikuri, 5, (255,255,255), -1)
            if coords.get(14) != None:
                pointsCheck[2] = True
                sikunan = coords[14]
                cv2.circle(original_image, sikunan, 5, (255,255,255), -1)
            if coords.get(15) != None:
                pointsCheck[1] = True #0,1 tangan kiri
                gelangri = coords[15]
                cv2.circle(original_image, gelangri, 5, (255,255,255), -1)
            if coords.get(16) != None:
                pointsCheck[3] = True #2,3 tangan kanan
                gelangnan = coords[16]
                cv2.circle(original_image, gelangnan, 5, (255,255,255), -1)

            if not False in pointsCheck:
                if abs(sikunan[1]-sikuri[1]) < 40 and abs(gelangnan[1]-gelangri[1]) < 40:
                    result = "*"
                elif abs(gelangnan[0]-sikunan[0]) < 200 or abs(gelangri[1]-sikuri[1]) < 200:
                    result = '+'
                elif abs(gelangri[0]-sikuri[0]) < 200 or abs(gelangnan[1]-sikunan[1]) < 200:
                    result = '+'
                cv2.line(original_image, sikuri, gelangri, (0, 100, 255),5)
                cv2.line(original_image, sikunan, gelangnan, (0, 100, 255), 5) 
            elif not False in pointsCheck[:2]: #tangan kiri
                if abs(gelangri[1]-sikuri[1]) < 70:
                    result = "-"
                elif abs(gelangri[0]-sikuri[0]) > 70:
                    result = "/"
                cv2.line(original_image, sikuri, gelangri, (0, 100, 255), 5)
            elif not False in pointsCheck[2:]: #tangan kanan
                if abs(gelangnan[1]-sikunan[1]) < 70:
                    result = "-"
                elif abs(gelangnan[0]-sikunan[0]) > 70:
                    result = "/"
                cv2.line(original_image, sikunan, gelangnan, (0, 100, 255), 5) 
        
        return result, original_image
    
    def drawText(self, image, imageShrink, text, fontScale, xhand0, ofset):
        try:

            w = imageShrink.shape[1]
            h = imageShrink.shape[0]

            wsofset = int(ofset[0] * image.shape[1]/100)
            hsofset = int(ofset[1] * image.shape[1]/100)

            wpixel = 144

            textsize = self.textSize(text=text)

            textX = 50 + (wpixel/2) - (textsize[0]/2) + wsofset
            
            textY = ((h + textsize[1]) / 2) + hsofset
            self.putText(image=image, text=text, pos=(textX,textY))
            return image
        except Exception as e:
            print(e)

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

    def drawEquation(self, image, imageShrink, eq, ofset):
        w = image.shape[1]
        h = imageShrink.shape[0]
        
        wsofset = int(ofset[0] * w/100)
        hsofset = int(ofset[1] * h/100)

        textsize1 = self.textSize(eq[0])
        textsize2 = self.textSize(eq[2])
        textsize3 = self.textSize('=')
        textsize4 = self.textSize(eq[3])
        wpixel = 144
        textX1 = 50 - (textsize1[0]/2) + wsofset
        textX2 = 50 + wpixel - (textsize2[0]/2) + wsofset
        textX3 = 50 + wpixel*1.5 - (textsize3[0]/2) + wsofset
        textX4 = 50 + wpixel*2.3 - (textsize4[0]/2) + wsofset
        textY = ((h + textsize1[1]) / 2) + hsofset
        image = self.putText(image = image, text = eq[0], pos=(textX1,textY))
        image = self.putText(image = image, text = eq[2], pos=(textX2,textY))
        image = self.putText(image = image, text = "=", pos=(textX3,textY))
        image = self.putText(image = image, text = eq[3], pos=(textX4,textY))
        
        return image
    
    def rndEquation(self, op):
        import random
        op1 = random.choice(op)

        num1 = random.randint(0,9)
        num2 = random.randint(0,9) if op1 != "/" else random.randint(1,9)
        
        while True:
            if op1 == '/' and num1 % num2 != 0:
                num1 = random.randint(0,9)
                num2 = random.randint(1,9)
            else:
                break

        result = int(eval(f"num1 {op1} num2"))

        return num1, op1, num2, result

    def otherOp(self, eq):
        import itertools

        # Define the target value and a list of numbers
        target_value = eq[3]  # Change this to your desired target value
        numbers = [eq[0], eq[2]]  # Change these to your desired numbers

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

if __name__ == "__main__":

    cam = cv2.VideoCapture(0)
    det = PoseDetection()
    win = True
    lastframeWin = True
    op = ['+','-','*','/']
    while True:
        if win:
            if not lastframeWin:
                print("reward")
            while True:
                eq = det.rndEquation(op)
                anss = det.otherOp(eq) # find the suitable ops
                if len(anss) > 0:
                    break
            win = False
        _, frame = cam.read()
        frame = cv2.flip(frame,1)
        
        result, frame= det.detectPose(frame, det.pose_image, draw=True, display=False)
        
        if not lastframeWin:
            frame = det.drawEquation(frame, frame, eq, ofset=(15,-35))
        else:
                image = det.drawEquation(frame, frame, ["LO", "", "AD", "ING"], ofset=(15,-35))
        
        if result in anss:
            win = True
            lastframeWin = True
        else:
            lastframeWin = False
        frame = det.drawText(frame, frame, result, 2, 50, ofset=(15,-27))
        cv2.imshow("frame", frame)

        if cv2.waitKey(1) == ord('q'):
            break
