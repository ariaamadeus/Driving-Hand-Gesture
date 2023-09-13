from pose import PoseDetection
import cv2
import time
from threading import Thread

from hardware import Arduino
from imanip import draw_overlay, resize_scale
from game import Game

arduino = Arduino()
game = Game()
soundThread = Thread(target=game.play_sound)

cam = cv2.VideoCapture(0)
det = PoseDetection()
win = True
lastframeWin = True

#initial
ori_backdrop = cv2.imread("images/back.png", -1)

check_true = resize_scale(cv2.imread("images/v1.png",-1), 40)
check_false = resize_scale(cv2.imread("images/v0.png",-1),40)
congrats = [resize_scale(cv2.imread("images/congrats.png",-1), 100),
            resize_scale(cv2.imread("images/congratsGlow.png",-1), 100)]
raiseHand = resize_scale(cv2.imread("images/Raise.png",-1), 100)
go = resize_scale(cv2.imread("images/Go.png",-1), 100)



op = {'+':None,'-':None,'*':None,'/':None}
equals = cv2.imread("images/=.png", -1)
equals = resize_scale(equals, 50)

num = []
for i in range(0,10):
    im = cv2.imread(f"images/{i}.png",-1)
    im = resize_scale(im,50)
    num.append(im)

for operator in list(op.keys()):
    theOp = f"{operator}"
    if operator == "/": theOp = "div"
    elif operator == "*": theOp = "x"
    im = cv2.imread(f"images/{theOp}.png",-1)
    im = resize_scale(im,40)
    op[operator] = im.copy()

backdrop = ori_backdrop.copy()
three_win = 0

starttime = time.time()
optimer = 0
last_result = ''
last_true_result = ''

checks_x = [106, 318, 530]

in_game = False


animate_timer = 0
animate_counter = 6
animate_bool = 0
done = False
give = True

def win_animate(frame):
    global animate_timer, animate_bool, animate_counter, check_false, check_true, done
    if animate_counter:
        if time.time() - animate_timer > 0.7:
            for x in range(3):
                if animate_bool:
                    frame = draw_overlay(frame, (checks_x[x] - check_false.shape[1]/2) , 10, check_false)
                    frame = draw_overlay(frame, (frame.shape[1]/2 - congrats[1].shape[1]/2) , 100, congrats[1])
                else:
                    frame = draw_overlay(frame, (checks_x[x] - check_true.shape[1]/2) , 10, check_true)
                    frame = draw_overlay(frame, (frame.shape[1]/2 - congrats[0].shape[1]/2) , 100, congrats[0])
            animate_bool = not animate_bool
            animate_counter -= 1
            animate_timer = time.time()
    else:
        for x in range(3):
            frame = draw_overlay(frame, (checks_x[x] - check_false.shape[1]/2) , 10, check_false)
        done = True
        animate_counter = 6
    return done

def count_animate(frame):
    global animate_timer, animate_bool, animate_counter, check_false, check_true, done
    if animate_counter:
        if animate_counter-1:
            frame = draw_overlay(frame, (frame.shape[1]/2 - num[animate_counter-1].shape[1]/2) , 100, num[animate_counter-1])
        else:
            frame = draw_overlay(frame, (frame.shape[1]/2 - go.shape[1]/2) , 100, go)
        if time.time() - animate_timer > 1:
            animate_counter -= 1
            animate_timer = time.time()
    else:
        for x in range(3):
            frame = draw_overlay(frame, (checks_x[x] - check_false.shape[1]/2) , 10, check_false)
        done = True
        animate_counter = 6
    return done

firstPlay = False
while True:

    # Detect
    _, frame = cam.read()
    frame = cv2.flip(frame,1)
    result, frame = det.detectPose(frame, det.pose_image, draw=True, display=False)
    if (result == "start" or result == "/") and not in_game:
        in_game = True
        give = True
        firstPlay = True
        animate_counter = 4
        animate_timer = 0
    # Draw Check
    for x in range(3):
        if three_win <= x:
            frame = draw_overlay(frame, (checks_x[x] - check_false.shape[1]/2) , 10, check_false)
        else:
            frame = draw_overlay(frame, (checks_x[x] - check_false.shape[1]/2) , 10, check_true)

    if firstPlay:
        if count_animate(frame): #- > done
            firstPlay = False
            done = False
    elif in_game :
        
        if result == "start": result = '/'
        if win:
            while True:
                eq = det.rndEquation(list(op.keys()))
                anss = det.otherOp(eq) # find the suitable ops
                if last_true_result in anss:
                    continue
                elif len(anss) > 0:
                    break
            win = False
        
        

        optimer = time.time() # Stay with the same operator to answer
        if three_win == 3:
            if give: #give reward
                if arduino.connected:
                    arduino.write("1")
                soundThread.start()
                give = False
            done = win_animate(frame)
            if done: #done celebration
                three_win = 0
                in_game = False
                done = False
        elif not lastframeWin:
            # Draw Equation
            x1,x2,x3,x4,y = det.drawEquation(frame, frame, eq, ofset=(15,-35))
            
            frame = draw_overlay(frame, x1[1],y, num[x1[0]]) #num1
            frame = draw_overlay(frame, x2[1],y, num[x2[0]]) #num2
            frame = draw_overlay(frame, x3, y, equals) # =
            
            if x4[0]>9: # 2 digits result
                x41 = int(str(x4[0])[0])
                x42 = int(str(x4[0])[1])
                frame = draw_overlay(frame, x4[1],y, num[x41])
                frame = draw_overlay(frame, x4[1] + num[x42].shape[1], y, num[x42])
            else:
                frame = draw_overlay(frame, x4[1],y, num[x4[0]])
            # else:
                    # image = det.drawEquation(frame, frame, ["LO", "", "AD", "ING"], ofset=(15,-35))
        if result!="":
            # draw the gesture
            if last_result == result:
                optimer = time.time() - starttime
            else:
                last_result = result
                optimer = 0
                starttime = time.time()
            x,y = det.drawText(frame, frame, result, 2, 50, ofset=(15,-25))
            frame = draw_overlay(frame, x, y, op[result])

        if result in anss:
            if optimer > 1.5: # Stay with the same operator more than 1.5s to answer
                last_true_result = result
                starttime = time.time()
                three_win += 1
                win = True
                lastframeWin = True
        else:
            lastframeWin = False

    else: #not in game
        frame = draw_overlay(frame, (frame.shape[1]/2 - raiseHand.shape[1]/2) , 100, raiseHand)   
        
    
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)
    frame = cv2.resize(frame,(1024,768), interpolation = cv2.INTER_AREA)
    
    # 448, 156 is the start (0, 0) of top left of frame, so the frame is in the center
    backdrop = draw_overlay(backdrop, 448, 156, frame)

    cv2.namedWindow("frame", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("frame", backdrop)
    
    if cv2.waitKey(1) == ord('q'):
        break