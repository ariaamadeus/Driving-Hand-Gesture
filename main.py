from detectMT import HandDetection
from joystick import Pad
from theKeyboard import Keyboard

fontScale = 3 # ukuran font
maxHands = 2 # 1 player
flip = False
show_image = False

det = HandDetection(maxHands = maxHands, flip = flip, show_image = show_image)
pad = Pad()
keyboard = Keyboard()
    
while det.cap.isOpened():
    left, right, image = det.detectHandNumber()
    if right[0]:
        pad.steerValue = right[1]
        pad.gasValue = 1 if right[2] > 0.17 else 0
        
    
    if left[0]:
        keyboard.gear(left[1])
        pad.brakeValue = 1 if left[2] > 0.17 else 0
    pad.update(mode=0)
    print("left:",left)
    print("right:",right)