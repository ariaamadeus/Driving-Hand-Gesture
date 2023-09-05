from detect import HandDetection
import cv2

shrink = 25 # 0 -> 100
ofset = (-10,0) # -50 -> 50
ofsetbox = (0,20) # -50 -> 50
maxHands = 2
flip = True
show_image = True
det = HandDetection(maxHands = maxHands, flip = flip, show_image = show_image)
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