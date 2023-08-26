import pyautogui

class Keyboard:
    def __init__(self):
        self.lastPressed = 0

    def gear(self, value):
        if self.lastPressed != value:
            if 0 <= value <= 6:
                pyautogui.keyUp(self.lastPressed)
                pyautogui.keyDown(str(value))
                self.lastPressed = value