import time

import serial
from serial.tools.list_ports import comports

import vgamepad as vg

class Arduino:
    def __init__(self):
        self.port = self.detectPort()
        self.baudrate = 115200
        self.timeout = 0.01
        self.Serial = serial.Serial(port = self.port, baudrate = self.baudrate, timeout = self.timeout)

    def detectPort(self):
        choosedPort = None
        ports = []
        for port in list(comports()):
            port = str(port)
            
            if "Arduino" in port:
                ports.append(port.split(' ')[0])
        
        if len(ports) > 1:
            message = "Choose Port:\n"
            for i, port in enumerate(ports):
                message += f"  {i}:{port}\n"
            message += "(example:0) :"
            choosedPort = ports[input(message)]
        elif len(ports) == 1:
            choosedPort = ports[0]    
        else:
            raise("Arduino UNO not detected") 
        
        print(f"Using port: {choosedPort}")
        return choosedPort

    def write(self, msg:str) -> bool:
        return self.Serial.write(bytes(msg, "utf-8"))

    def read(self):
        bmsg = self.Serial.readline()
        try:
            msg = str(bmsg.decode())[:-2]
            inputType = msg[0]
            msg = msg[1:]
            if msg.isdigit():
                msg = int(msg)
            return inputType, msg
        except:
            print(f"Skip message:{bmsg}")
            return self.read()

class Pad:
    def __init__(self):
        self.gamepad = vg.VX360Gamepad()
        self.steerValue = 0
        self.gasValue = 0
        self.brakeValue = 0

    def allInput(self, msg):
        inputType = msg[0]
        value = msg[1]
        if inputType == 's':
            self.steerValue = value
        elif inputType == 'g':
            self.gasValue = value
        else:
            self.brakeValue = value

    def fMap(self, value):
        # x1, x2, y1, y2 = (0,1023, -1, 1)
        x1, x2, y1, y2 = (-100, 100, -1, 1) # hand detection
        x = ((value - x1) * (y2 - y1) / (x2 - x1)) + y1
        return x
    
    def fMapgb(self, value):
        x1, x2, y1, y2 = (0,1023, 0, 1)
        x = ((value - x1) * (y2 - y1) / (x2 - x1)) + y1
        return x

    def steer(self, value):
        self.gamepad.left_joystick_float(x_value_float = self.fMap(value), y_value_float = 0)
    
    def gas(self, value):
        # self.gamepad.right_trigger_float(value_float = self.fMapgb(value))
        self.gamepad.right_trigger_float(value_float = value) #hand detection
    
    def brake(self, value):
        #self.gamepad.right_joystick_float(x_value_float = 0, y_value_float = self.fMapgb(value))
        self.gamepad.right_joystick_float(x_value_float = 0, y_value_float = value) # hand detection
        #self.gamepad.left_trigger_float(value_float = self.fMapgb(value))


    #def gas_brake(self, gasValue, brakeValue):
    #    self.gamepad.right_joystick_float(x_value_float = self.fMap(gasValue), y_value_float = self.fMap(brakeValue))

    def update(self):
        self.steer(self.steerValue)
        self.gas(self.gasValue)
        self.brake(self.brakeValue)
        #self.gas_brake(self.gasValue, self.brakeValue)
        self.gamepad.update()

if __name__ == "__main__":
    arduino = Arduino()
    pad = Pad()
    while True:
        pad.allInput(arduino.read())
        pad.update()
