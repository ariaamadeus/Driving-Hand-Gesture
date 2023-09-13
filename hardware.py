import serial
from serial.tools.list_ports import comports

class Arduino:
    def __init__(self):
        self.port = self.detectPort()
        self.baudrate = 9600
        self.timeout = 0.01
        self.Serial = serial.Serial(port = self.port, baudrate = self.baudrate, timeout = self.timeout)
        self.connected = False

    def detectPort(self):
        choosedPort = None
        ports = []
        for port in list(comports()):
            
            port = str(port)
            
            if "Arduino" in port:
                ports.append(port.split(' ')[0])
            elif "CP210" in port:
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
            print("Arduino UNO / ESP32 not detected") 
        
        if choosedPort:
            self.connected = True
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

if __name__ == "__main__":
    arduino = Arduino()