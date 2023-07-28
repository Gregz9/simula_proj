import socket
# import rover
import time
import os 
import sys

src_dir = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(src_dir) 

from path_generator.path_tools import * 
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
port = 12345
s.connect(("192.168.50.185", port))
print("Connected to server")

servo_FL = 9
servo_RL = 11
servo_FR = 15
servo_RR = 13
servo_MA = 0

# def readchar():
#     fd = sys.stdin.fileno()
#     old_settings = termios.tcgetattr(fd)
#     try:
#         tty.setraw(sys.stdin.fileno())
#         ch = sys.stdin.read(1)
#     finally:
#         termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
#     if ch == '0x03':
#         raise KeyboardInterrupt
#     return ch
#
# def readkey(getchar_fn=None):
#     getchar = getchar_fn or readchar
#     c1 = getchar()
#     if ord(c1) != 0x1b:
#         return c1
#     c2 = getchar()
#     if ord(c2) != 0x5b:
#         return c1
#     c3 = getchar()
#     return chr(0x10 + ord(c3) - 65)  # 16=Up, 17=Down, 18=Right, 19=Left arrows

def goForward(timestep): 
    rover.forwward(speed)
    time.sleep(timestep)
    rover.stop()

def goReverse(timestep): 
    rover.reverse(speed)
    time.sleep(timestep)
    rover.stop()

def turnLeft(timestep): 
    rover.spinLeft(speed)
    time.sleep(timestep)
    rover.stop()

def turnRight(timestep): 
    rover.spinRight(speed)
    time.sleep(timestep) 
    rover.stop()

Timestep = 5*1e-1
speed = 100
RearDir = 0
FrontDir = 0
MA_dir = 0

speed = 60
try:
    while True:
        command = s.recv(1024).decode("utf-8").lower()
        
        reply = "Received: " + str(command) 


        s.send(reply.encode("utf-8"))
        print(reply)

    s.close()
except socket.error:
    print("Something's wrong")
