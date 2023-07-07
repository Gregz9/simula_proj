# Mars Rover Simple Drive Mode
# Similar to motortest.py but integrates servo steering
# Moves: Forward, Reverse, turn Right, turn Left, Stop 
# Press Ctrl-C to stop

from __future__ import print_function
import rover, time

#======================================================================
# Reading single character by forcing stdin to raw mode
import sys
import tty
import termios
import pygame

# Servo numbers
servo_FL = 9
servo_RL = 11
servo_FR = 15
servo_RR = 13
servo_MA = 0

def readchar():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    if ch == '0x03':
        raise KeyboardInterrupt
    return ch

def readkey(getchar_fn=None):
    getchar = getchar_fn or readchar
    c1 = getchar()
    if ord(c1) != 0x1b:
        return c1
    c2 = getchar()
    if ord(c2) != 0x5b:
        return c1
    c3 = getchar()
    return chr(0x10 + ord(c3) - 65)  # 16=Up, 17=Down, 18=Right, 19=Left arrows

# End of single character reading
#======================================================================

def goForward():
    # rover.setServo(servo_FL, 0)
    # rover.setServo(servo_FR, 0)
    # rover.setServo(servo_RL, 0)
    # rover.setServo(servo_RR, 0)
    rover.forward(speed)
    time.sleep(1e-1)
    rover.stop()

def goReverse():
    # rover.setServo(servo_FL, 0)
    # rover.setServo(servo_FR, 0)
    # rover.setServo(servo_RL, 0)
    # rover.setServo(servo_RR, 0)
    rover.reverse(speed)
    time.sleep(1e-1)
    rover.stop()

def goLeft():
    rover.setServo(servo_FL, FrontDir)
    rover.setServo(servo_FR, FrontDir)
    rover.setServo(servo_RL, RearDir)
    rover.setServo(servo_RR, RearDir)

def sidewaysLeftForward():
    rover.setServo(servo_FL, -40)
    rover.setServo(servo_FR, -40)
    rover.setServo(servo_RL, -40)
    rover.setServo(servo_RR, -40)
    rover.forward(speed)
    time.sleep(1e-1)
    rover.stop()

def sidewaysRightForward():
    rover.setServo(servo_FL, 40)
    rover.setServo(servo_FR, 40)
    rover.setServo(servo_RL, 40)
    rover.setServo(servo_RR, 40)
    rover.forward(speed)
    time.sleep(1e-1)
    rover.stop()

def sidewaysLeftReverse():
    rover.setServo(servo_FL, 40)
    rover.setServo(servo_FR, 40)
    rover.setServo(servo_RL, 40)
    rover.setServo(servo_RR, 40)
    rover.reverse(speed)
    time.sleep(1e-1)
    rover.stop()

def sidewaysRightReverse():
    rover.setServo(servo_FL, -40)
    rover.setServo(servo_FR, -40)
    rover.setServo(servo_RL, -40)
    rover.setServo(servo_RR, -40)
    rover.reverse(speed)
    time.sleep(1e-1)
    rover.stop()


def driveLeft():
    rover.setServo(servo_FL, -20)
    rover.setServo(servo_FR, -20)
    rover.setServo(servo_RL, 20)
    rover.setServo(servo_RR, 20)
    rover.forward(speed)
    time.sleep(1e-1)
    rover.stop()

def driveLeftB():
    rover.setServo(servo_FL, -20)
    rover.setServo(servo_FR, -20)
    rover.setServo(servo_RL, 20)
    rover.setServo(servo_RR, 20)
    rover.reverse(speed)
    time.sleep(1e-1)
    rover.stop()


def driveRight():
    rover.setServo(servo_FL, 20)
    rover.setServo(servo_FR, 20)
    rover.setServo(servo_RL, -20)
    rover.setServo(servo_RR, -20)
    rover.forward(speed)
    time.sleep(1e-1)
    rover.stop()

def driveRightB():
    rover.setServo(servo_FL, 20)
    rover.setServo(servo_FR, 20)
    rover.setServo(servo_RL, -20)
    rover.setServo(servo_RR, -20)
    rover.reverse(speed)
    time.sleep(1e-1)
    rover.stop()

def goRight():
    rover.setServo(servo_FL, FrontDir)
    rover.setServo(servo_FR, FrontDir)
    rover.setServo(servo_RL, RearDir)
    rover.setServo(servo_RR, RearDir)

def spinRight(): 
    rover.setServo(servo_FL, 0)
    rover.setServo(servo_FR, 0)
    rover.setServo(servo_RL, 0)
    rover.setServo(servo_RR, 0)
    rover.spinRight(speed)

def spinLeft(): 
    rover.setServo(servo_FL, 0)
    rover.setServo(servo_FR, 0)
    rover.setServo(servo_RL, 0)
    rover.setServo(servo_RR, 0)
    rover.spinLeft(speed)

def turnMA(): 
    rover.setServo(servo_MA, MA_dir)

def measDistance(): 
    dist = rover.getDistance()
    print("Distance: ", (int(dist * 10)) / 10.0)
    return dist

speed = 60
RearDir = 0
FrontDir = 0
MA_dir = 0 

print ("Drive M.A.R.S. Rover around")
print ("Use arrow keys to steer")
print ("Use , or < to slow down")
print ("Use . or > to speed up")
print ("Press space bar to coast to stop")
print ("Press b to brake and stop quickly")
print ("Press Ctrl-C to end")
print

rover.init(0)

# main loop
try:
    while True:
        keyp = readkey()
        dist = measDistance()
        if keyp == 'w' or ord(keyp) == 16:
            goForward()
            # print ('Forward', speed)
        elif keyp == 's' or ord(keyp) == 17:
            goReverse()
            # print ('Reverse', speed)
        elif keyp == 'd' or ord(keyp) == 18:
            RearDir = RearDir - 20 if RearDir > -40 else RearDir 
            FrontDir = FrontDir + 20 if FrontDir < 40 else FrontDir 
            goRight()
            # print ('Go Right', speed)
        elif keyp == 'a' or ord(keyp) == 19:
            RearDir = RearDir + 20 if RearDir < 40 else RearDir 
            FrontDir = FrontDir - 20 if FrontDir > -40 else FrontDir 
            goLeft()
            # print ('Go Left', speed)
        elif keyp == 'q': 
            driveLeft()
            # print ('Go Left', speed)
        elif keyp == 'e': 
            driveRight()
            # print ('Go Left', speed)
        elif keyp == 'z': 
            driveLeftB()
            # print ('Go Left', speed)
        elif keyp == 'c': 
            driveRightB()
            # print ('Go Right', speed)
        elif keyp == 'r': 
            spinRight()
        elif keyp == 'l':
            spinLeft()
        elif keyp == '7':
            sidewaysLeftForward()
        elif keyp == '9':
            sidewaysRightForward()
        elif keyp == '1': 
            sidewaysLeftReverse()
        elif keyp == '3': 
            sidewaysRightReverse()
        # Turning the mast left by 5 degrees
        elif keyp == '4':
            MA_dir = MA_dir + 5 if MA_dir < 70 else MA_dir 
            turnMA()
        elif keyp == '6':
            MA_dir = MA_dir - 5 if MA_dir > -70 else MA_dir 
            turnMA()
        elif keyp == '.' or keyp == '>':
            speed = min(100, speed+10)
            # print ('Speed+', speed)
        elif keyp == ',' or keyp == '<':
            speed = max(0, speed-10)
            # print ('Speed-', speed)
        elif keyp == ' ':
            rover.stop()
            print ('Stop')
        elif keyp == 'b':
            rover.brake()
            rover.setServo(servo_FL, 0)
            rover.setServo(servo_FR, 0)
            rover.setServo(servo_RL, 0)
            rover.setServo(servo_RR, 0)
            rover.setServo(servo_MA, 0)
            print ('Brake')
        elif ord(keyp) == 3:
            break
        print("Distance: ", (int(dist * 10)) / 10.0)
        print("Current speed: ", speed)

except KeyboardInterrupt:
    pass

finally:
    rover.cleanup()
    
