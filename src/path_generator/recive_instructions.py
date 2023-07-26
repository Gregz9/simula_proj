# Rover

import rover, time, socket
import sys
import tty
import termios
import json

# speed
speed = 100
time_step = 1

# Servo numbers
servo_FL = 9
servo_RL = 11
servo_FR = 15
servo_RR = 13
servo_MA = 0

# Set up server
HOST = ''  # Symbolic name meaning all available interfaces
PORT = 50007  # Arbitrary non-privileged port
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((HOST, PORT))

def resetServos():
    """Resets the servos to their default positions."""
    rover.setServo(servo_FL, 0)
    rover.setServo(servo_FR, 0)
    rover.setServo(servo_RL, 0)
    rover.setServo(servo_RR, 0)

def goForward():
    """Moves the rover forward by one step (4 cm)."""
    # rover.setServo(servo_FL, 0)
    # rover.setServo(servo_FR, 0)
    # rover.setServo(servo_RL, 0)
    # rover.setServo(servo_RR, 0)
    rover.forward(speed)


def goReverse():
    """Moves the rover in reverse by one step (4 cm)."""
    # rover.setServo(servo_FL, 0)
    # rover.setServo(servo_FR, 0)
    # rover.setServo(servo_RL, 0)
    # rover.setServo(servo_RR, 0)
    rover.reverse(speed)

def spinRight(): 
    """Rotates the rover to the right by one rotation step (90/4 degrees)."""
    rover.setServo(servo_FL, 0)
    rover.setServo(servo_FR, 0)
    rover.setServo(servo_RL, 0)
    rover.setServo(servo_RR, 0)
    rover.spinRight(speed)
    time.sleep(time_step)
    rover.stop()

def spinLeft(): 
    """Rotates the rover to the right by one rotation step (90/4 degrees)."""
    rover.setServo(servo_FL, 0)
    rover.setServo(servo_FR, 0)
    rover.setServo(servo_RL, 0)
    rover.setServo(servo_RR, 0)
    rover.spinLeft(speed)
    time.sleep(time_step)
    rover.stop()


def executeInstructions(instructions):
    """Parses the instructions provided and calls the appropriate functions.

    Args:
        instructions (list of dict): List of instructions where each instruction is a dictionary 
        with 'action' and 'value' as keys. 'action' can be 'Move' or 'Turn' and 'value' is the 
        magnitude of the action.
    """
    for instruction in instructions:
        action = instruction['action']
        value = instruction['value']

        print(f'Executing {action} {value}')

        if action == 'Move':
            moveDistance(value) # we will define this function next
        elif action == 'Turn':
            turnAngle(value) # we will define this function next
        
        time.sleep(time_step) 


def moveDistance(distance):
    """Move the rover a certain distance in cm

    Args:
        distance (float): Distance in centimeters to move.
    """
    print('Moving', distance)
    steps = int(distance / 10)  
    for _ in range(steps):
        goForward()  # Move the rover one step forward
        time.sleep(time_step)  # Adjust this delay as needed, according to your rover's speed
    rover.stop()  # Stop the rover after moving


def turnAngle(angle):
    """Rotate the rover a certain angle in degrees

    Args:
        angle (float): Angle in degrees to rotate. Positive values rotate to the right and negative values to the left.
    """
    print('Turning', angle)
    steps = int(angle / (90/4))  # Convert the angle to steps (one rotation step = 90/4 degrees)
    if steps > 0:  # Positive steps mean turning right
        for _ in range(steps):
            spinRight()  # Rotate the rover one step to the right
            time.sleep(time_step)  # Adjust this delay as needed, according to your rover's rotation speed
    else:  # Negative steps mean turning left
        for _ in range(abs(steps)):
            spinLeft()  # Rotate the rover one step to the left
            time.sleep(time_step)  # Adjust this delay as needed, according to your rover's rotation speed
    rover.stop()  # Stop the rover after rotating


def main():
    """Main function to initialize the rover and the server, and to listen for and execute incoming instructions.
    """
    rover.init(0)
    resetServos()
    print("Waiting for connection...")
    s.listen(1)
    conn, addr = s.accept()
    print('Connected by', addr)
    try:
        while True:
            data = conn.recv(1024)
            if not data: break
            instructions = json.loads(json.loads(data))
            executeInstructions(instructions)
    except KeyboardInterrupt:
        pass
    finally:
        resetServos()
        conn.close()
        rover.cleanup()


if __name__ == "__main__":
    main()
