import socket
import rover
import time

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
port = 12345
s.connect(("192.168.50.185", port))
print("Connected to server")

speed = 60
while True:
    command = s.recv(1024).decode("utf-8")

    if command == "W":
        rover.forward(speed)

    elif commad == "":
        rover.stop()

    if not command:
        print("Connection closed by server.")
        break

    if command.lower() == "terminate":
        break

    output = command

    s.send(output.encode("utf-8"))
    print(output)

s.close()
