import socket

s = socket.socket()
print("Socket succesfully created")

port = 12345
s.bind(("", port))
print("socket binded to %s" % (port))

s.listen(5)
print("socket is listening")

command = ""
while command != "Terminate":
    c, addr = s.accept()
    print("Got connection from", addr)

    command = str(input("Enter command: "))
    c.send(command.encode())
    continue
c.close()
