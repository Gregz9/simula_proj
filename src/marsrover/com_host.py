import socket

HOST = "192.168.50.108"
PORT = 12345
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print("Socket connected")

try:
    s.bind((HOST, PORT))
except socket.error:
    print("Bind failed")

s.listen(1)
print("Socket awaiting message")
(conn, addr) = s.accept()
print('Connected')

while True:
    data = conn.recv(1024)
    print("I sent a message back in response to: ", data)

    # process your message
    if data == "Hello":
        reply = "Hi, back"
    elif data == "This is important":
        reply = "Important message received"

    elif data == "quit":
        conn.send("Terminating")
        break
    else:
        reply = "Unknown command"

    conn.send(reply)
conn.close()
