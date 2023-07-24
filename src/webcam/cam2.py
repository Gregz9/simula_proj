import cv2
import matplotlib.pyplot as plt

camera = cv2.VideoCapture(0) #cv2.CAP_DSHOW)

camera.set(cv2.CAP_PROP_FPS, 30.0)
camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc("m", "j", "p", "g"))
camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc("M", "J", "P", "G"))
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

while True:
    retval, im = camera.read()

    cv2.imshow(" ", im)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

camera.release()
cv2.destroyAllWindows()
