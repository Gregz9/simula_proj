import cv2 as cv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output", required=True, type=str, help="Name of the file to be saved by the program")
args = parser.parse_args()

cam_port = 0
camera = cv.VideoCapture(cam_port)

camera.set(cv.CAP_PROP_FPS, 30.0)
camera.set(cv.CAP_PROP_FOURCC, cv.VideoWriter.fourcc("m", "j", "p", "g"))
camera.set(cv.CAP_PROP_FOURCC, cv.VideoWriter.fourcc("M", "J", "P", "G"))
camera.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
camera.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)

result, image = camera.read()

if result:

    cv.imshow(" ", image)
    cv.imwrite(str(args.output) + ".jpg", image)

    cv.waitKey(0)
    cv.destroyWindow(" ")

else:
    print("No image detected")
