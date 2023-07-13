from __future__ import annotations
import cv2 as cv
import argparse

ap = argparse.ArgumentParser(description="Program to filter out the background")
ap.add_argument(
    "-i", "--input", type=str, required=True, help="Path to the input image."
)
ap.add_argument(
    "-alg",
    "--algorithm",
    type=str,
    required=True,
    help="Algorithm to be used for background extaction (KNN, MOG2).",
    default="MOG2",
)

if args.algo == "MOG2":
    backSub = cv.createBackgroundSubtractorMOG2()
else:
    backSub = sv.createBackgroundSubtractorKNN()

capture = cv.VideoCapture(cv.samples.findFileOrKeep(args.input))
if not capture.isOpened():
    print("Unable to open: " + args.input)
    exit(0)

while True:
    ret, frame = capture.read()
    if frame is None:
        break
