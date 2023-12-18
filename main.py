from Detector import *
import cv2
import os
opencv_version = cv2.__version__
# Sürümü yazdır
print("OpenCV Sürümü:", opencv_version)
detector = Detector(use_cuda=True)
#detector.processImage("input_data/faces-banner.jpg")
#detector.processVideo("input_data/vid.mp4")

#detector.saveFaces("output_data")

