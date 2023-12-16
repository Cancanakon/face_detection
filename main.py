from Detector import *
import cv2

opencv_version = cv2.__version__
# Sürümü yazdır
print("OpenCV Sürümü:", opencv_version)
detector = Detector(use_cuda=True)
#detector.processImage("input_data/IMG_5967.jpg")
detector.processVideo("input_data/WDBP2177.MOV")


