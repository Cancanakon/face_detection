import numpy as np
import cv2
import os

class Detector:
    def __init__(self, use_cuda=False):
        self.faceModel = cv2.dnn.readNetFromCaffe("models/res10_300x300_ssd_iter_140000.prototxt",
                                                  caffeModel="models/res10_300x300_ssd_iter_140000.caffemodel")

        if use_cuda:
            self.faceModel.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.faceModel.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        self.faces_count = 0  # Yüz sayısını takip etmek için faces_count özelliği eklenmiştir.

    def processImage(self, imgName):
        self.img = cv2.imread(imgName)
        (self.height, self.width) = self.img.shape[:2]

        # Yüzleri kaydet ve dosya adlarını al
        faces = self.saveFaces(os.path.join('output', 'images'))
        self.faces_count = len(faces)

    def processVideo(self, videoName):
        cap = cv2.VideoCapture(videoName)

        if not cap.isOpened():
            print("ERROR!")
            return
        (success, self.img) = cap.read()
        (self.height, self.width) = self.img.shape[:2]
        fps = cv2.getTickFrequency()

        while success:
            self.processFrame()
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - fps)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            success, self.img = cap.read()

        print("E-TIME: {:.2f}".format(fps))

        cap.release()
        cv2.destroyAllWindows()

    def processFrame(self):
        blob = cv2.dnn.blobFromImage(self.img, 1.0, (300, 300), (104.0, 177.0, 123.0),
                                     swapRB=False, crop=False)
        self.faceModel.setInput(blob)
        predictions = self.faceModel.forward()

        for i in range(0, predictions.shape[2]):
            if predictions[0, 0, i, 2] > 0.5:
                bbox = predictions[0, 0, i, 3:7] * np.array([self.width, self.height, self.width, self.height])
                (xmin, ymin, xmax, ymax) = bbox.astype("int")

                cv2.rectangle(self.img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
                self.faces_count += 1  # Her bir yüz tespiti olduğunda faces_count artırılır.

    def saveFaces(self, output_folder="output_faces"):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        faces = []

        blob = cv2.dnn.blobFromImage(self.img, 1.0, (300, 300), (104.0, 177.0, 123.0),
                                     swapRB=False, crop=False)
        self.faceModel.setInput(blob)
        predictions = self.faceModel.forward()

        for i in range(0, predictions.shape[2]):
            confidence = predictions[0, 0, i, 2]
            if confidence > 0.5:
                bbox = predictions[0, 0, i, 3:7] * np.array([self.width, self.height, self.width, self.height])
                (xmin, ymin, xmax, ymax) = bbox.astype("int")

                # Yüzü kare içine al
                face_roi = self.img[ymin:ymax, xmin:xmax]

                # Yüz boyutlarını kontrol et
                if face_roi.shape[0] > 0 and face_roi.shape[1] > 0:
                    # Görüntü boyutları boş değilse resize işlemi yap
                    resized_face_roi = cv2.resize(face_roi, (100, 100))  # İstediğiniz boyuta uygun olarak ayarlayın

                    # Yüzü kaydet
                    face_filename = os.path.join(output_folder, f"face_{i}.jpg")
                    cv2.imwrite(face_filename, resized_face_roi)
                    faces.append(face_filename)

                    # Yüzü çerçeve içine al
                    cv2.rectangle(self.img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
                    self.faces_count += 1  # Her bir yüz tespiti olduğunda faces_count artırılır.

        return faces
