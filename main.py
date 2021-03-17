from win32api import GetSystemMetrics
from pathlib import Path
import cv2
import dlib
import numpy as np
import argparse
from contextlib import contextmanager
from omegaconf import OmegaConf
from tensorflow.keras.utils import get_file
from src.factory import get_model
from PIL import Image, ImageQt
import requests
import os
########################
from PyQt5.QtWidgets import *
from PyQt5.QtCore import QStringListModel, pyqtSignal, pyqtSlot, Qt, QThread, QCoreApplication, QObject, pyqtSignal
from PyQt5 import QtGui 
from PyQt5.QtGui import QIcon, QPixmap, QFont, QFontDatabase
from PyQt5 import QtCore
from PyQt5 import QtWidgets
import time
from datetime import timedelta
########################
try:
    import configparser
except:
    from six.moves import configparser

from configparser import SafeConfigParser

pretrained_model = "https://github.com/yu4u/age-gender-estimation/releases/download/v0.6/EfficientNetB3_224_weights.11-3.44.hdf5"
modhash = '6d7f7b7ced093a8b3ef6399163da6ece'

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray, list, list, list)

    def __init__(self):
        super().__init__()
        self._run_flag = True

    def get_args(self):
        parser = argparse.ArgumentParser(description="This script detects faces from web cam input, "
                                                    "and estimates age and gender for the detected faces.",
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument("--weight_file", type=str, default=None,
                            help="path to weight file (e.g. weights.28-3.73.hdf5)")
        parser.add_argument("--margin", type=float, default=0.4,
                            help="margin around detected face for age-gender estimation")
        parser.add_argument("--image_dir", type=str, default=None,
                            help="target image directory; if set, images in image_dir are used instead of webcam")
        args = parser.parse_args()
        return args


    def draw_label(self, image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
                font_scale=0.8, thickness=1):
        size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        x, y = point
        cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
        cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)


    @contextmanager
    def video_capture(self):
        # cap = cv2.VideoCapture(*args, **kwargs)
        cap = cv2.VideoCapture(0)
        try:
            yield cap
        finally:
            cap.release()

    def yield_images(self):
        # capture video   video_capture(0)
        with self.video_capture() as cap:
            while self._run_flag:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

                while True:
                    # get video frame
                    ret, img = cap.read()
                    if not ret:
                        raise RuntimeError("Failed to capture image")
                    yield img


    def yield_images_from_dir(self, image_dir):
        image_dir = Path(image_dir)

        for image_path in image_dir.glob("*.*"):
            img = cv2.imread(str(image_path), 1)

            if img is not None:
                h, w, _ = img.shape
                r = 640 / max(w, h)
                yield cv2.resize(img, (int(w * r), int(h * r)))
        
    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()

    def showCam(self):
        args = self.get_args()
        weight_file = args.weight_file
        margin = args.margin
        image_dir = args.image_dir

        if not weight_file:
            weight_file = get_file("EfficientNetB3_224_weights.11-3.44.hdf5", pretrained_model, cache_subdir="pretrained_models",
                                file_hash=modhash, cache_dir=str(Path(__file__).resolve().parent))

        # for face detection
        detector = dlib.get_frontal_face_detector()

        # load model and weights
        model_name, img_size = Path(weight_file).stem.split("_")[:2]
        img_size = int(img_size)
        cfg = OmegaConf.from_dotlist([f"model.model_name={model_name}", f"model.img_size={img_size}"])
        model = get_model(cfg)
        model.load_weights(weight_file)

        image_generator = self.yield_images_from_dir(image_dir) if image_dir else self.yield_images()

        for img in image_generator:
            input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_h, img_w, _ = np.shape(input_img)

            # detect faces using dlib detector
            detected = detector(input_img, 1)
            faces = np.empty((len(detected), img_size, img_size, 3))
            lst_ages = []
            cropped_faces = []
            lst_genders = []
            if len(detected) > 0:
                for i, d in enumerate(detected):
                    x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
                    xw1 = max(int(x1 - margin * w), 0)
                    yw1 = max(int(y1 - margin * h), 0)
                    xw2 = min(int(x2 + margin * w), img_w - 1)
                    yw2 = min(int(y2 + margin * h), img_h - 1)
                    img = cv2.rectangle(img, (x1, y1), (x2, y2), (50, 200, 40), 2)
                    cropped_faces.append(img[y1:y2, x1:x2])
                    # cv2.rectangle(img, (xw1, yw1), (xw2, yw2), (255, 0, 0), 2)
                    faces[i] = cv2.resize(img[yw1:yw2 + 1, xw1:xw2 + 1], (img_size, img_size))
                # predict ages and genders of the detected faces
                results = model.predict(faces)
                predicted_genders = results[0]
                ages = np.arange(0, 101).reshape(101, 1)
                predicted_ages = results[1].dot(ages).flatten()
                # draw results
                for i, d in enumerate(detected):
                    label = "{}, {}".format(int(predicted_ages[i]),
                                            "M" if predicted_genders[i][0] < 0.5 else "F")
                    self.draw_label(img, (d.left(), d.top()), label)
                    lst_ages.append(int(predicted_ages[i]))
                    lst_genders.append("M" if predicted_genders[i][0] < 0.5 else "F")
            # cv2.imshow("result", img)
            key = cv2.waitKey(-1) if image_dir else cv2.waitKey(30)
            if key == 27:  # ESC
                break
            self.change_pixmap_signal.emit(img, cropped_faces, lst_ages, lst_genders)

class progressThread(QThread):
    progress_update = pyqtSignal(int) # or pyqtSignal(int)

    def __init__(self):
        QThread.__init__(self)

    def stop(self):
        self.wait()
        self.deleteLater()


    def run(self):
        # your logic here
        counter = 1
        while 1:
            maxVal = 1 # NOTE THIS CHANGED to 1 since updateProgressBar was updating the value by 1 every time
            self.progress_update.emit(maxVal) # self.emit(SIGNAL('PROGRESS'), maxVal)
            # Tell the thread to sleep for 1 second and let other things run
            if counter == 100:
                break
            time.sleep(0.01)
            counter += 1

class MainWindow(QWidget):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        # self.disply_width = 640
        # self.display_height = 480
        self.setWindowIcon(QtGui.QIcon('camera.ico')) 
        self.setWindowTitle("Face Detector")
        self.setMaximumWidth(960)
        self.setMaximumHeight(768)
        self.setMinimumWidth(960)
        self.setMinimumHeight(768)
        self.setGeometry((GetSystemMetrics(0) - 960) / 2, (GetSystemMetrics(1) - 768) / 2, 960, 768)

        self.wdgCamera = QLabel(self)
        self.progressBar = QProgressBar(self)
        self.progressBar.setValue(0)
        self.progressBar.setTextVisible(False)
        self.wdgCamera.setText("Please wait for a while loading camera...")
        self.wdgCamera.setAlignment(QtCore.Qt.AlignCenter)
        self.wdgCamera.setFont(QFont('Roboto', 20)) 

        mainLayout = QVBoxLayout()
        mainLayout.addWidget(self.progressBar)
        mainLayout.addWidget(self.wdgCamera)

        self.progress_thread = progressThread()
        self.progress_thread.start()
        self.progress_thread.progress_update.connect(self.updateProgressBar) # self.connect(self.progress_thread, SIGNAL('PROGRESS'), self.updateProgressBar)
        
        self.prev_img = []
        
        # Config
        parser = SafeConfigParser()
        parser.read('server.config')
        self.test_url = parser.get('SERVER', 'url') + '/uploadFace'
        self.user_email = parser.get('SERVER', 'email')

        self.setLayout(mainLayout)

    def updateProgressBar(self, maxVal):
        self.progressBar.setValue(self.progressBar.value() + maxVal)
        if self.progressBar.value() == 100:
            self.progress_thread.progress_update.disconnect(self.updateProgressBar)
            self.progress_thread.stop()
            self.progressBar.setValue(100)
            self.showCamera()

    def showCamera(self):
        # create the video capture thread
        self.thread = VideoThread()
        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_image)

        # start the thread
        self.thread.showCam()

    def closeEvent(self, event):
        close = QtWidgets.QMessageBox.question(self,
                                        "QUIT",
                                        "Are you sure want to stop process?",
                                        QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        if close == QtWidgets.QMessageBox.Yes:
            sys.exit()
            event.accept()
        else:
            event.ignore()

    def stop(self):
        self.thread.stop()

    @pyqtSlot(np.ndarray, list, list, list)
    def update_image(self, cv_img, cropped_faces, ages, genders):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img, self.wdgCamera.width(), self.wdgCamera.height())
        self.wdgCamera.setPixmap(qt_img)
        if len(cropped_faces) > 0:
            i = 0
            while i < len(cropped_faces):
                img = cropped_faces[i]
                age = ages[i]
                gender = genders[i]

                success, encoded_image = cv2.imencode('.png', img)
                if success == True:
                    test_file = encoded_image.tobytes()
                    try:
                        test_response = requests.post(self.test_url, data = {"age": age, "gender": gender, 'user_email': self.user_email}, files = {"picture": test_file})
                    except requests.exceptions.RequestException as e:  # This is the correct syntax
                        QtWidgets.QMessageBox.information(self,
                                        "Error",
                                        "Server connection was failed. Check server address")
                        raise SystemExit(e)
                i += 1

    def convert_cv_qt(self, cv_img, width, height):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(width, height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())