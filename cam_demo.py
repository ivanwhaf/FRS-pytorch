import argparse
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtWidgets import QPushButton, QApplication, QLabel, QMainWindow, QMessageBox

from detect import predict_class_name_and_confidence
from models import build_model
from utils import load_prices, parse_cfg

parser = argparse.ArgumentParser(description='Food Recognition System')
parser.add_argument("--cfg", "-c", help="Your config file path", type=str, default="cfg/frs.cfg")
parser.add_argument("--price", "-p", help="Your price file path", type=str, default="cfg/prices.cfg")
parser.add_argument("--weights", "-w", help="Path of model weight", type=str, default="weights/frs_cnn.pth")
parser.add_argument("--cam_width", "-cw", help="Camera width", type=int, default=848)
parser.add_argument("--cam_height", "-ch", help="Camera height", type=int, default=480)
parser.add_argument("--font_type", "-ft", help="Path of font type", type=str, default="data/simsun.ttc")
args = parser.parse_args()


class Window(QMainWindow):
    """
    customized Qt window
    """

    def __init__(self, weight_path, cfg, cam_width, cam_height):
        super(Window, self).__init__()
        self.setGeometry(500, 300, cam_width, cam_height + 150)
        self.setFixedSize(cam_width, cam_height + 150)
        self.setWindowTitle('Food Recognition System')

        self.img_label = QLabel(self)
        self.img_label.setGeometry(0, 0, cam_width, cam_height)

        self.dish_label = QLabel(self)
        self.dish_label.move(50, cam_height + 25)
        self.dish_label.resize(450, 35)
        self.dish_label.setText("菜品名称：")
        self.dish_label.setFont(QFont("Roman times", 16, QFont.Bold))

        self.price_label = QLabel(self)
        self.price_label.move(50, cam_height + 70)
        self.price_label.resize(450, 35)
        self.price_label.setText("金额：")
        self.price_label.setFont(QFont("Roman times", 16, QFont.Bold))

        self.statusbar = self.statusBar()

        self.frame = None
        self.isChecking = False

        check_button = QPushButton("结算", self)
        check_button.move(500, cam_height + 50)
        check_button.resize(130, 40)
        check_button.clicked.connect(self.check)  # Check Button

        confirm_button = QPushButton("确定", self)
        confirm_button.move(650, cam_height + 50)
        confirm_button.resize(130, 40)
        confirm_button.clicked.connect(self.confirm)  # Confirm Button

        # camera init
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, cam_width)
        self.cap.set(4, cam_height)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self.update_frame)
        self._timer.start(33)  # 30fps

        # self.setCentralWidget(self.img_label)
        self.show()

        self.cfg = cfg
        self.model = build_model(
            weight_path, self.cfg)
        print('Model successfully loaded!')
        self.statusbar.showMessage("Model successfully loaded!")
        self.prices = load_prices(args.price)

    def update_frame(self):
        # get camera frame and convert to pixmap to show on img label
        if self.isChecking:
            return

        if not self.cap.isOpened():
            print('Camera not open!')
            self.statusbar.showMessage("Camera not open!")
            return

        _, self.frame = self.cap.read()  # read camera frame

        self.statusbar.showMessage(
            'Frame shape: ' + str(self.frame.shape))
        frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        img = QImage(frame, w, h, QImage.Format_RGB888)
        img = QPixmap.fromImage(img)
        self.img_label.setPixmap(img)  # show on img label
        self.img_label.setScaledContents(True)  # self adaption

    def check(self):
        # check function, draw class name,confidence and price on the image
        if self.isChecking:
            return

        frame = self.frame

        if frame is None:
            print('Frame is none!')
            reply = QMessageBox.information(self, 'Warning!', '摄像头未打开！', QMessageBox.Ok,
                                            QMessageBox.Ok)
            if reply == QMessageBox.Ok:
                pass
            return

        class_name, confidence = predict_class_name_and_confidence(
            frame, self.model, int(self.cfg['input_size']))

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)  # ndarray to pil Image

        draw = ImageDraw.Draw(img)
        font_text = ImageFont.truetype(args.font_type, 26, encoding="utf-8")
        draw.text((5, 5), class_name, (0, 255, 0), font=font_text)
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        print('Class name:', class_name, 'Confidence:', str(confidence) + '%')
        self.statusbar.showMessage(
            'Class name: ' + class_name + ' Confidence: ' + str(confidence) + '%')

        h, w = img.shape[:2]
        img = QImage(img, w, h, QImage.Format_RGB888)
        img = QPixmap.fromImage(img)
        self.img_label.setPixmap(img)  # show on img label
        self.img_label.setScaledContents(True)  # self adaption
        self.isChecking = True
        self.dish_label.setText("菜品名称：" + class_name)
        self.price_label.setText("金额：" + self.prices[class_name] + "元")

    def confirm(self):
        self.isChecking = False
        self.dish_label.setText("菜品名称：")
        self.price_label.setText("金额：")


if __name__ == '__main__':
    weight_path, cfg_path, cam_width, cam_height = args.weights, args.cfg, args.cam_width, args.cam_height
    cfg = parse_cfg(cfg_path)

    app = QApplication(sys.argv)
    window = Window(weight_path, cfg, cam_width, cam_height)
    sys.exit(app.exec_())
