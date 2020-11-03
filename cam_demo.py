import sys
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PyQt5.QtWidgets import QPushButton, QApplication, QLabel, QMainWindow
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt, QTimer
from detect import predict_class_name_and_confidence
from utils.util import load_pytorch_model, load_prices


# camera shape
CAM_WIDTH, CAM_HEIGHT = 848, 480

# window shape
WINDOW_WIDTH, WINDOW_HEIGHT = 1600, 1200


class MyWindow(QMainWindow):
    """
    customized Qt window
    """

    def __init__(self):
        super().__init__()
        self.setGeometry(500, 300, CAM_WIDTH, CAM_HEIGHT + 150)
        self.setFixedSize(CAM_WIDTH, CAM_HEIGHT + 150)
        self.setWindowTitle('Food Recogintion System')

        self.img_label = QLabel(self)
        self.img_label.setGeometry(0, 0, CAM_WIDTH, CAM_HEIGHT)

        self.dish_label = QLabel(self)
        self.dish_label.move(50, CAM_HEIGHT + 25)
        self.dish_label.resize(450, 35)
        self.dish_label.setText("菜品名称：")
        self.dish_label.setFont(QFont("Roman times", 16, QFont.Bold))

        self.price_label = QLabel(self)
        self.price_label.move(50, CAM_HEIGHT + 70)
        self.price_label.resize(450, 35)
        self.price_label.setText("金额：")
        self.price_label.setFont(QFont("Roman times", 16, QFont.Bold))

        self.frame = None
        self.isChecking = False

        check_button = QPushButton("结算", self)
        check_button.move(500, CAM_HEIGHT + 50)
        check_button.resize(130, 40)
        check_button.clicked.connect(self.check)  # Check Button

        confirm_button = QPushButton("确定", self)
        confirm_button.move(650, CAM_HEIGHT + 50)
        confirm_button.resize(130, 40)
        confirm_button.clicked.connect(self.confirm)  # Confirm Button

        self.model = load_pytorch_model('weights/frs_cnn.pth','cfg/frs.cfg')
        self.prices = load_prices('cfg/prices.cfg')

        # camera init
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, CAM_WIDTH)
        self.cap.set(4, CAM_HEIGHT)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self.update_frame)
        self._timer.start(33)  # 30fps

        # self.setCentralWidget(self.img_label)
        self.show()

    def update_frame(self):
        # get camera frame and convert to pixmap to show on img label
        if self.isChecking:
            return

        ret, self.frame = self.cap.read()  # read camera frame
        print('frame shape:', self.frame.shape)
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

        class_name, confidence = predict_class_name_and_confidence(
            frame, self.model)

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)  # ndarray to pil Image

        draw = ImageDraw.Draw(img)
        font_text = ImageFont.truetype("data/simsun.ttc", 26, encoding="utf-8")
        draw.text((5, 5), class_name, (0, 255, 0), font=font_text)
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        print('Class name:', class_name, 'Confidence:', str(confidence)+'%')

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


def main():
    app = QApplication(sys.argv)
    window = MyWindow()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
