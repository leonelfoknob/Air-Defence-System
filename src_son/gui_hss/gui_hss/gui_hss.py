import sys
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from PyQt5.QtWidgets import (QApplication, QLabel, QWidget, QComboBox, QPushButton, 
                             QVBoxLayout, QHBoxLayout, QSpacerItem, QSizePolicy)
from PyQt5.QtCore import QTimer, Qt, QDateTime
from PyQt5.QtGui import QPixmap, QImage

class MainGUI(QWidget):
    def __init__(self, node):
        super().__init__()
        self.node = node
        self.bridge = CvBridge()
        self.cv_image = None

        self.butonlar = []
        self.initUI()
        self.initROS()

    def initUI(self):
        self.setWindowTitle("MakinaFleo-HSS")
        self.setGeometry(100, 100, 1000, 600)
        self.setStyleSheet("background-color: black;")

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)

        header_layout = QHBoxLayout()
        self.title = QLabel("MakinaFleo-HSS")
        self.title.setStyleSheet("color: white; font-size: 18px; font-weight: bold;")
        header_layout.addWidget(self.title, alignment=Qt.AlignLeft)
        header_layout.addItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        main_layout.addLayout(header_layout)

        middle_layout = QHBoxLayout()
        main_layout.addLayout(middle_layout, stretch=1)

        self.video_frame = QLabel()
        self.video_frame.setStyleSheet("""
            background-color: lightgray;
            border: 2px solid darkgray;
            border-radius: 10px;
        """)
        self.video_frame.setMinimumSize(640, 480)
        middle_layout.addWidget(self.video_frame, stretch=3)

        right_panel = QVBoxLayout()
        right_panel.setSpacing(15)
        right_panel.setAlignment(Qt.AlignTop)
        middle_layout.addLayout(right_panel, stretch=1)

        self.logo = QLabel()
        self.logo.setPixmap(QPixmap("MF.png").scaled(120, 120, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.logo.setAlignment(Qt.AlignCenter)
        right_panel.addWidget(self.logo)

        self.asama_sec = QComboBox()
        self.asama_sec.addItems(["Aşama 1", "Aşama 2", "Aşama 3"])
        self.asama_sec.setStyleSheet("""
            QComboBox {
                color: white;
                background-color: #444;
                font-size: 14px;
                padding: 6px;
                border-radius: 5px;
            }
            QComboBox QAbstractItemView {
                background-color: #444;
                color: white;
                selection-background-color: #666;
            }
        """)
        self.asama_sec.currentIndexChanged.connect(self.arayuzu_guncelle)
        right_panel.addWidget(self.asama_sec)

        self.right_panel = right_panel
        self.arayuzu_guncelle()

        self.time_label = QLabel()
        self.time_label.setStyleSheet("color: blue; font-size: 14px;")
        main_layout.addWidget(self.time_label, alignment=Qt.AlignRight | Qt.AlignBottom)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_time)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(1000 // 30)
        self.update_time()

    def arayuzu_guncelle(self):
        for b in self.butonlar:
            self.right_panel.removeWidget(b)
            b.deleteLater()
        self.butonlar = []

        secilen = self.asama_sec.currentText()
        if secilen == "Aşama 1":
            self.butonlar.extend([
                self.create_button("Hedef Takibini Başlat", "#FFA500"),
                self.create_button("Ateş Et", "#FFA500"),
                self.create_button("Yeni Hedefe Yönel", "#FFA500")
            ])
        elif secilen == "Aşama 2":
            self.butonlar.append(self.create_button("Otonom Modu Başlat", "#FFA500"))
        elif secilen == "Aşama 3":
            self.butonlar.extend([
                self.create_button("Hedef Tanımla", "#FFA500"),
                self.create_button("Otonom Modu Başlat", "#FFA500")
            ])

        for b in self.butonlar:
            self.right_panel.addWidget(b)

    def create_button(self, text, color):
        btn = QPushButton(text)
        btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {color};
                font-size: 14px;
                min-width: 140px;
                min-height: 36px;
                border-radius: 10px;
                border: 1px solid gray;
                color: black;
            }}
            QPushButton:hover {{
                background-color: {color};
                opacity: 0.85;
            }}
        """)
        return btn

    def update_time(self):
        current_time = QDateTime.currentDateTime().toString("dd/MM/yyyy  HH:mm:ss")
        self.time_label.setText(current_time)

    def initROS(self):
        self.subscription = self.node.create_subscription(
            Image,
            '/camera2/image_raw',
            self.image_callback,
            10
        )

    def image_callback(self, msg):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            if msg.encoding != 'bgr8':
                if len(self.cv_image.shape) == 2:
                    self.cv_image = cv2.cvtColor(self.cv_image, cv2.COLOR_GRAY2BGR)
                else:
                    self.cv_image = cv2.cvtColor(self.cv_image, cv2.COLOR_RGB2BGR)
        except Exception as e:
            self.node.get_logger().error(f"CV Bridge Error: {e}")

    def update_frame(self):
        if self.cv_image is None:
            return

        frame = self.cv_image.copy()

        h, w, _ = frame.shape
        cx, cy = w // 2, h // 2
        rect_w, rect_h = 80, 80
        top_left = (cx - rect_w // 2, cy - rect_h // 2)
        bottom_right = (cx + rect_w // 2, cy + rect_h // 2)
        cv2.rectangle(frame, top_left, bottom_right, (0, 0, 255), 2)
        cv2.putText(frame, "MOD: OTONOM", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        h, w, ch = frame.shape
        bytes_per_line = ch * w
        img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pix = img.scaled(self.video_frame.width(), self.video_frame.height(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
        self.video_frame.setPixmap(QPixmap.fromImage(pix))

    def resizeEvent(self, event):
        new_size = min(self.width()//8, self.height()//5)
        self.logo.setPixmap(QPixmap("MF.png").scaled(
            new_size, new_size, 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        ))
        super().resizeEvent(event)


def main():
    rclpy.init()
    node = rclpy.create_node('gui_hss')
    app = QApplication(sys.argv)
    gui = MainGUI(node)
    gui.show()

    timer = QTimer()
    timer.timeout.connect(lambda: rclpy.spin_once(node, timeout_sec=0))
    timer.start(10)

    app.exec_()
    node.destroy_node()
    rclpy.shutdown()
