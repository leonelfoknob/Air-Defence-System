import sys
import cv2
from PyQt5.QtWidgets import (QApplication, QLabel, QWidget, QComboBox, QPushButton, 
                             QVBoxLayout, QHBoxLayout, QSpacerItem, QSizePolicy)
from PyQt5.QtCore import QTimer, Qt, QDateTime
from PyQt5.QtGui import QPixmap, QImage

class MainGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.butonlar = []  # tanımlamayı buraya al
        self.initUI()
        self.initCamera()

    def initUI(self):
        self.setWindowTitle("MakinaFleo-HSS")
        self.setGeometry(100, 100, 1000, 600)
        self.setStyleSheet("background-color: black;")

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # Üst başlık
        header_layout = QHBoxLayout()
        self.title = QLabel("MakinaFleo-HSS")
        self.title.setStyleSheet("color: white; font-size: 18px; font-weight: bold;")
        header_layout.addWidget(self.title, alignment=Qt.AlignLeft)
        header_layout.addItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        main_layout.addLayout(header_layout)

        # Orta alan
        middle_layout = QHBoxLayout()
        main_layout.addLayout(middle_layout, stretch=1)

        # Video paneli
        self.video_frame = QLabel()
        self.video_frame.setStyleSheet("""
            background-color: lightgray;
            border: 2px solid darkgray;
            border-radius: 10px;
        """)
        self.video_frame.setMinimumSize(640, 480)
        #self.video_frame.setAlignment(Qt.AlignCenter)
        middle_layout.addWidget(self.video_frame, stretch=3)

        # Sağ panel
        right_panel = QVBoxLayout()
        right_panel.setSpacing(15)
        right_panel.setAlignment(Qt.AlignTop)
        middle_layout.addLayout(right_panel, stretch=1)

        # Logo
        self.logo = QLabel()
        self.logo.setPixmap(QPixmap("MF.png").scaled(120, 120, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.logo.setAlignment(Qt.AlignCenter)
        right_panel.addWidget(self.logo)

        # Aşama seçimi (ComboBox)
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

        # Butonlar
        self.right_panel = right_panel
        self.arayuzu_guncelle()

        # Saat
        self.time_label = QLabel()
        self.time_label.setStyleSheet("color: blue; font-size: 14px;")
        main_layout.addWidget(self.time_label, alignment=Qt.AlignRight | Qt.AlignBottom)

        # Saat güncelleme
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_time)
        self.timer.start(1000)
        self.update_time()

    def arayuzu_guncelle(self):
        for b in self.butonlar:
            self.right_panel.removeWidget(b)
            b.deleteLater()
        self.butonlar = []

        secilen = self.asama_sec.currentText()
        if secilen == "Aşama 1":
            btn1 = self.create_button("Hedef Takibini Başlat", "#FFA500")
            btn2 = self.create_button("Ateş Et", "#FFA500")
            btn3 = self.create_button("Yeni Hedefe Yönel", "#FFA500")
            self.butonlar.extend([btn1, btn2, btn3])
        elif secilen == "Aşama 2":
            btn = self.create_button("Otonom Modu Başlat", "#FFA500")
            self.butonlar.append(btn)
        elif secilen == "Aşama 3":
            btn1 = self.create_button("Hedef Tanımla", "#FFA500")
            btn2 = self.create_button("Otonom Modu Başlat", "#FFA500")
            self.butonlar.extend([btn1, btn2])

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

    def initCamera(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Kamera açılamadı!")
            return
        self.cam_timer = QTimer(self)
        self.cam_timer.timeout.connect(self.update_frame)
        self.cam_timer.start(30)

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # Görüntü boyutlarını al
            h, w, _ = frame.shape
            # Hedef karesi boyutları (örnek: 80x80 piksel)
            rect_w, rect_h = 80, 80
            # Merkez koordinatları
            cx, cy = w // 2, h // 2

            # Sol üst ve sağ alt köşeler
            top_left = (cx - rect_w // 2, cy - rect_h // 2)
            bottom_right = (cx + rect_w // 2, cy + rect_h // 2)

            # Dikdörtgen çiz
            cv2.rectangle(frame, top_left, bottom_right, (0, 0, 255), 2)#konum, kırmızı, kalınlık
            cv2.putText(frame, "MOD: OTONOM", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            # RGB formatına çevir
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

    def closeEvent(self, event):
        if hasattr(self, 'cap'):
            self.cap.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainGUI()
    window.show()
    sys.exit(app.exec_())