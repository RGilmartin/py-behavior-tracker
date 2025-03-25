import sys
import cv2
import numpy as np
from ultralytics import YOLO
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QPushButton, QHBoxLayout, QFileDialog, QLabel,
                             QSlider, QCheckBox)
from PySide6.QtMultimedia import QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtCore import Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QKeyEvent, QImage, QPixmap

class VideoPlayer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Player with Timer")
        self.setGeometry(100, 100, 800, 600)

        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Create video widget and label for processed frames
        self.video_widget = QVideoWidget()
        self.processed_frame_label = QLabel()
        layout.addWidget(self.video_widget)
        layout.addWidget(self.processed_frame_label)

        # Create media player
        self.media_player = QMediaPlayer()
        self.media_player.setVideoOutput(self.video_widget)
        
        # Mouse tracking variables
        self.tracking_enabled = False
        self.model = YOLO('yolov8n.pt')  # Load YOLOv8 model
        self.cap = None
        self.mouse_positions = []
        self.frame_counter = 0  # Add frame counter

        # Create controls
        controls_layout = QHBoxLayout()

        # Play/Pause button
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.play_pause)
        controls_layout.addWidget(self.play_button)

        # Seek slider
        self.position_slider = QSlider(Qt.Horizontal)
        self.position_slider.setRange(0, 0)
        self.position_slider.sliderMoved.connect(self.set_position)
        controls_layout.addWidget(self.position_slider)

        # Timer controls and display
        self.timer_button = QPushButton("Start Timer (1)")
        self.timer_button.clicked.connect(self.toggle_timer)
        controls_layout.addWidget(self.timer_button)

        # Mouse tracking toggle
        self.tracking_checkbox = QCheckBox("Enable Mouse Tracking")
        self.tracking_checkbox.stateChanged.connect(self.toggle_tracking)
        controls_layout.addWidget(self.tracking_checkbox)

        self.reset_button = QPushButton("Reset Timer")
        self.reset_button.clicked.connect(self.reset_timer)
        controls_layout.addWidget(self.reset_button)

        self.timer_label = QLabel("Timer: 0:00.000")
        controls_layout.addWidget(self.timer_label)

        # Add controls to main layout
        layout.addLayout(controls_layout)

        # Open file button
        self.open_button = QPushButton("Open Video")
        self.open_button.clicked.connect(self.open_file)
        layout.addWidget(self.open_button)

        # Timer setup
        self.timer = QTimer()
        self.timer.setInterval(1)  # Update every 1ms for precise timing
        self.timer.timeout.connect(self.update_timer)
        self.timer_active = False
        self.timer_value = 0.0

        # Media player signals
        self.media_player.durationChanged.connect(self.duration_changed)
        self.media_player.positionChanged.connect(self.position_changed)

    def open_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Video File", "",
                                                 "Video Files (*.mp4 *.avi *.mkv)")
        if file_name:
            self.media_player.setSource(QUrl.fromLocalFile(file_name))
            self.play_button.setEnabled(True)
            # Initialize video capture for tracking
            if self.cap is not None:
                self.cap.release()
            self.cap = cv2.VideoCapture(file_name)
            self.mouse_positions = []

    def play_pause(self):
        if self.media_player.playbackState() == QMediaPlayer.PlayingState:
            self.media_player.pause()
            self.play_button.setText("Play")
        else:
            self.media_player.play()
            self.play_button.setText("Pause")
            if self.tracking_enabled and self.cap is not None:
                QTimer.singleShot(50, self.process_frame)

    def duration_changed(self, duration):
        self.position_slider.setRange(0, duration)

    def position_changed(self, position):
        self.position_slider.setValue(position)

    def set_position(self, position):
        self.media_player.setPosition(position)

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key_1:
            self.toggle_timer()
        super().keyPressEvent(event)

    def toggle_timer(self):
        if not self.timer_active:
            self.timer_active = True
            self.timer.start()
            self.timer_button.setText("Stop Timer (1)")
        else:
            self.timer_active = False
            self.timer.stop()
            self.timer_button.setText("Start Timer (1)")

    def reset_timer(self):
        self.timer_active = False
        self.timer.stop()
        self.timer_value = 0.0
        self.timer_button.setText("Start Timer (1)")
        self.timer_label.setText("Timer: 0:00.000")

    def update_timer(self):
        if self.timer_active:
            self.timer_value += 0.001  # Add 1ms
            minutes = int(self.timer_value // 60)
            seconds = int(self.timer_value % 60)
            milliseconds = int((self.timer_value % 1) * 1000)
            self.timer_label.setText(f"Timer: {minutes}:{seconds:02d}.{milliseconds:03d}")

    def toggle_tracking(self, state):
        self.tracking_enabled = bool(state)
        if self.tracking_enabled and self.cap is not None and self.media_player.playbackState() == QMediaPlayer.PlayingState:
            self.process_frame()

    def process_frame(self):
        if not self.tracking_enabled or self.cap is None or self.media_player.playbackState() != QMediaPlayer.PlayingState:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return

        self.frame_counter += 1
        if self.frame_counter % 5 != 0:  # Only process every 5th frame
            QTimer.singleShot(50, self.process_frame)
            return

        # Run YOLOv8 inference on the frame
        results = self.model(frame, conf=0.25)  # Confidence threshold of 0.25
        
        # Process YOLOv8 detections
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get box coordinates and confidence
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf)
                cls = int(box.cls)
                
                # Draw bounding box and confidence
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                center = (int((x1 + x2) // 2), int((y1 + y2) // 2))
                self.mouse_positions.append(center)
                
                # Draw detection results
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, center, 5, (0, 255, 0), -1)
                cv2.putText(frame, f'Mouse: {conf:.2f}', (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Convert frame to QImage and display
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        self.processed_frame_label.setPixmap(QPixmap.fromImage(q_image).scaled(
            self.processed_frame_label.size(), Qt.KeepAspectRatio))

        # Schedule next frame processing
        if self.tracking_enabled:
            QTimer.singleShot(50, self.process_frame)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    player = VideoPlayer()
    player.show()
    sys.exit(app.exec())