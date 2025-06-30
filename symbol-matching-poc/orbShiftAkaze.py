import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QGraphicsView, QGraphicsScene,
    QGraphicsPixmapItem, QProgressDialog, QToolBar, QAction, QSlider, QLabel,
    QMessageBox
)
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QPen
from PyQt5.QtCore import Qt, QRectF


class ImageViewer(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        self.image_item = QGraphicsPixmapItem()
        self.scene.addItem(self.image_item)

        self._image = None
        self._clone = None
        self._boxes = []
        self._drawing = False
        self._start = None
        self._end = None
        self._object_id = 1
        self._object_colors = {}

        self._zoom = 1.0
        self._pan = False
        self._start_pan_pos = None

        self._enable_drawing = False
        self._threshold = 0.8
        self._preview_box = None

        self.setRenderHint(QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.NoDrag)

    def set_threshold(self, value):
        self._threshold = value

    def toggle_drawing_mode(self):
        self._enable_drawing = not self._enable_drawing
        cursor = Qt.CrossCursor if self._enable_drawing else Qt.ArrowCursor
        self.setCursor(cursor)

    def load_image(self, path):
        self._image = cv2.imread(path)
        self._clone = self._image.copy()
        h, w, ch = self._image.shape
        q_img = QImage(self._image.data, w, h, ch * w, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(q_img)
        self.image_item.setPixmap(pixmap)
        self.setSceneRect(QRectF(pixmap.rect()))
        self._boxes.clear()
        self._object_colors.clear()
        self._object_id = 1
        self._zoom = 1.0
        self.resetTransform()
        self.update()

    def wheelEvent(self, event):
        if event.modifiers() & Qt.ControlModifier:
            zoom_in_factor = 2.25
            zoom_out_factor = 0.8
            max_zoom = 4.0
            min_zoom = 0.1

            cursor_pos = event.pos()
            scene_pos_before = self.mapToScene(cursor_pos)

            factor = zoom_in_factor if event.angleDelta().y() > 0 else zoom_out_factor
            new_zoom = self._zoom * factor

            if new_zoom < min_zoom or new_zoom > max_zoom:
                return

            self._zoom = new_zoom
            self.scale(factor, factor)

            scene_pos_after = self.mapToScene(cursor_pos)
            delta = scene_pos_after - scene_pos_before
            self.translate(delta.x(), delta.y())
        else:
            scroll_amount = 20
            if event.modifiers() & Qt.ShiftModifier:
                if event.angleDelta().y() > 0:
                    self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - scroll_amount)
                else:
                    self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() + scroll_amount)
            else:
                if event.angleDelta().y() > 0:
                    self.verticalScrollBar().setValue(self.verticalScrollBar().value() - scroll_amount)
                else:
                    self.verticalScrollBar().setValue(self.verticalScrollBar().value() + scroll_amount)

    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton:
            self._pan = True
            self._start_pan_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
        elif event.button() == Qt.LeftButton and self._enable_drawing:
            self._drawing = True
            self._start = self.mapToScene(event.pos())
            self._end = self._start
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._pan and self._start_pan_pos:
            delta = event.pos() - self._start_pan_pos
            self._start_pan_pos = event.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            return

        if self._drawing:
            self._end = self.mapToScene(event.pos())
            self.viewport().update()
            self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.RightButton:
            self._pan = False
            self.setCursor(Qt.ArrowCursor)

        if self._drawing and event.button() == Qt.LeftButton:
            self._drawing = False
            x1, y1 = int(self._start.x()), int(self._start.y())
            x2, y2 = int(self._end.x()), int(self._end.y())
            x, y = min(x1, x2), min(y1, y2)
            w, h = abs(x2 - x1), abs(y2 - y1)

            if w > 10 and h > 10:
                self._preview_box = (x, y, w, h)
                self.viewport().update()
                reply = QMessageBox.question(self, "Confirm", "Is this the correct mark?", QMessageBox.Ok | QMessageBox.Cancel)
                if reply == QMessageBox.Ok:
                    self.detect_objects(self._preview_box)
                self._preview_box = None
                self._enable_drawing = False
                self.setCursor(Qt.ArrowCursor)
        super().mouseReleaseEvent(event)

    # ORB - Method
    # def detect_objects(self, box):
    #     progress = QProgressDialog("Detecting similar objects...", None, 0, 0)
    #     progress.setWindowTitle("Please wait")
    #     progress.setWindowModality(Qt.ApplicationModal)
    #     progress.setCancelButton(None)
    #     progress.show()
    #     QApplication.processEvents()

    #     x, y, w, h = box
    #     template = self._clone[y:y + h, x:x + w]
    #     gray_img = cv2.cvtColor(self._clone, cv2.COLOR_BGR2GRAY)
    #     gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    #     # Initialize ORB detector
    #     orb = cv2.ORB_create(1000)
    #     kp1, des1 = orb.detectAndCompute(gray_template, None)
    #     kp2, des2 = orb.detectAndCompute(gray_img, None)

    #     if des1 is None or des2 is None:
    #         QMessageBox.warning(self, "Detection Failed", "Not enough features found.")
    #         progress.close()
    #         return

    #     bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    #     matches = bf.match(des1, des2)
    #     matches = sorted(matches, key=lambda x: x.distance)

    #     points = []
    #     for m in matches[:30]:  # Use top 30 matches
    #         pt = kp2[m.trainIdx].pt
    #         points.append(pt)

    #     if not points:
    #         QMessageBox.information(self, "No Objects", "No similar objects found.")
    #         progress.close()
    #         return

    #     # Cluster points (very simple)
    #     clustered = []
    #     for pt in points:
    #         found = False
    #         for cluster in clustered:
    #             if abs(cluster[0] - pt[0]) < w // 2 and abs(cluster[1] - pt[1]) < h // 2:
    #                 found = True
    #                 break
    #         if not found:
    #             clustered.append(pt)

    #     # Assign color
    #     used_hues = {c.hue() for c in self._object_colors.values()}
    #     hue = (self._object_id * 47) % 360
    #     for i in range(360):
    #         if hue not in used_hues:
    #             break
    #         hue = (hue + 30) % 360
    #     color = QColor.fromHsv(hue, 255, 255)
    #     self._object_colors[self._object_id] = color

    #     # Save results
    #     for cx, cy in clustered:
    #         self._boxes.append(((int(cx - w / 2), int(cy - h / 2), w, h), self._object_id))

    #     print(f"✅ Detected: {len(clustered)} objects")
    #     self._object_id += 1
    #     progress.close()
    #     self.update()

    def detect_objects(self, box):
        x, y, w, h = box
        template = self._clone[y:y + h, x:x + w]
        gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        gray_scene = cv2.cvtColor(self._clone, cv2.COLOR_BGR2GRAY)

        # ORB detector
        orb = cv2.ORB_create(5000)
        kp1, des1 = orb.detectAndCompute(gray_template, None)
        kp2, des2 = orb.detectAndCompute(gray_scene, None)

        if des1 is None or des2 is None:
            QMessageBox.warning(self, "Warning", "Not enough features detected.")
            return

        # Match descriptors using BFMatcher
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        if len(matches) < 10:
            QMessageBox.warning(self, "Warning", "Not enough good matches.")
            return

        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Find homography to detect rotated/scaled positions
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if M is not None:
            h, w = gray_template.shape
            pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)
            x, y, w, h = cv2.boundingRect(dst)

            # Assign color and object ID
            used_hues = {c.hue() for c in self._object_colors.values()}
            for i in range(360):
                hue = (self._object_id * 47 + i * 29) % 360
                if hue not in used_hues:
                    break
            color = QColor.fromHsv(hue, 255, 255)
            self._object_colors[self._object_id] = color

            self._boxes.append(((x, y, w, h), self._object_id))
            print(f"✅ Detected object with rotation/scale at: {x}, {y}")
            self._object_id += 1
            self.update()
        else:
            QMessageBox.warning(self, "Warning", "Failed to compute homography.")







    def drawForeground(self, painter, rect):
        painter.setRenderHint(QPainter.Antialiasing)
        if self._drawing and self._start and self._end:
            painter.setPen(QPen(QColor(255, 0, 0), 2, Qt.DashLine))
            painter.drawRect(QRectF(self._start, self._end))

        if self._preview_box:
            x, y, w, h = self._preview_box
            painter.setPen(QPen(QColor(255, 0, 0), 2, Qt.DashDotLine))
            painter.drawRect(x, y, w, h)

        for (x, y, w, h), obj_id in self._boxes:
            color = self._object_colors.get(obj_id, QColor(0, 255, 0))
            painter.setPen(QPen(color, 2))
            painter.drawRect(x, y, w, h)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.viewer = ImageViewer()
        self.setCentralWidget(self.viewer)
        self.setWindowTitle("Smart Object Detection")
        self.resize(1200, 800)

        self.toolbar = QToolBar("Toolbar")
        self.addToolBar(self.toolbar)

        self.marker_action = QAction("Marker", self)
        self.marker_action.setCheckable(True)
        self.marker_action.toggled.connect(self.toggle_marker_mode)
        self.toolbar.addAction(self.marker_action)

        self.slider_label = QLabel("Detection Threshold: 80%")
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(50)
        self.threshold_slider.setMaximum(100)
        self.threshold_slider.setValue(80)
        self.threshold_slider.valueChanged.connect(self.update_threshold)

        self.toolbar.addWidget(self.slider_label)
        self.toolbar.addWidget(self.threshold_slider)

        self.open_image()

    def toggle_marker_mode(self, checked):
        self.viewer.toggle_drawing_mode()

    def update_threshold(self, value):
        self.slider_label.setText(f"Detection Threshold: {value}%")
        self.viewer.set_threshold(value / 100.0)

    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.bmp)")
        if path:
            self.viewer.load_image(path)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())