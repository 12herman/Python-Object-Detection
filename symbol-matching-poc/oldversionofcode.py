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
from sklearn.cluster import DBSCAN 

class ImageViewer(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.scene = QGraphicsScene()
        self.image_item = QGraphicsPixmapItem()
        
        self.setScene(self.scene)
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
        self.resetTransform()
        # self.fitInView(self.image_item, Qt.KeepAspectRatio) # add new one
        self._zoom = 1.0
        # self._base_transform = self.transform() 
        self.update()

    def wheelEvent(self, event):
        if event.modifiers() & Qt.ControlModifier:
            zoom_in_factor = 2.0
            zoom_out_factor = 0.8
            max_zoom = 3.0
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
            scroll_amount = 150
            if event.modifiers() & Qt.ShiftModifier:
                self.horizontalScrollBar().setValue(
                    self.horizontalScrollBar().value() - scroll_amount if event.angleDelta().y() > 0 else self.horizontalScrollBar().value() + scroll_amount
                )
            else:
                self.verticalScrollBar().setValue(
                    self.verticalScrollBar().value() - scroll_amount if event.angleDelta().y() > 0 else self.verticalScrollBar().value() + scroll_amount
                )

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

    # shift mathching well
    def detect_objects(self, box):
        progress = QProgressDialog("Detecting similar objects...", None, 0, 0)
        progress.setWindowTitle("Please wait")
        progress.setWindowModality(Qt.ApplicationModal)
        progress.setCancelButton(None)
        progress.show()
        QApplication.processEvents()

        x, y, w, h = box
        template = self._clone[y:y + h, x:x + w]
        gray_img = cv2.cvtColor(self._clone, cv2.COLOR_BGR2GRAY)
        gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(gray_template, None)
        kp2, des2 = sift.detectAndCompute(gray_img, None)

        print(f"Template keypoints: {len(kp1)} | Image keypoints: {len(kp2)}")

        if des1 is not None and des2 is not None and len(kp1) >= 4 and len(kp2) >= 10:
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des1, des2, k=2)

            good_matches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

            print(f"Good matches found: {len(good_matches)}")

            if len(good_matches) >= 4:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                if M is not None:
                    h_t, w_t = gray_template.shape
                    pts = np.float32([[0, 0], [w_t, 0], [w_t, h_t], [0, h_t]]).reshape(-1, 1, 2)
                    dst = cv2.perspectiveTransform(pts, M)

                    xs = [p[0][0] for p in dst]
                    ys = [p[0][1] for p in dst]
                    x_box, y_box, w_box, h_box = int(min(xs)), int(min(ys)), int(max(xs) - min(xs)), int(max(ys) - min(ys))

                    hue = (self._object_id * 47) % 360
                    color = QColor.fromHsv(hue, 255, 255)
                    self._object_colors[self._object_id] = color
                    self._boxes.append(((x_box, y_box, w_box, h_box), self._object_id))
                    print(f"✅ Object detected with SIFT + Homography")
                    self._object_id += 1
                    progress.close()
                    self.update()
                    return

        # If not enough keypoints or matches, fallback to template matching
        print("⚠️ Not enough good matches. Falling back to template matching...")
        result = cv2.matchTemplate(gray_img, gray_template, cv2.TM_CCOEFF_NORMED)
        loc = np.where(result >= self._threshold)
        matches = list(zip(*loc[::-1]))

        final_boxes = []
        for pt in matches:
            final_boxes.append((pt[0], pt[1], w, h))

        unique_boxes = []
        for new_box in final_boxes:
            x1, y1, w1, h1 = new_box
            overlap = False
            for ub in unique_boxes:
                x2, y2, w2, h2 = ub
                if abs(x1 - x2) < w1 * 0.5 and abs(y1 - y2) < h1 * 0.5:
                    overlap = True
                    break
            if not overlap:
                unique_boxes.append(new_box)

        hue = (self._object_id * 47) % 360
        color = QColor.fromHsv(hue, 255, 255)
        self._object_colors[self._object_id] = color
        for b in unique_boxes:
            self._boxes.append((b, self._object_id))

        print(f"✅ Fallback detected {len(unique_boxes)} object(s)")
        self._object_id += 1
        progress.close()
        self.update()


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