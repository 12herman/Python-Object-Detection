import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QFileDialog, QMainWindow
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen
from PyQt5.QtCore import Qt, QRectF, QPointF

class ImageMatcher(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        self.image_item = QGraphicsPixmapItem()
        self.scene.addItem(self.image_item)

        self._image = None
        self._start = None
        self._end = None
        self._drawing = False

    def load_image(self, path):
        self._image = cv2.imread(path)
        h, w, ch = self._image.shape
        qimg = QImage(self._image.data, w, h, ch * w, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(qimg)
        self.image_item.setPixmap(pixmap)
        self.setSceneRect(QRectF(pixmap.rect()))

    def mousePressEvent(self, event):
        if self._image is None:
            return
        self._start = self.mapToScene(event.pos()).toPoint()
        self._drawing = True

    def mouseMoveEvent(self, event):
        if self._drawing:
            self._end = self.mapToScene(event.pos()).toPoint()
            self.update()

    def mouseReleaseEvent(self, event):
        if self._drawing:
            self._end = self.mapToScene(event.pos()).toPoint()
            self._drawing = False
            self.match_template()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._drawing and self._start and self._end:
            painter = QPainter(self.viewport())
            painter.setPen(QPen(Qt.red, 2, Qt.SolidLine))
            rect = QRectF(self._start, self._end)
            painter.drawRect(rect)

    def match_template(self):
        x1, y1 = self._start.x(), self._start.y()
        x2, y2 = self._end.x(), self._end.y()
        x1, x2 = sorted([int(x1), int(x2)])
        y1, y2 = sorted([int(y1), int(y2)])
        needle = self._image[y1:y2, x1:x2]

        result = cv2.matchTemplate(self._image, needle, cv2.TM_CCOEFF_NORMED)
        threshold = 0.8
        loc = np.where(result >= threshold)

        # Draw rectangles for all matches
        for pt in zip(*loc[::-1]):
            cv2.rectangle(self._image, pt, (pt[0] + needle.shape[1], pt[1] + needle.shape[0]), (0, 255, 0), 2)

        # Refresh display
        h, w, ch = self._image.shape
        qimg = QImage(self._image.data, w, h, ch * w, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(qimg)
        self.image_item.setPixmap(pixmap)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.viewer = ImageMatcher()
        self.setCentralWidget(self.viewer)
        self.setWindowTitle("Object Detection using Manual Region")
        self.resize(800, 600)
        self.load_image()

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg)")
        if path:
            self.viewer.load_image(path)

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
