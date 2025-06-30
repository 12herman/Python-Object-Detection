import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QProgressDialog
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

        self.setRenderHint(QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.NoDrag)

    def wheelEvent(self, event):
        if event.modifiers() & Qt.ControlModifier:
            zoom_in_factor = 2.25
            zoom_out_factor = 0.8
            max_zoom = 4.0

            if event.angleDelta().y() > 0:
                factor = zoom_in_factor
            else:
                factor = zoom_out_factor

            new_zoom = self._zoom * factor

            # Max zoom-in
            if new_zoom > max_zoom:
                return

            # Limit zoom-out
            if new_zoom < 1.0:
                image_rect = self.image_item.pixmap().rect()
                view_rect = self.viewport().rect()
                if image_rect.width() * new_zoom < view_rect.width() * 0.8 or \
                   image_rect.height() * new_zoom < view_rect.height() * 0.8:
                    return

            self._zoom = new_zoom
            self.scale(factor, factor)

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

    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton:
            self._pan = True
            self._start_pan_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
        elif event.button() == Qt.LeftButton:
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
                print(f"🎯 Selected box: {x},{y},{w},{h}")
                self.detect_objects((x, y, w, h))
        super().mouseReleaseEvent(event)

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

        h_temp, w_temp = gray_template.shape
        threshold = 0.8
        all_matches = []

        for scale in [1.0, 0.95, 1.05]:
            resized_template = cv2.resize(gray_template, None, fx=scale, fy=scale)
            r_h, r_w = resized_template.shape
            if r_h >= gray_img.shape[0] or r_w >= gray_img.shape[1]:
                continue

            result = cv2.matchTemplate(gray_img, resized_template, cv2.TM_CCOEFF_NORMED)
            loc = np.where(result >= threshold)
            for pt in zip(*loc[::-1]):
                all_matches.append((pt[0], pt[1], r_w, r_h))

        final_boxes = []
        for new_box in all_matches:
            x1, y1, w1, h1 = new_box
            overlap = False
            for fb in final_boxes:
                x2, y2, w2, h2 = fb
                if abs(x1 - x2) < w1 * 0.5 and abs(y1 - y2) < h1 * 0.5:
                    overlap = True
                    break
            if not overlap:
                final_boxes.append(new_box)

        color = QColor.fromHsv((self._object_id * 60) % 360, 255, 255)
        self._object_colors[self._object_id] = color
        for b in final_boxes:
            self._boxes.append((b, self._object_id))

        print(f"✅ Detected: {len(final_boxes)} objects")
        self._object_id += 1
        progress.close()
        self.update()

    def drawForeground(self, painter, rect):
        painter.setRenderHint(QPainter.Antialiasing)
        if self._drawing and self._start and self._end:
            painter.setPen(QPen(QColor(255, 0, 0), 2, Qt.DashLine))
            painter.drawRect(QRectF(self._start, self._end))

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
        self.open_image()

    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.bmp)")
        if path:
            self.viewer.load_image(path)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())




# # ----------------------------------------------------------------------------
# # v2 detect the object with different color (important)
# import sys
# import cv2
# import numpy as np
# from PyQt5.QtWidgets import (
#     QApplication, QMainWindow, QFileDialog,
#     QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QProgressDialog
# )
# from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QPen
# from PyQt5.QtCore import Qt, QRectF


# class ImageViewer(QGraphicsView):
#     def __init__(self):
#         super().__init__()
#         self.scene = QGraphicsScene()
#         self.setScene(self.scene)
#         self.image_item = QGraphicsPixmapItem()
#         self.scene.addItem(self.image_item)

#         self._image = None
#         self._clone = None
#         self._boxes = []
#         self._drawing = False
#         self._start = None
#         self._end = None
#         self._object_id = 1
#         self._object_colors = {}

#         self._zoom = 1.0
#         self._pan = False
#         self._start_pan_pos = None
#         self.setRenderHint(QPainter.Antialiasing)
#         self.setDragMode(QGraphicsView.NoDrag)
    

#     def wheelEvent(self, event):
#         if event.modifiers() & Qt.ControlModifier:
#             zoom_in_factor = 1.25
#             zoom_out_factor = 0.8
#             max_zoom = 1.0  # Maximum zoom-in limit
            
#             if event.angleDelta().y() > 0:
#                 factor = zoom_in_factor
#             else:
#                 factor = zoom_out_factor

#             new_zoom = self._zoom * factor

#             # ✅ Limit max zoom-in
#             if new_zoom > max_zoom:
#                 return  # Don't zoom in more

#             # ✅ Limit zoom-out to keep image visible
#             if new_zoom < 1.0:
#                 image_rect = self.image_item.pixmap().rect()
#                 view_rect = self.viewport().rect()
#                 if image_rect.width() * new_zoom < view_rect.width() * 0.8 or \
#                 image_rect.height() * new_zoom < view_rect.height() * 0.8:
#                     return  # Don't zoom out more

#             self._zoom = new_zoom
#             self.scale(factor, factor)



#     def load_image(self, path):
#         self._image = cv2.imread(path)
#         self._clone = self._image.copy()
#         h, w, ch = self._image.shape
#         q_img = QImage(self._image.data, w, h, ch * w, QImage.Format_BGR888)
#         pixmap = QPixmap.fromImage(q_img)
#         self.image_item.setPixmap(pixmap)
#         self.setSceneRect(QRectF(pixmap.rect()))
#         self._boxes.clear()
#         self._object_colors.clear()
#         self._object_id = 1
#         self.update()

#     def mousePressEvent(self, event):
#         if event.button() == Qt.LeftButton:
#             self._drawing = True
#             self._start = self.mapToScene(event.pos())
#             self._end = self._start
#         super().mousePressEvent(event)

#     def mouseMoveEvent(self, event):
#         if self._drawing:
#             self._end = self.mapToScene(event.pos())
#             self.viewport().update()
#             self.update()
#         super().mouseMoveEvent(event)

#     def mouseReleaseEvent(self, event):
#         if self._drawing and event.button() == Qt.LeftButton:
#             self._drawing = False
#             x1, y1 = int(self._start.x()), int(self._start.y())
#             x2, y2 = int(self._end.x()), int(self._end.y())
#             x, y = min(x1, x2), min(y1, y2)
#             w, h = abs(x2 - x1), abs(y2 - y1)

#             if w > 10 and h > 10:
#                 print(f"🎯 Selected box: {x},{y},{w},{h}")
#                 self.detect_objects((x, y, w, h))
#         super().mouseReleaseEvent(event)



#     def detect_objects(self, box):
#         progress = QProgressDialog("Detecting similar objects...", None, 0, 0)
#         progress.setWindowTitle("Please wait")
#         progress.setWindowModality(Qt.ApplicationModal)
#         progress.setCancelButton(None)
#         progress.show()
#         QApplication.processEvents()

#         x, y, w, h = box
#         template = self._clone[y:y + h, x:x + w]
#         gray_img = cv2.cvtColor(self._clone, cv2.COLOR_BGR2GRAY)
#         gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

#         h_temp, w_temp = gray_template.shape
#         threshold = 0.8
#         all_matches = []

#         for scale in [1.0, 0.95, 1.05]:
#             resized_template = cv2.resize(gray_template, None, fx=scale, fy=scale)
#             r_h, r_w = resized_template.shape
#             if r_h >= gray_img.shape[0] or r_w >= gray_img.shape[1]:
#                 continue

#             result = cv2.matchTemplate(gray_img, resized_template, cv2.TM_CCOEFF_NORMED)
#             loc = np.where(result >= threshold)
#             for pt in zip(*loc[::-1]):
#                 all_matches.append((pt[0], pt[1], r_w, r_h))

#         final_boxes = []
#         for new_box in all_matches:
#             x1, y1, w1, h1 = new_box
#             overlap = False
#             for fb in final_boxes:
#                 x2, y2, w2, h2 = fb
#                 if abs(x1 - x2) < w1 * 0.5 and abs(y1 - y2) < h1 * 0.5:
#                     overlap = True
#                     break
#             if not overlap:
#                 final_boxes.append(new_box)

#         color = QColor.fromHsv((self._object_id * 60) % 360, 255, 255)
#         self._object_colors[self._object_id] = color
#         for b in final_boxes:
#             self._boxes.append((b, self._object_id))

#         print(f"✅ Detected: {len(final_boxes)} objects")
#         self._object_id += 1
#         progress.close()
#         self.update()

#     def drawForeground(self, painter, rect):
#         painter.setRenderHint(QPainter.Antialiasing)
#         if self._drawing and self._start and self._end:
#             painter.setPen(QPen(QColor(255, 0, 0), 2, Qt.DashLine))
#             painter.drawRect(QRectF(self._start, self._end))

#         for (x, y, w, h), obj_id in self._boxes:
#             color = self._object_colors.get(obj_id, QColor(0, 255, 0))
#             painter.setPen(QPen(color, 2))
#             painter.drawRect(x, y, w, h)


# class MainWindow(QMainWindow):
#     def __init__(self):
#         super().__init__()
#         self.viewer = ImageViewer()
#         self.setCentralWidget(self.viewer)
#         self.setWindowTitle("Smart Object Detection")
#         self.resize(1200, 800)
#         self.open_image()

#     def open_image(self):
#         path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.bmp)")
#         if path:
#             self.viewer.load_image(path)


# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     win = MainWindow()
#     win.show()
#     sys.exit(app.exec_())



# ------------------------------------------------------------------------------------------------
# v1 only detect same color object detection 
# import sys
# import cv2
# import numpy as np
# from PyQt5.QtWidgets import (
#     QApplication, QLabel, QMainWindow, QFileDialog,
#     QGraphicsScene, QGraphicsView, QGraphicsPixmapItem,
#     QProgressDialog
# )
# from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QPen
# from PyQt5.QtCore import Qt, QRectF


# class ImageViewer(QGraphicsView):
#     def __init__(self):
#         super().__init__()
#         self.scene = QGraphicsScene()
#         self.setScene(self.scene)
#         self.image_item = QGraphicsPixmapItem()
#         self.scene.addItem(self.image_item)
#         self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)

#         self._zoom = 1
#         self._pan = False
#         self._start_pos = None
#         self._image = None
#         self._bounding_boxes = []

#         self.drawing = False
#         self.rect_start = None
#         self.rect_end = None

#     def load_image(self, image_path):
#         self._image = cv2.imread(image_path)
#         self._clone = self._image.copy()
#         h, w, ch = self._image.shape
#         bytes_per_line = ch * w
#         q_img = QImage(self._image.data, w, h, bytes_per_line, QImage.Format_BGR888)
#         self.pixmap = QPixmap.fromImage(q_img)
#         self.image_item.setPixmap(self.pixmap)
#         self.setSceneRect(QRectF(self.pixmap.rect()))
#         self._zoom = 1

#     def wheelEvent(self, event):
#         zoom_factor = 1.25 if event.angleDelta().y() > 0 else 0.8
#         self.scale(zoom_factor, zoom_factor)
#         self._zoom *= zoom_factor

#     def mousePressEvent(self, event):
#         if event.button() == Qt.LeftButton:
#             self.drawing = True
#             self.rect_start = self.mapToScene(event.pos())
#             self.rect_end = self.rect_start
#         elif event.button() == Qt.RightButton:
#             self._pan = True
#             self._start_pos = event.pos()
#         super().mousePressEvent(event)

#     def mouseMoveEvent(self, event):
#         if self.drawing:
#             self.rect_end = self.mapToScene(event.pos())
#             self.update()
#         elif self._pan:
#             delta = event.pos() - self._start_pos
#             self._start_pos = event.pos()
#             self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
#             self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
#         super().mouseMoveEvent(event)

#     def mouseReleaseEvent(self, event):
#         if self.drawing and event.button() == Qt.LeftButton:
#             self.drawing = False
#             x1 = int(self.rect_start.x())
#             y1 = int(self.rect_start.y())
#             x2 = int(self.rect_end.x())
#             y2 = int(self.rect_end.y())
#             box = (min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
#             self._bounding_boxes.append(box)
#             print("🎯 Template selected:", box)
#             self.find_template_matches(box)
#             self.update()
#         elif self._pan and event.button() == Qt.RightButton:
#             self._pan = False
#         super().mouseReleaseEvent(event)

#     def drawForeground(self, painter, rect):
#         pen = QPen(QColor(0, 255, 0), 2)
#         painter.setPen(pen)
#         for box in self._bounding_boxes:
#             painter.drawRect(QRectF(*box))
#         if self.drawing and self.rect_start and self.rect_end:
#             painter.setPen(QPen(QColor(255, 0, 0), 2, Qt.DashLine))
#             painter.drawRect(QRectF(self.rect_start, self.rect_end))
        
    
#     def find_template_matches(self, box):
        
#         progress = QProgressDialog("Detecting similar objects...", None, 0, 0)
#         progress.setWindowTitle("Please wait")
#         progress.setWindowModality(Qt.ApplicationModal)
#         progress.setCancelButton(None)
#         progress.show()
#         QApplication.processEvents()

#         x, y, w, h = box
#         template = self._clone[y:y + h, x:x + w]
#         gray_img = cv2.cvtColor(self._clone, cv2.COLOR_BGR2GRAY)
#         gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

#         h_temp, w_temp = gray_template.shape
#         threshold = 0.8
#         all_matches = []

#         # ⚡ Slight scale ranges only (not wide like 0.8 or 1.2)
#         for scale in [1.0, 0.95, 1.05]:
#             resized_template = cv2.resize(gray_template, None, fx=scale, fy=scale)
#             r_h, r_w = resized_template.shape
#             if r_h >= gray_img.shape[0] or r_w >= gray_img.shape[1]:
#                 continue

#             result = cv2.matchTemplate(gray_img, resized_template, cv2.TM_CCOEFF_NORMED)
#             loc = np.where(result >= threshold)
#             for pt in zip(*loc[::-1]):
#                 all_matches.append((pt[0], pt[1], r_w, r_h))

#         # ✅ Remove overlapping matches
#         final_boxes = []
#         for new_box in all_matches:
#             x1, y1, w1, h1 = new_box
#             overlap = False
#             for fb in final_boxes:
#                 x2, y2, w2, h2 = fb
#                 if abs(x1 - x2) < w1 * 0.5 and abs(y1 - y2) < h1 * 0.5:
#                     overlap = True
#                     break
#             if not overlap:
#                 final_boxes.append(new_box)

#         self._bounding_boxes.extend(final_boxes)
#         progress.close()
#         print(f"✅ Found {len(final_boxes)} matches.")
#         self.update()


# class MainWindow(QMainWindow):
#     def __init__(self):
#         super().__init__()
#         self.viewer = ImageViewer()
#         self.setCentralWidget(self.viewer)
#         self.setWindowTitle("Multi-Scale Template Matching Tool")
#         self.resize(1200, 800)
#         self.open_image()

#     def open_image(self):
#         path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.bmp)")
#         if path:
#             self.viewer.load_image(path)


# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     win = MainWindow()
#     win.show()
#     sys.exit(app.exec_())