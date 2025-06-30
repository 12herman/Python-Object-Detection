import cv2
import numpy as np
from tkinter import filedialog, Tk

# === GUI File Select ===
root = Tk()
root.withdraw()

# Select main image
main_img_path = filedialog.askopenfilename(title="Select Main Image")
if not main_img_path:
    print("No main image selected.")
    exit()

# Select template images
template_paths = filedialog.askopenfilenames(title="Select Template Images")
if not template_paths:
    print("No template images selected.")
    exit()

# === Load Main Image and ORB Init ===
main_img = cv2.imread(main_img_path, 0)
orb = cv2.ORB_create(nfeatures=1000)
kp_main, des_main = orb.detectAndCompute(main_img, None)

# === Match Each Template ===
for template_path in template_paths:
    template = cv2.imread(template_path, 0)
    if template is None:
        print(f"Error loading: {template_path}")
        continue

    kp_temp, des_temp = orb.detectAndCompute(template, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des_temp, des_main)
    matches = sorted(matches, key=lambda x: x.distance)

    matched_img = cv2.drawMatches(template, kp_temp, main_img, kp_main, matches[:20], None, flags=2)
    matched_img = cv2.resize(matched_img, (1000, 500))

    cv2.imshow(f"Matches for {template_path.split('/')[-1]}", matched_img)

cv2.waitKey(0)
cv2.destroyAllWindows()




# import sys
# import os
# import cv2
# from PyQt5.QtWidgets import (
#     QApplication, QMainWindow, QFileDialog,
#     QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
#     QPushButton, QVBoxLayout, QWidget, QHBoxLayout
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
#         self._boxes = []
#         self._start = None
#         self._end = None
#         self._zoom = 1.0
#         self.crop_mode = False
#         self.image_marked = False  # Prevent re-marking same image

#         self.setRenderHint(QPainter.Antialiasing)
#         self.setDragMode(QGraphicsView.NoDrag)

#     def load_image(self, path):
#         self._image = cv2.imread(path)
#         if self._image is None:
#             return
#         h, w, ch = self._image.shape
#         bytes_per_line = ch * w
#         q_img = QImage(self._image.data, w, h, bytes_per_line, QImage.Format_BGR888)
#         pixmap = QPixmap.fromImage(q_img)
#         self.image_item.setPixmap(pixmap)
#         self.setSceneRect(QRectF(pixmap.rect()))
#         self.resetTransform()
#         self._zoom = 1.0
#         self._boxes = []
#         self._start = None
#         self._end = None
#         self.image_marked = False
#         self.update()

#     def wheelEvent(self, event):
#         if event.modifiers() & Qt.ControlModifier:
#             self.verticalScrollBar().setValue(self.verticalScrollBar().value() - event.angleDelta().y())
#         elif event.modifiers() & Qt.ShiftModifier:
#             self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - event.angleDelta().y())
#         else:
#             factor = 1.25 if event.angleDelta().y() > 0 else 0.8
#             self._zoom *= factor
#             self.scale(factor, factor)

#     def mousePressEvent(self, event):
#         if self.crop_mode and event.button() == Qt.LeftButton:
#             self._start = self.mapToScene(event.pos())
#             self._end = self._start
#         super().mousePressEvent(event)

#     def mouseMoveEvent(self, event):
#         if self.crop_mode and self._start:
#             self._end = self.mapToScene(event.pos())
#             self.viewport().update()
#         super().mouseMoveEvent(event)

#     def mouseReleaseEvent(self, event):
#         if self.crop_mode and event.button() == Qt.LeftButton and self._start and self._end:
#             rect = QRectF(self._start, self._end).normalized()
#             if rect.width() > 5 and rect.height() > 5:
#                 self._boxes.append(rect)
#                 self.image_marked = True
#             self._start = None
#             self._end = None
#             self.viewport().update()
#         super().mouseReleaseEvent(event)

#     def drawForeground(self, painter, rect):
#         painter.setRenderHint(QPainter.Antialiasing)
#         if self._start and self._end:
#             live_rect = QRectF(self._start, self._end).normalized()
#             painter.setPen(QPen(QColor(255, 0, 0), 2, Qt.DashLine))
#             painter.drawRect(live_rect)
#         for box in self._boxes:
#             painter.setPen(QPen(QColor(255, 0, 0), 2, Qt.SolidLine))
#             painter.drawRect(box)

#     def crop_and_save_all(self):
#         if self._image is None or not self._boxes:
#             return
#         base_path = QFileDialog.getExistingDirectory(None, "Select Folder to Save Crops")
#         if not base_path:
#             return
#         for i, rect in enumerate(self._boxes):
#             x = int(rect.x())
#             y = int(rect.y())
#             w = int(rect.width())
#             h = int(rect.height())
#             cropped = self._image[y:y + h, x:x + w]
#             if cropped.size > 0:
#                 filename = os.path.join(base_path, f"crop_{i+1}.png")
#                 cv2.imwrite(filename, cropped)

#     def clear_crop_markers(self):
#         self._boxes = []
#         self.image_marked = False
#         self.viewport().update()

#     def toggle_crop_mode(self):
#         self.crop_mode = not self.crop_mode
#         self.setCursor(Qt.CrossCursor if self.crop_mode else Qt.ArrowCursor)

#     def keyPressEvent(self, event):
#         if event.key() == Qt.Key_Escape:
#             self.crop_mode = False
#             self.setCursor(Qt.ArrowCursor)
#         super().keyPressEvent(event)


# class MainWindow(QMainWindow):
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("Image Crop Tool")
#         self.resize(1200, 800)

#         self.viewer = ImageViewer()

#         self.upload_btn = QPushButton("Upload Image")
#         self.upload_btn.clicked.connect(self.upload_image)

#         self.marker_btn = QPushButton("Toggle Marker")
#         self.marker_btn.clicked.connect(self.viewer.toggle_crop_mode)

#         self.crop_btn = QPushButton("Save All Crops")
#         self.crop_btn.clicked.connect(self.viewer.crop_and_save_all)

#         self.clear_btn = QPushButton("Clear Markers")
#         self.clear_btn.clicked.connect(self.viewer.clear_crop_markers)

#         btn_layout = QHBoxLayout()
#         btn_layout.addWidget(self.upload_btn)
#         btn_layout.addWidget(self.marker_btn)
#         btn_layout.addWidget(self.crop_btn)
#         btn_layout.addWidget(self.clear_btn)

#         layout = QVBoxLayout()
#         layout.addLayout(btn_layout)
#         layout.addWidget(self.viewer)

#         container = QWidget()
#         container.setLayout(layout)
#         self.setCentralWidget(container)

#     def upload_image(self):
#         path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.bmp)")
#         if path:
#             self.viewer.load_image(path)


# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     window = MainWindow()
#     window.show()
#     sys.exit(app.exec_())







# # tuned version
# import sys
# import os
# import cv2
# from PyQt5.QtWidgets import (
#     QApplication, QMainWindow, QFileDialog,
#     QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
#     QPushButton, QVBoxLayout, QWidget, QHBoxLayout
# )
# from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QPen, QCursor
# from PyQt5.QtCore import Qt, QRectF


# class ImageViewer(QGraphicsView):
#     def __init__(self):
#         super().__init__()
#         self.scene = QGraphicsScene()
#         self.setScene(self.scene)
#         self.image_item = QGraphicsPixmapItem()
#         self.scene.addItem(self.image_item)

#         self._image = None
#         self._crop_rect = None
#         self._start = None
#         self._end = None
#         self._zoom = 1.0

#         self.crop_mode = False  # 👈 Only allow cropping if this is True

#         self.setRenderHint(QPainter.Antialiasing)
#         self.setDragMode(QGraphicsView.NoDrag)

#     def load_image(self, path):
#         self._image = cv2.imread(path)
#         if self._image is None:
#             return
#         h, w, ch = self._image.shape
#         bytes_per_line = ch * w
#         q_img = QImage(self._image.data, w, h, bytes_per_line, QImage.Format_BGR888)
#         pixmap = QPixmap.fromImage(q_img)
#         self.image_item.setPixmap(pixmap)
#         self.setSceneRect(QRectF(pixmap.rect()))
#         self.resetTransform()
#         self._zoom = 1.0
#         self._crop_rect = None
#         self._start = None
#         self._end = None

#     def wheelEvent(self, event):
#         if event.modifiers() & Qt.ControlModifier:
#             self.verticalScrollBar().setValue(self.verticalScrollBar().value() - event.angleDelta().y())
#         elif event.modifiers() & Qt.ShiftModifier:
#             self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - event.angleDelta().y())
#         else:
#             factor = 1.25 if event.angleDelta().y() > 0 else 0.8
#             self._zoom *= factor
#             self.scale(factor, factor)

#     def mousePressEvent(self, event):
#         if self.crop_mode and event.button() == Qt.LeftButton:
#             self._start = self.mapToScene(event.pos())
#             self._end = self._start
#         super().mousePressEvent(event)

#     def mouseMoveEvent(self, event):
#         if self.crop_mode and self._start:
#             self._end = self.mapToScene(event.pos())
#             self.viewport().update()
#         super().mouseMoveEvent(event)

#     def mouseReleaseEvent(self, event):
#         if self.crop_mode and event.button() == Qt.LeftButton and self._start and self._end:
#             self._crop_rect = QRectF(self._start, self._end).normalized()
#             self._start = None
#             self._end = None
#             self.viewport().update()
#         super().mouseReleaseEvent(event)

#     def drawForeground(self, painter, rect):
#         painter.setRenderHint(QPainter.Antialiasing)
#         if self._start and self._end:
#             live_rect = QRectF(self._start, self._end).normalized()
#             painter.setPen(QPen(QColor(255, 0, 0), 2, Qt.DashLine))
#             painter.drawRect(live_rect)
#         elif self._crop_rect:
#             painter.setPen(QPen(QColor(255, 0, 0), 2, Qt.SolidLine))
#             painter.drawRect(self._crop_rect)

#     def crop_and_save(self):
#         if self._image is None or self._crop_rect is None:
#             return
#         x = int(self._crop_rect.x())
#         y = int(self._crop_rect.y())
#         w = int(self._crop_rect.width())
#         h = int(self._crop_rect.height())
#         cropped = self._image[y:y + h, x:x + w]
#         if cropped.size == 0:
#             return
#         save_path = QFileDialog.getSaveFileName(None, "Save Cropped Image", "", "Images (*.png *.jpg *.bmp)")[0]
#         if save_path:
#             cv2.imwrite(save_path, cropped)

#     def clear_crop_marker(self):
#         self._crop_rect = None
#         self.viewport().update()

#     def set_crop_mode(self, enable: bool):
#         self.crop_mode = enable
#         if enable:
#             self.setCursor(Qt.CrossCursor)
#         else:
#             self.setCursor(Qt.ArrowCursor)

#     def keyPressEvent(self, event):
#         if event.key() == Qt.Key_Escape:
#             self.set_crop_mode(False)
#         super().keyPressEvent(event)


# class MainWindow(QMainWindow):
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("Image Crop Tool")
#         self.resize(1200, 800)

#         self.viewer = ImageViewer()

#         self.upload_btn = QPushButton("Upload Image")
#         self.upload_btn.clicked.connect(self.upload_image)

#         self.marker_btn = QPushButton("Marker")
#         self.marker_btn.clicked.connect(lambda: self.viewer.set_crop_mode(True))

#         self.crop_btn = QPushButton("Crop and Save")
#         self.crop_btn.clicked.connect(self.viewer.crop_and_save)

#         self.clear_btn = QPushButton("Clear Marker")
#         self.clear_btn.clicked.connect(self.viewer.clear_crop_marker)

#         btn_layout = QHBoxLayout()
#         btn_layout.addWidget(self.upload_btn)
#         btn_layout.addWidget(self.marker_btn)
#         btn_layout.addWidget(self.crop_btn)
#         btn_layout.addWidget(self.clear_btn)

#         layout = QVBoxLayout()
#         layout.addLayout(btn_layout)
#         layout.addWidget(self.viewer)

#         container = QWidget()
#         container.setLayout(layout)
#         self.setCentralWidget(container)

#     def upload_image(self):
#         path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.bmp)")
#         if path:
#             self.viewer.load_image(path)


# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     window = MainWindow()
#     window.show()
#     sys.exit(app.exec_())






