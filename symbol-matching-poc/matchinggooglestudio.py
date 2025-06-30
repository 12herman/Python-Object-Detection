import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QGraphicsView, QGraphicsScene,
    QGraphicsPixmapItem, QProgressDialog, QToolBar, QAction, QSlider, QLabel,
    QMessageBox
)
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QPen
from PyQt5.QtCore import Qt, QRectF, pyqtSignal

class ImageViewer(QGraphicsView):
    detection_finished = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.scene = QGraphicsScene()
        self.image_item = QGraphicsPixmapItem()
        self.setScene(self.scene); self.scene.addItem(self.image_item)
        self._image, self._clone = None, None
        self._boxes, self._object_colors = [], {}
        self._drawing, self._pan = False, False
        self._start, self._end, self._start_pan_pos = None, None, None
        self._object_id = 1; self._enable_drawing = False
        self._threshold = 0.8; self._preview_box = None
        self.setRenderHint(QPainter.Antialiasing); self.setDragMode(QGraphicsView.NoDrag)

    def set_threshold(self, value): self._threshold = value
    def toggle_drawing_mode(self, checked): self._enable_drawing = checked; self.setCursor(Qt.CrossCursor if checked else Qt.ArrowCursor)

    def load_image(self, path):
        self._image = cv2.imread(path)
        if self._image is None: QMessageBox.critical(self, "Error", f"Failed to load image from {path}"); return
        self._clone = self._image.copy()
        h, w, ch = self._image.shape
        q_img = QImage(self._image.data, w, h, ch * w, QImage.Format_BGR888)
        self.image_item.setPixmap(QPixmap.fromImage(q_img))
        self.setSceneRect(QRectF(self.image_item.pixmap().rect()))
        self._boxes.clear(); self._object_colors.clear(); self._object_id = 1
        self.resetTransform(); self.fitInView(self.image_item, Qt.KeepAspectRatio); self.update()

    def wheelEvent(self, event):
        modifiers = event.modifiers()
        zoom_factor_in, zoom_factor_out = 1.1, 0.9
        if modifiers == Qt.ControlModifier:
            old_pos = self.mapToScene(event.pos())
            factor = zoom_factor_in if event.angleDelta().y() > 0 else zoom_factor_out
            self.scale(factor, factor); new_pos = self.mapToScene(event.pos())
            self.translate((new_pos - old_pos).x(), (new_pos - old_pos).y())
        elif modifiers == Qt.ShiftModifier: self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - event.angleDelta().y())
        else: self.verticalScrollBar().setValue(self.verticalScrollBar().value() - event.angleDelta().y())

    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton: self._pan = True; self._start_pan_pos = event.pos(); self.setCursor(Qt.ClosedHandCursor)
        elif event.button() == Qt.LeftButton and self._enable_drawing: self._drawing = True; self._start = self._end = self.mapToScene(event.pos())
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._pan:
            delta = event.pos() - self._start_pan_pos; self._start_pan_pos = event.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            return
        if self._drawing: self._end = self.mapToScene(event.pos()); self.viewport().update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.RightButton and self._pan: self._pan = False; self.setCursor(Qt.CrossCursor if self._enable_drawing else Qt.ArrowCursor)
        if self._drawing and event.button() == Qt.LeftButton:
            self._drawing = False
            x1, y1 = int(self._start.x()), int(self._start.y()); x2, y2 = int(self._end.x()), int(self._end.y())
            x, y, w, h = min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1)
            if w > 10 and h > 10:
                self._preview_box = (x, y, w, h); self.viewport().update()
                reply = QMessageBox.question(self, "Confirm Selection", "Detect objects based on this selection?", QMessageBox.Yes | QMessageBox.Cancel)
                if reply == QMessageBox.Yes: self.detect_objects(self._preview_box)
                self._preview_box = None; self.viewport().update()
                self.detection_finished.emit()
        super().mouseReleaseEvent(event)

    def detect_objects(self, box):
        if self._clone is None: return
        progress = QProgressDialog("Detecting objects...", None, 0, 0, self)
        progress.setWindowTitle("Please wait"); progress.setWindowModality(Qt.ApplicationModal)
        progress.setCancelButton(None); progress.show(); QApplication.processEvents()
        
        try:
            x_ref, y_ref, w_ref, h_ref = box
            template_ref = self._clone[y_ref:y_ref + h_ref, x_ref:x_ref + w_ref]
            gray_img = cv2.cvtColor(self._clone, cv2.COLOR_BGR2GRAY)
            
            flipped_template_ref = cv2.flip(template_ref, 1)
            templates_to_check = [(template_ref, "Original"), (flipped_template_ref, "Flipped")]
            
            all_found_rects = []
            sift = cv2.SIFT_create(nfeatures=5000)
            kp_img, des_img = sift.detectAndCompute(gray_img, None)

            for tpl, tpl_name in templates_to_check:
                gray_tpl = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY)
                h_tpl, w_tpl = gray_tpl.shape
                
                # --- [METHOD 1] THE "PERFECT" MATCH (High Certainty) ---
                print(f"🚀 [METHOD 1] Searching for a 'Perfect Match' with {tpl_name} template...")
                result = cv2.matchTemplate(gray_img, gray_tpl, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)
                
                if max_val > 0.95:
                    print(f"✅ Found a perfect match (Score: {max_val:.2f})!")
                    all_found_rects.append((max_loc[0], max_loc[1], w_tpl, h_tpl))
                    continue # Found a perfect match, no need for other methods for THIS template

                # --- [METHOD 2] "GOOD ENOUGH" MATCHES (User Threshold) ---
                print(f"⚠️ [METHOD 2] No perfect match. Searching with threshold {self._threshold:.2f}...")
                loc = np.where(result >= self._threshold)
                found_count = len(loc[0])
                if found_count > 0:
                    print(f"✅ Found {found_count} good enough matches.")
                    rects = [(pt[0], pt[1], w_tpl, h_tpl) for pt in zip(*loc[::-1])]
                    all_found_rects.extend(rects)
                    continue # Found good matches, no need for SIFT for THIS template

                # --- [METHOD 3] "TRANSFORMED" MATCHES (SIFT + Homography) ---
                if des_img is None: continue
                print(f"⚠️ [METHOD 3] Template matching failed. Trying SIFT...")
                kp_tpl, des_tpl = sift.detectAndCompute(gray_tpl, None)
                if des_tpl is not None and len(kp_tpl) > 4:
                    bf = cv2.BFMatcher(); matches = bf.knnMatch(des_tpl, des_img, k=2)
                    good_matches = [m for m, n in matches if len(m_n) == 2 and m.distance < 0.75 * n.distance for m_n in matches]
                    
                    if len(good_matches) >= 7:
                        src_pts = np.float32([kp_tpl[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
                        dst_pts = np.float32([kp_img[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)
                        M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                        if M is not None:
                            pts = np.float32([[0,0],[0,h_tpl-1],[w_tpl-1,h_tpl-1],[w_tpl-1,0]]).reshape(-1,1,2)
                            dst = cv2.perspectiveTransform(pts, M)
                            all_found_rects.append(cv2.boundingRect(dst))
                            print(f"✅ SIFT found a transformed match.")

            # --- Post-processing: Merge all found boxes from all methods ---
            if all_found_rects:
                merged_rects, _ = cv2.groupRectangles(all_found_rects, 1, 0.4)
                print(f"📦 Found {len(merged_rects)} unique object(s) after merging.")
                
                hue = (self._object_id * 47) % 360
                color = QColor.fromHsv(hue, 255, 255)
                self._object_colors[self._object_id] = color
                for rect in merged_rects: self._boxes.append((rect, self._object_id))
                self._object_id += 1
            else:
                QMessageBox.information(self, "Detection Complete", "No similar objects could be identified.")

        except Exception as e:
            print(f"An error occurred during detection: {e}"); QMessageBox.critical(self, "Error", f"An error occurred: {e}")
        finally:
            progress.close(); self.update()

    def drawForeground(self, painter, rect):
        painter.setRenderHint(QPainter.Antialiasing)
        if self._drawing: painter.setPen(QPen(QColor(255,0,0,200),2,Qt.DashLine)); painter.setBrush(QColor(255,0,0,50)); painter.drawRect(QRectF(self._start,self._end))
        if self._preview_box: x,y,w,h = self._preview_box; painter.setPen(QPen(QColor(0,255,0,220),2,Qt.DashDotLine)); painter.setBrush(QColor(0,255,0,60)); painter.drawRect(x,y,w,h)
        for (x,y,w,h), obj_id in self._boxes:
            color = self._object_colors.get(obj_id, QColor(0,255,0))
            painter.setPen(QPen(color, 2)); painter.setBrush(Qt.NoBrush); painter.drawRect(x,y,w,h)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__(); self.viewer = ImageViewer(); self.setCentralWidget(self.viewer)
        self.setWindowTitle("Smart Object Detection (Reliable)"); self.resize(1200, 800)
        self.toolbar = QToolBar("Main Toolbar"); self.addToolBar(self.toolbar)
        open_action = QAction("Open Image", self); open_action.triggered.connect(self.open_image)
        self.toolbar.addAction(open_action); self.toolbar.addSeparator()
        self.marker_action = QAction("Enable Marker", self); self.marker_action.setCheckable(True)
        self.marker_action.toggled.connect(self.toggle_marker_mode); self.toolbar.addAction(self.marker_action)
        self.slider_label = QLabel("  Template Fallback Threshold: 80%")
        self.threshold_slider = QSlider(Qt.Horizontal); self.threshold_slider.setMinimum(50)
        self.threshold_slider.setMaximum(99); self.threshold_slider.setValue(80)
        self.threshold_slider.valueChanged.connect(self.update_threshold)
        self.toolbar.addWidget(self.slider_label); self.toolbar.addWidget(self.threshold_slider)
        self.viewer.detection_finished.connect(lambda: self.marker_action.setChecked(False))
        self.open_image()
    def toggle_marker_mode(self, checked): self.viewer.toggle_drawing_mode(checked); self.marker_action.setText("Disable Marker" if checked else "Enable Marker")
    def update_threshold(self, value): self.slider_label.setText(f"  Template Fallback Threshold: {value}%"); self.viewer.set_threshold(value / 100.0)
    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if path: self.viewer.load_image(path)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())