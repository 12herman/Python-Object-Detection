import cv2
import numpy as np
from tkinter import filedialog, Tk
import os

# === Step 1: GUI File Selection ===
root = Tk()
root.withdraw()

main_img_path = filedialog.askopenfilename(title="Select Main Image")
if not main_img_path:
    print("No main image selected.")
    exit()

template_paths = filedialog.askopenfilenames(title="Select Template Images")
if not template_paths:
    print("No template images selected.")
    exit()

# === Step 2: Load Main Image ===
main_gray = cv2.imread(main_img_path, 0)
main_color = cv2.cvtColor(main_gray, cv2.COLOR_GRAY2BGR)

# === Step 3: Template Matching ===
method = cv2.TM_CCOEFF_NORMED
threshold = 0.7  # Adjusted for better accuracy
boxes = []

for path in template_paths:
    label = os.path.basename(path)
    template = cv2.imread(path, 0)
    if template is None:
        print(f"Error loading template: {label}")
        continue
    h, w = template.shape
    res = cv2.matchTemplate(main_gray, template, method)
    loc = np.where(res >= threshold)

    for pt in zip(*loc[::-1]):
        score = res[pt[1], pt[0]]
        boxes.append((pt, (pt[0] + w, pt[1] + h), label, score))

# === Step 4: Non-Maximum Suppression ===
def nms(boxes, overlapThresh=0.3):
    if len(boxes) == 0:
        return []
    rects = np.array([[x1, y1, x2, y2, score] for (x1, y1), (x2, y2), _, score in boxes])
    x1 = rects[:, 0]
    y1 = rects[:, 1]
    x2 = rects[:, 2]
    y2 = rects[:, 3]
    scores = rects[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= overlapThresh)[0]
        order = order[inds + 1]
    return [boxes[i] for i in keep]

filtered_boxes = nms(boxes)

# === Step 5: Draw Labels & Boxes ===
for (pt1, pt2, label, _) in filtered_boxes:
    cv2.rectangle(main_color, pt1, pt2, (0, 255, 0), 2)
    cv2.putText(main_color, label, (pt1[0], pt1[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

print(f"Detected: {len(filtered_boxes)} elements")

# === Step 6: Zoom + Pan Viewer ===
zoom = 1.0
pan = np.array([0.0, 0.0], dtype=np.float32)
dragging = False
start_point = np.array([0.0, 0.0], dtype=np.float32)
win_name = "Viewer"

cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(win_name, 1200, 800)

def mouse_event(event, x, y, flags, param):
    global dragging, start_point, pan, zoom
    if event == cv2.EVENT_LBUTTONDOWN:
        dragging = True
        start_point[:] = [x, y]
    elif event == cv2.EVENT_MOUSEMOVE and dragging:
        delta = np.array([x, y], dtype=np.float32) - start_point
        pan += delta
        start_point[:] = [x, y]
    elif event == cv2.EVENT_LBUTTONUP:
        dragging = False
    elif event == cv2.EVENT_MOUSEWHEEL:
        mouse_pos = np.array([x, y], dtype=np.float32)
        image_coord_before = (mouse_pos - pan) / zoom
        zoom *= 1.2 if flags > 0 else 0.8
        zoom = np.clip(zoom, 0.2, 5.0)
        pan = mouse_pos - image_coord_before * zoom

cv2.setMouseCallback(win_name, mouse_event)

# === Step 7: Show Viewer ===
while True:
    _, _, win_w, win_h = cv2.getWindowImageRect(win_name)
    canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)

    x0 = int(max(0, (-pan[0]) / zoom))
    y0 = int(max(0, (-pan[1]) / zoom))
    x1 = int(min(main_color.shape[1], x0 + win_w / zoom))
    y1 = int(min(main_color.shape[0], y0 + win_h / zoom))

    crop = main_color[y0:y1, x0:x1]
    if crop.size == 0:
        cv2.imshow(win_name, canvas)
        if cv2.waitKey(10) == 27:
            break
        continue

    resized = cv2.resize(crop, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_LINEAR)
    offset_x = int(pan[0] + x0 * zoom)
    offset_y = int(pan[1] + y0 * zoom)
    paste_x = max(0, offset_x)
    paste_y = max(0, offset_y)
    src_x = max(0, -offset_x)
    src_y = max(0, -offset_y)
    paste_w = min(win_w - paste_x, resized.shape[1] - src_x)
    paste_h = min(win_h - paste_y, resized.shape[0] - src_y)

    if paste_w > 0 and paste_h > 0:
        canvas[paste_y:paste_y + paste_h, paste_x:paste_x + paste_w] = \
            resized[src_y:src_y + paste_h, src_x:src_x + paste_w]

    cv2.imshow(win_name, canvas)
    if cv2.waitKey(10) == 27:
        break

cv2.destroyAllWindows()




# #Perfect Version 
# import cv2
# import numpy as np
# # import tkinter as tk
# from tkinter import filedialog

# # === Step 1: File Selection Using Tkinter ===
# # root = tk.Tk()
# # root.withdraw()  # Hide the main tkinter window

# # Ask user to select the main image
# main_img_path = filedialog.askopenfilename(
#     title="Select Main Image",
#     filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")]
# )

# if not main_img_path:
#     print("Main image not selected. Exiting.")
#     exit()

# # Ask user to select template images
# template_paths = filedialog.askopenfilenames(
#     title="Select Template Images",
#     filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")]
# )

# if not template_paths:
#     print("No templates selected. Exiting.")
#     exit()

# # === Step 2: Load Images ===
# img_gray = cv2.imread(main_img_path, 0)
# if img_gray is None:
#     print("Failed to load main image.")
#     exit()

# img_color = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
# method = cv2.TM_CCOEFF_NORMED
# threshold = 0.55 #0.755 , 0.544
# boxes = []

# # === Step 3: Rotate Template Function ===
# def rotate_template(template, angle):
#     if angle == 0:
#         return template
#     elif angle == 90:
#         return cv2.rotate(template, cv2.ROTATE_90_CLOCKWISE)
#     elif angle == 180:
#         return cv2.rotate(template, cv2.ROTATE_180)
#     elif angle == 270:
#         return cv2.rotate(template, cv2.ROTATE_90_COUNTERCLOCKWISE)

# # === Step 4: Template Matching ===
# for template_path in template_paths:
#     template = cv2.imread(template_path, 0)
#     if template is None:
#         print(f"Template not found: {template_path}")
#         continue

#     for angle in [0, 90, 180, 270]:
#         rotated_template = rotate_template(template, angle)
#         h, w = rotated_template.shape
#         res = cv2.matchTemplate(img_gray, rotated_template, method)
#         loc = np.where(res >= threshold)
#         for pt in zip(*loc[::-1]):
#             boxes.append([pt[0], pt[1], pt[0] + w, pt[1] + h, res[pt[1], pt[0]]])

# # === Step 5: Non-Maximum Suppression ===
# def nms(boxes, overlapThresh=0.3):
#     if len(boxes) == 0:
#         return []
#     boxes = np.array(boxes)
#     x1, y1, x2, y2, scores = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3], boxes[:, 4]
#     areas = (x2 - x1 + 1) * (y2 - y1 + 1)
#     order = scores.argsort()[::-1]
#     keep = []

#     while order.size > 0:
#         i = order[0]
#         keep.append(i)

#         xx1 = np.maximum(x1[i], x1[order[1:]])
#         yy1 = np.maximum(y1[i], y1[order[1:]])
#         xx2 = np.minimum(x2[i], x2[order[1:]])
#         yy2 = np.minimum(y2[i], y2[order[1:]])

#         w = np.maximum(0, xx2 - xx1 + 1)
#         h = np.maximum(0, yy2 - yy1 + 1)
#         inter = w * h
#         iou = inter / (areas[i] + areas[order[1:]] - inter)
#         inds = np.where(iou <= overlapThresh)[0]
#         order = order[inds + 1]

#     return boxes[keep]

# # Draw matched boxes
# filtered_boxes = nms(boxes)
# for (x1, y1, x2, y2, _) in filtered_boxes:
#     cv2.rectangle(img_color, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

# print(f"Total accurate matches found: {len(filtered_boxes)}")

# # === Step 6: Interactive Viewer with Zoom & Pan ===
# zoom = 1.0
# pan = np.array([0.0, 0.0], dtype=np.float32)
# dragging = False
# start_point = np.array([0.0, 0.0], dtype=np.float32)
# fullscreen = False

# win_name = "Viewer"
# cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
# cv2.resizeWindow(win_name, 1000, 800)

# def mouse_event(event, x, y, flags, param):
#     global dragging, start_point, pan, zoom

#     if event == cv2.EVENT_LBUTTONDOWN:
#         dragging = True
#         start_point[:] = [x, y]
#     elif event == cv2.EVENT_MOUSEMOVE and dragging:
#         delta = np.array([x, y], dtype=np.float32) - start_point
#         pan += delta
#         start_point[:] = [x, y]
#     elif event == cv2.EVENT_LBUTTONUP:
#         dragging = False
#     elif event == cv2.EVENT_MOUSEWHEEL:
#         mouse_pos = np.array([x, y], dtype=np.float32)
#         image_coord_before = (mouse_pos - pan) / zoom
#         zoom_change = 1.2 if flags > 0 else 0.8
#         zoom = np.clip(zoom * zoom_change, 0.2, 5.0)
#         pan = mouse_pos - image_coord_before * zoom

# cv2.setMouseCallback(win_name, mouse_event)

# while True:
#     _, _, canvas_w, canvas_h = cv2.getWindowImageRect(win_name)
#     canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

#     x0 = int(max(0, (-pan[0]) / zoom))
#     y0 = int(max(0, (-pan[1]) / zoom))
#     x1 = int(min(img_color.shape[1], x0 + canvas_w / zoom))
#     y1 = int(min(img_color.shape[0], y0 + canvas_h / zoom))

#     crop = img_color[y0:y1, x0:x1]
#     if crop.size == 0:
#         cv2.imshow(win_name, canvas)
#         if cv2.waitKey(10) == 27:
#             break
#         continue

#     resized = cv2.resize(crop, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_LINEAR)
#     offset_x = int(pan[0] + x0 * zoom)
#     offset_y = int(pan[1] + y0 * zoom)
#     paste_x = max(0, offset_x)
#     paste_y = max(0, offset_y)
#     src_x = max(0, -offset_x)
#     src_y = max(0, -offset_y)
#     paste_w = min(canvas_w - paste_x, resized.shape[1] - src_x)
#     paste_h = min(canvas_h - paste_y, resized.shape[0] - src_y)

#     if paste_w > 0 and paste_h > 0:
#         canvas[paste_y:paste_y + paste_h, paste_x:paste_x + paste_w] = \
#             resized[src_y:src_y + paste_h, src_x:src_x + paste_w]

#     cv2.imshow(win_name, canvas)
#     key = cv2.waitKey(10)

#     if key == 27:  # ESC
#         break
#     elif key == ord('f'):
#         fullscreen = not fullscreen
#         cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN,
#                               cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)

# cv2.destroyAllWindows()