import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog

# === Step 1: File Selection ===
root = tk.Tk()
root.withdraw()

main_img_path = filedialog.askopenfilename(
    title="Select Main Image",
    filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")]
)

template_paths = filedialog.askopenfilenames(
    title="Select Template Images",
    filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")]
)

if not main_img_path or not template_paths:
    print("Image not selected.")
    exit()

img_gray = cv2.imread(main_img_path, 0)
img_color = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
method = cv2.TM_CCOEFF_NORMED
boxes = []

# === Rotate Template ===
def rotate_template(template, angle):
    if angle == 0:
        return template
    elif angle == 90:
        return cv2.rotate(template, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(template, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(template, cv2.ROTATE_90_COUNTERCLOCKWISE)

# === Step 2: Template Matching with Shading ===
shade_levels = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

for template_path in template_paths:
    original_template = cv2.imread(template_path, 0)
    if original_template is None:
        print(f"Template not found: {template_path}")
        continue

    for shade in shade_levels:
        shaded_template = (original_template * shade).astype(np.uint8)

        for angle in [0, 90, 180, 270]:
            rotated_template = rotate_template(shaded_template, angle)
            h, w = rotated_template.shape
            res = cv2.matchTemplate(img_gray, rotated_template, method)
            loc = np.where(res >= 0.55)  # default low threshold

            for pt in zip(*loc[::-1]):
                boxes.append([pt[0], pt[1], pt[0] + w, pt[1] + h, res[pt[1], pt[0]]])

# === Step 3: Non-Maximum Suppression ===
def nms(boxes, overlapThresh=0.3):
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes)
    x1, y1, x2, y2, scores = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3], boxes[:, 4]
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
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= overlapThresh)[0]
        order = order[inds + 1]

    return boxes[keep]

# === Step 4: Draw Boxes ===
filtered_boxes = nms(boxes)
for (x1, y1, x2, y2, _) in filtered_boxes:
    cv2.rectangle(img_color, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

print(f"Total matches found: {len(filtered_boxes)}")

# === Step 5: Interactive Viewer (Zoom + Pan) ===
zoom = 1.0
pan = np.array([0.0, 0.0], dtype=np.float32)
dragging = False
start_point = np.array([0.0, 0.0], dtype=np.float32)
fullscreen = False

win_name = "Viewer"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(win_name, 1000, 800)

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
        zoom_change = 1.2 if flags > 0 else 0.8
        zoom = np.clip(zoom * zoom_change, 0.2, 5.0)
        pan = mouse_pos - image_coord_before * zoom

cv2.setMouseCallback(win_name, mouse_event)

while True:
    _, _, canvas_w, canvas_h = cv2.getWindowImageRect(win_name)
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    x0 = int(max(0, (-pan[0]) / zoom))
    y0 = int(max(0, (-pan[1]) / zoom))
    x1 = int(min(img_color.shape[1], x0 + canvas_w / zoom))
    y1 = int(min(img_color.shape[0], y0 + canvas_h / zoom))

    crop = img_color[y0:y1, x0:x1]
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
    paste_w = min(canvas_w - paste_x, resized.shape[1] - src_x)
    paste_h = min(canvas_h - paste_y, resized.shape[0] - src_y)

    if paste_w > 0 and paste_h > 0:
        canvas[paste_y:paste_y + paste_h, paste_x:paste_x + paste_w] = \
            resized[src_y:src_y + paste_h, src_x:src_x + paste_w]

    cv2.imshow(win_name, canvas)
    key = cv2.waitKey(10)
    if key == 27:
        break
    elif key == ord('f'):
        fullscreen = not fullscreen
        cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)

cv2.destroyAllWindows()








# import cv2
# import numpy as np
# import tkinter as tk
# from tkinter import filedialog

# # === Step 1: File Selection Using Tkinter ===
# root = tk.Tk()
# root.withdraw()  # Hide the main tkinter window

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
# threshold = 0.755 # 0.544
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





















# all is working well but lag on zoom-in-out
# import cv2
# import numpy as np

# # Load grayscale main image
# img = cv2.imread('../assets/Electrical IFC Set (05.14.2025) 46_page_1.png', 0)
# if img is None:
#     print("Main image not found!")
#     exit()

# img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
# method = cv2.TM_CCOEFF_NORMED
# threshold = 0.544
# boxes = []

# # Template image paths
# template_paths = [
#     '../assets/object-2/object2.1-0.52.png',
#     '../assets/object-2/object2.2-0.477.jpg',
#     '../assets/object-2/object2.3.png',
#     '../assets/object-2/object2.4-0.544.png',
# ]

# # Rotate template
# def rotate_template(template, angle):
#     if angle == 0:
#         return template
#     elif angle == 90:
#         return cv2.rotate(template, cv2.ROTATE_90_CLOCKWISE)
#     elif angle == 180:
#         return cv2.rotate(template, cv2.ROTATE_180)
#     elif angle == 270:
#         return cv2.rotate(template, cv2.ROTATE_90_COUNTERCLOCKWISE)

# # Template matching with rotation
# for template_path in template_paths:
#     template = cv2.imread(template_path, 0)
#     if template is None:
#         print(f"Template not found: {template_path}")
#         continue

#     for angle in [0, 90, 180, 270]:
#         rotated_template = rotate_template(template, angle)
#         h, w = rotated_template.shape
#         res = cv2.matchTemplate(img, rotated_template, method)
#         loc = np.where(res >= threshold)

#         for pt in zip(*loc[::-1]):
#             boxes.append([pt[0], pt[1], pt[0] + w, pt[1] + h, res[pt[1], pt[0]]])

# # Non-Maximum Suppression
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

# # Draw filtered boxes
# filtered_boxes = nms(boxes)
# for (x1, y1, x2, y2, _) in filtered_boxes:
#     cv2.rectangle(img_color, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

# print(f"Total accurate matches found: {len(filtered_boxes)}")

# # === Smooth Viewer with Mouse-Centered Zoom & Resizable Window ===

# zoom = 1.0
# pan = np.array([0, 0], dtype=np.float32)
# dragging = False
# start_point = np.array([0, 0], dtype=np.float32)
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
#         old_zoom = zoom
#         if flags > 0:
#             zoom = min(zoom + 0.1, 5.0)
#         else:
#             zoom = max(zoom - 0.1, 0.2)

#         # Mouse-centered zoom
#         factor = zoom / old_zoom
#         mouse_pos = np.array([x, y], dtype=np.float32)
#         pan = (pan - mouse_pos) * factor + mouse_pos

# cv2.setMouseCallback(win_name, mouse_event)

# while True:
#     # Resize the image based on zoom
#     zoomed = cv2.resize(img_color, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_LINEAR)
#     zh, zw = zoomed.shape[:2]

#     # Get dynamic window size
#     _, _, canvas_w, canvas_h = cv2.getWindowImageRect(win_name)
#     canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

#     # Calculate crop area
#     start_x = int(max(0, -pan[0]))
#     start_y = int(max(0, -pan[1]))
#     end_x = int(min(start_x + canvas_w, zw))
#     end_y = int(min(start_y + canvas_h, zh))
#     crop = zoomed[start_y:end_y, start_x:end_x]

#     # Paste on canvas
#     paste_x = int(max(0, pan[0]))
#     paste_y = int(max(0, pan[1]))
#     paste_h = min(crop.shape[0], canvas_h - paste_y)
#     paste_w = min(crop.shape[1], canvas_w - paste_x)

#     if paste_h > 0 and paste_w > 0:
#         canvas[paste_y:paste_y + paste_h, paste_x:paste_x + paste_w] = crop[:paste_h, :paste_w]

#     # Show the image
#     cv2.imshow(win_name, canvas)
#     key = cv2.waitKey(10)

#     if key == 27:  # ESC to exit
#         break
#     elif key == ord('f'):  # Toggle fullscreen
#         fullscreen = not fullscreen
#         cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN,
#                               cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)

# cv2.destroyAllWindows()





# zoom-in-out is working but not correctly (is struced)
# import cv2
# import numpy as np

# # Load grayscale main image
# img = cv2.imread('../assets/Electrical IFC Set (05.14.2025) 46_page_1.png', 0)
# if img is None:
#     print("Main image not found!")
#     exit()

# img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
# method = cv2.TM_CCOEFF_NORMED
# threshold = 0.544
# boxes = []

# # List of template image paths
# template_paths = [
#     '../assets/object-2/object2.1-0.52.png',
#     '../assets/object-2/object2.2-0.477.jpg',
#     '../assets/object-2/object2.3.png',
#     '../assets/object-2/object2.4-0.544.png',
#     # '../assets/object-2/object2.5.png',
# ]

# # Rotate function
# def rotate_template(template, angle):
#     if angle == 0:
#         return template
#     elif angle == 90:
#         return cv2.rotate(template, cv2.ROTATE_90_CLOCKWISE)
#     elif angle == 180:
#         return cv2.rotate(template, cv2.ROTATE_180)
#     elif angle == 270:
#         return cv2.rotate(template, cv2.ROTATE_90_COUNTERCLOCKWISE)

# # Match each template
# for template_path in template_paths:
#     template = cv2.imread(template_path, 0)
#     if template is None:
#         print(f"Template not found: {template_path}")
#         continue

#     for angle in [0, 90, 180, 270]:
#         rotated_template = rotate_template(template, angle)
#         h, w = rotated_template.shape
#         res = cv2.matchTemplate(img, rotated_template, method)
#         loc = np.where(res >= threshold)

#         for pt in zip(*loc[::-1]):
#             boxes.append([pt[0], pt[1], pt[0] + w, pt[1] + h, res[pt[1], pt[0]]])

# # Non-Maximum Suppression (NMS)
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

# # Apply NMS and draw boxes
# filtered_boxes = nms(boxes)
# for (x1, y1, x2, y2, _) in filtered_boxes:
#     cv2.rectangle(img_color, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

# print(f"Total accurate matches found: {len(filtered_boxes)}")

# # --- Improved Viewer with Smooth Zoom, Pan, Fullscreen ---
# zoom = 1.0
# pan = np.array([0, 0], dtype=np.float32)
# dragging = False
# start_point = np.array([0, 0], dtype=np.float32)
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
#         old_zoom = zoom
#         if flags > 0:
#             zoom = min(zoom + 0.1, 5.0)
#         else:
#             zoom = max(zoom - 0.1, 0.2)

#         factor = zoom / old_zoom
#         mouse_pos = np.array([x, y], dtype=np.float32)
#         pan = (pan - mouse_pos) * factor + mouse_pos

# cv2.setMouseCallback(win_name, mouse_event)

# while True:
#     zoomed = cv2.resize(img_color, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_LINEAR)
#     zh, zw = zoomed.shape[:2]

#     canvas_h, canvas_w = 800, 1000
#     canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

#     start_x = int(max(0, -pan[0]))
#     start_y = int(max(0, -pan[1]))
#     end_x = int(min(start_x + canvas_w, zw))
#     end_y = int(min(start_y + canvas_h, zh))
#     crop = zoomed[start_y:end_y, start_x:end_x]

#     paste_x = int(max(0, pan[0]))
#     paste_y = int(max(0, pan[1]))
#     paste_h = min(crop.shape[0], canvas_h - paste_y)
#     paste_w = min(crop.shape[1], canvas_w - paste_x)

#     if paste_h > 0 and paste_w > 0:
#         canvas[paste_y:paste_y + paste_h, paste_x:paste_x + paste_w] = crop[:paste_h, :paste_w]

#     cv2.imshow(win_name, canvas)
#     key = cv2.waitKey(10)

#     if key == 27:  # ESC
#         break
#     elif key == ord('f'):  # Toggle fullscreen
#         fullscreen = not fullscreen
#         cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN,
#                               cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)

# cv2.destroyAllWindows()




# most refind accurate version
# import cv2
# import numpy as np

# # Load grayscale main image
# img = cv2.imread('../assets/Electrical IFC Set (05.14.2025) 46_page_1.png', 0)
# if img is None:
#     print("Main image not found!")
#     exit()

# img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
# method = cv2.TM_CCOEFF_NORMED
# threshold = 0.544
# boxes = []

# # List of template image paths
# template_paths = [
#     '../assets/object-2/object2.1-0.52.png',
#     '../assets/object-2/object2.2-0.477.jpg',
#     '../assets/object-2/object2.3.png',
#     '../assets/object-2/object2.4-0.544.png',
#     # '../assets/object-2/object2.5.png',
# ]

# # Rotate function
# def rotate_template(template, angle):
#     if angle == 0:
#         return template
#     elif angle == 90:
#         return cv2.rotate(template, cv2.ROTATE_90_CLOCKWISE)
#     elif angle == 180:
#         return cv2.rotate(template, cv2.ROTATE_180)
#     elif angle == 270:
#         return cv2.rotate(template, cv2.ROTATE_90_COUNTERCLOCKWISE)

# # Match each template
# for template_path in template_paths:
#     template = cv2.imread(template_path, 0)
#     if template is None:
#         print(f"Template not found: {template_path}")
#         continue

#     for angle in [0, 90, 180, 270]:
#         rotated_template = rotate_template(template, angle)
#         h, w = rotated_template.shape
#         res = cv2.matchTemplate(img, rotated_template, method)
#         loc = np.where(res >= threshold)

#         for pt in zip(*loc[::-1]):
#             boxes.append([pt[0], pt[1], pt[0] + w, pt[1] + h, res[pt[1], pt[0]]])

# # Non-Maximum Suppression (NMS)
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

# # Apply NMS and draw boxes
# filtered_boxes = nms(boxes)

# for (x1, y1, x2, y2, _) in filtered_boxes:
#     cv2.rectangle(img_color, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

# print(f"Total accurate matches found: {len(filtered_boxes)}")

# # --- Viewer State ---
# zoom = 1.0
# pan = [0, 0]
# dragging = False
# start_point = [0, 0]
# win_w, win_h = 1000, 800

# def mouse_event(event, x, y, flags, param):
#     global dragging, start_point, pan, zoom
#     if event == cv2.EVENT_LBUTTONDOWN:
#         dragging = True
#         start_point = [x - pan[0], y - pan[1]]
#     elif event == cv2.EVENT_MOUSEMOVE and dragging:
#         pan = [x - start_point[0], y - start_point[1]]
#     elif event == cv2.EVENT_LBUTTONUP:
#         dragging = False
#     elif event == cv2.EVENT_MOUSEWHEEL:
#         if flags > 0:
#             zoom_in()
#         else:
#             zoom_out()

# def zoom_in():
#     global zoom
#     zoom = min(zoom + 0.1, 5.0)

# def zoom_out():
#     global zoom
#     zoom = max(zoom - 0.1, 0.2)

# cv2.namedWindow("Viewer")
# cv2.setMouseCallback("Viewer", mouse_event)

# while True:
#     zoomed = cv2.resize(img_color, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_LINEAR)
#     canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)

#     start_x = max(0, -pan[0])
#     start_y = max(0, -pan[1])
#     end_x = min(start_x + win_w, zoomed.shape[1])
#     end_y = min(start_y + win_h, zoomed.shape[0])
#     crop = zoomed[start_y:end_y, start_x:end_x]

#     paste_x = max(0, pan[0])
#     paste_y = max(0, pan[1])
#     canvas_h, canvas_w = canvas.shape[:2]
#     crop_h, crop_w = crop.shape[:2]
#     paste_h = min(crop_h, canvas_h - paste_y)
#     paste_w = min(crop_w, canvas_w - paste_x)

#     if paste_h > 0 and paste_w > 0:
#         canvas[paste_y:paste_y + paste_h, paste_x:paste_x + paste_w] = crop[:paste_h, :paste_w]

#     cv2.imshow("Viewer", canvas)
#     if cv2.waitKey(10) == 27:  # ESC
#         break

# cv2.destroyAllWindows()







# give the accurate object count method
# import cv2
# import numpy as np

# # Load grayscale images
# img = cv2.imread('../assets/Electrical IFC Set (05.14.2025) 46_page_1.png', 0)
# template = cv2.imread('../assets/object2-0-477.jpg', 0)

# if img is None or template is None:
#     print("Image or template not found!")
#     exit()

# img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
# method = cv2.TM_CCOEFF_NORMED
# threshold = 0.477
# boxes = []

# # Rotate function
# def rotate_template(template, angle):
#     if angle == 0:
#         return template
#     elif angle == 90:
#         return cv2.rotate(template, cv2.ROTATE_90_CLOCKWISE)
#     elif angle == 180:
#         return cv2.rotate(template, cv2.ROTATE_180)
#     elif angle == 270:
#         return cv2.rotate(template, cv2.ROTATE_90_COUNTERCLOCKWISE)

# # Match with 4 rotated templates
# for angle in [0, 90, 180, 270]:
#     rotated_template = rotate_template(template, angle)
#     h, w = rotated_template.shape
#     res = cv2.matchTemplate(img, rotated_template, method)
#     loc = np.where(res >= threshold)

#     for pt in zip(*loc[::-1]):
#         boxes.append([pt[0], pt[1], pt[0] + w, pt[1] + h, res[pt[1], pt[0]]])

# # Non-Maximum Suppression (NMS)
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

# filtered_boxes = nms(boxes)

# # Draw bounding boxes
# for (x1, y1, x2, y2, _) in filtered_boxes:
#     cv2.rectangle(img_color, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

# print(f"Total accurate matches found: {len(filtered_boxes)}")

# # --- Viewer State ---
# zoom = 1.0
# pan = [0, 0]
# dragging = False
# start_point = [0, 0]
# win_w, win_h = 1000, 800

# def mouse_event(event, x, y, flags, param):
#     global dragging, start_point, pan, zoom
#     if event == cv2.EVENT_LBUTTONDOWN:
#         dragging = True
#         start_point = [x - pan[0], y - pan[1]]
#     elif event == cv2.EVENT_MOUSEMOVE and dragging:
#         pan = [x - start_point[0], y - start_point[1]]
#     elif event == cv2.EVENT_LBUTTONUP:
#         dragging = False
#     elif event == cv2.EVENT_MOUSEWHEEL:
#         if flags > 0:
#             zoom_in()
#         else:
#             zoom_out()

# def zoom_in():
#     global zoom
#     zoom = min(zoom + 0.1, 5.0)

# def zoom_out():
#     global zoom
#     zoom = max(zoom - 0.1, 0.2)

# cv2.namedWindow("Viewer")
# cv2.setMouseCallback("Viewer", mouse_event)

# while True:
#     zoomed = cv2.resize(img_color, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_LINEAR)
#     canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)

#     start_x = max(0, -pan[0])
#     start_y = max(0, -pan[1])
#     end_x = min(start_x + win_w, zoomed.shape[1])
#     end_y = min(start_y + win_h, zoomed.shape[0])
#     crop = zoomed[start_y:end_y, start_x:end_x]

#     paste_x = max(0, pan[0])
#     paste_y = max(0, pan[1])
#     canvas_h, canvas_w = canvas.shape[:2]
#     crop_h, crop_w = crop.shape[:2]
#     paste_h = min(crop_h, canvas_h - paste_y)
#     paste_w = min(crop_w, canvas_w - paste_x)

#     if paste_h > 0 and paste_w > 0:
#         canvas[paste_y:paste_y + paste_h, paste_x:paste_x + paste_w] = crop[:paste_h, :paste_w]

#     cv2.imshow("Viewer", canvas)
#     if cv2.waitKey(10) == 27:  # ESC
#         break

# cv2.destroyAllWindows()







# accurate detection method but did not give the object accurate detection count
# import cv2
# import numpy as np

# # Load grayscale images
# img = cv2.imread('../assets/upload-img-2.png', 0)
# template = cv2.imread('../assets/cropped_image.png', 0)

# if img is None or template is None:
#     print("Image or template not found!")
#     exit()

# # Prepare for drawing
# img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
# method = cv2.TM_CCOEFF_NORMED
# threshold = 0.499
# matches = []

# # Rotate function
# def rotate_template(template, angle):
#     if angle == 0:
#         return template
#     elif angle == 90:
#         return cv2.rotate(template, cv2.ROTATE_90_CLOCKWISE)
#     elif angle == 180:
#         return cv2.rotate(template, cv2.ROTATE_180)
#     elif angle == 270:
#         return cv2.rotate(template, cv2.ROTATE_90_COUNTERCLOCKWISE)

# # Match rotated templates
# for angle in [0, 90, 180, 270]:
#     rotated_template = rotate_template(template, angle)
#     h, w = rotated_template.shape
#     res = cv2.matchTemplate(img, rotated_template, method)
#     loc = np.where(res >= threshold)

#     for pt in zip(*loc[::-1]):
#         top_left = pt
#         bottom_right = (pt[0] + w, pt[1] + h)
#         matches.append((top_left, bottom_right))
#         cv2.rectangle(img_color, top_left, bottom_right, (0, 255, 0), 2)

# print(f"Total matches found: {len(matches)}")

# # --- Viewer State ---
# zoom = 1.0
# pan = [0, 0]
# dragging = False
# start_point = [0, 0]
# win_w, win_h = 1000, 800  # Window size

# def mouse_event(event, x, y, flags, param):
#     global dragging, start_point, pan, zoom
#     if event == cv2.EVENT_LBUTTONDOWN:
#         dragging = True
#         start_point = [x - pan[0], y - pan[1]]
#     elif event == cv2.EVENT_MOUSEMOVE and dragging:
#         pan = [x - start_point[0], y - start_point[1]]
#     elif event == cv2.EVENT_LBUTTONUP:
#         dragging = False
#     elif event == cv2.EVENT_MOUSEWHEEL:
#         if flags > 0:
#             zoom_in()
#         else:
#             zoom_out()

# def zoom_in():
#     global zoom
#     zoom = min(zoom + 0.1, 5.0)

# def zoom_out():
#     global zoom
#     zoom = max(zoom - 0.1, 0.2)

# cv2.namedWindow("Viewer")
# cv2.setMouseCallback("Viewer", mouse_event)

# # --- Display Loop ---
# while True:
#     zoomed = cv2.resize(img_color, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_LINEAR)
#     canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)

#     # Define visible area
#     start_x = max(0, -pan[0])
#     start_y = max(0, -pan[1])
#     end_x = min(start_x + win_w, zoomed.shape[1])
#     end_y = min(start_y + win_h, zoomed.shape[0])
#     crop = zoomed[start_y:end_y, start_x:end_x]

#     # Safe paste
#     paste_x = max(0, pan[0])
#     paste_y = max(0, pan[1])
#     canvas_h, canvas_w = canvas.shape[:2]
#     crop_h, crop_w = crop.shape[:2]
#     paste_h = min(crop_h, canvas_h - paste_y)
#     paste_w = min(crop_w, canvas_w - paste_x)

#     if paste_h > 0 and paste_w > 0:
#         canvas[paste_y:paste_y + paste_h, paste_x:paste_x + paste_w] = crop[:paste_h, :paste_w]

#     cv2.imshow("Viewer", canvas)
#     if cv2.waitKey(10) == 27:  # ESC
#         break
# cv2.destroyAllWindows()







# import cv2
# import numpy as np

# # Load grayscale images
# img = cv2.imread('../assets/upload-img-2.png', 0)
# template = cv2.imread('../assets/cropped_image.png', 0)

# if img is None or template is None:
#     print("Image or template not found!")
#     exit()

# img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
# method = cv2.TM_CCOEFF_NORMED
# threshold = 0.8
# matches = []

# # Function to rotate image 0, 90, 180, 270 degrees
# def rotate_template(template, angle):
#     if angle == 0:
#         return template
#     elif angle == 90:
#         return cv2.rotate(template, cv2.ROTATE_90_CLOCKWISE)
#     elif angle == 180:
#         return cv2.rotate(template, cv2.ROTATE_180)
#     elif angle == 270:
#         return cv2.rotate(template, cv2.ROTATE_90_COUNTERCLOCKWISE)

# # Try matching at 0°, 90°, 180°, 270°
# for angle in [0, 90, 180, 270]:
#     rotated_template = rotate_template(template, angle)
#     h, w = rotated_template.shape

#     res = cv2.matchTemplate(img, rotated_template, method)
#     loc = np.where(res >= threshold)

#     for pt in zip(*loc[::-1]):
#         top_left = pt
#         bottom_right = (pt[0] + w, pt[1] + h)
#         matches.append((top_left, bottom_right))
#         cv2.rectangle(img_color, top_left, bottom_right, (0, 255, 0), 2)

# print(f"Total matches found at 0/90/180/270°: {len(matches)}")

# # Display result
# cv2.imshow("Matched Result", img_color)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# import cv2
# import numpy as np

# # Load grayscale images
# img = cv2.imread('../assets/upload-img-2.png', 0)
# template = cv2.imread('../assets/cropped_image.png', 0)

# if img is None or template is None:
#     print("Image or template not found!")
#     exit()

# h, w = template.shape
# method = cv2.TM_CCOEFF_NORMED
# res = cv2.matchTemplate(img, template, method)

# # Threshold for detecting matches (adjustable)
# threshold = 0.8
# locations = np.where(res >= threshold)
# match_locations = list(zip(*locations[::-1]))  # (x, y) format

# # Convert grayscale to BGR to draw colored rectangles
# img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
# for loc in match_locations:
#     top_left = loc
#     bottom_right = (top_left[0] + w, top_left[1] + h)
#     cv2.rectangle(img_color, top_left, bottom_right, (0, 255, 0), 2)

# print(f"Total matches found: {len(match_locations)}")

# # --- Viewer State ---
# zoom = 1.0
# pan = [0, 0]
# dragging = False
# start_point = [0, 0]

# def mouse_event(event, x, y, flags, param):
#     global dragging, start_point, pan, zoom
#     if event == cv2.EVENT_LBUTTONDOWN:
#         dragging = True
#         start_point = [x - pan[0], y - pan[1]]
#     elif event == cv2.EVENT_MOUSEMOVE and dragging:
#         pan = [x - start_point[0], y - start_point[1]]
#     elif event == cv2.EVENT_LBUTTONUP:
#         dragging = False
#     elif event == cv2.EVENT_MOUSEWHEEL:
#         if flags > 0:
#             zoom_in()
#         else:
#             zoom_out()

# def zoom_in():
#     global zoom
#     zoom = min(zoom + 0.1, 5.0)

# def zoom_out():
#     global zoom
#     zoom = max(zoom - 0.1, 0.2)

# cv2.namedWindow("High-Quality Viewer")
# cv2.setMouseCallback("High-Quality Viewer", mouse_event)

# win_w, win_h = 1000, 800  # Output window size

# while True:
#     # Resize the image according to current zoom
#     zoomed = cv2.resize(img_color, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_LINEAR)
#     canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)

#     # Crop visible region from zoomed image
#     start_x = max(0, -pan[0])
#     start_y = max(0, -pan[1])
#     end_x = min(start_x + win_w, zoomed.shape[1])
#     end_y = min(start_y + win_h, zoomed.shape[0])
#     crop = zoomed[start_y:end_y, start_x:end_x]

#     # Paste crop into canvas safely
#     paste_x = max(0, pan[0])
#     paste_y = max(0, pan[1])

#     canvas_h, canvas_w = canvas.shape[:2]
#     crop_h, crop_w = crop.shape[:2]
#     paste_h = min(crop_h, canvas_h - paste_y)
#     paste_w = min(crop_w, canvas_w - paste_x)

#     if paste_h > 0 and paste_w > 0:
#         canvas[paste_y:paste_y + paste_h, paste_x:paste_x + paste_w] = crop[:paste_h, :paste_w]

#     cv2.imshow("High-Quality Viewer", canvas)
#     key = cv2.waitKey(10)
#     if key == 27:  # ESC
#         break

# cv2.destroyAllWindows()