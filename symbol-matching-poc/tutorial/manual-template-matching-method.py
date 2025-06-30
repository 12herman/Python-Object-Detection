import cv2
import numpy as np

# Load grayscale and color image
img_gray = cv2.imread('../assets/Electrical IFC Set (05.14.2025) 46_page_1.png', 0)
if img_gray is None:
    print("Main image not found!")
    exit()

img_color = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
method = cv2.TM_CCOEFF_NORMED
threshold = 0.544
boxes = []

# Template image paths
template_paths = [
    '../assets/object-2/object2.1-0.52.png',
    '../assets/object-2/object2.2-0.477.jpg',
    '../assets/object-2/object2.3.png',
    '../assets/object-2/object2.4-0.544.png',
]

# Rotate template
def rotate_template(template, angle):
    if angle == 0:
        return template
    elif angle == 90:
        return cv2.rotate(template, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(template, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(template, cv2.ROTATE_90_COUNTERCLOCKWISE)

# Template matching
for template_path in template_paths:
    template = cv2.imread(template_path, 0)
    if template is None:
        print(f"Template not found: {template_path}")
        continue

    for angle in [0, 90, 180, 270]:
        rotated_template = rotate_template(template, angle)
        h, w = rotated_template.shape
        res = cv2.matchTemplate(img_gray, rotated_template, method)
        loc = np.where(res >= threshold)
        for pt in zip(*loc[::-1]):
            boxes.append([pt[0], pt[1], pt[0] + w, pt[1] + h, res[pt[1], pt[0]]])

# Non-Maximum Suppression
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

# Draw boxes on color image
filtered_boxes = nms(boxes)
for (x1, y1, x2, y2, _) in filtered_boxes:
    cv2.rectangle(img_color, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

print(f"Total accurate matches found: {len(filtered_boxes)}")

# === Optimized Viewer ===

zoom = 1.0
pan = np.array([0.0, 0.0], dtype=np.float32)
dragging = False
start_point = np.array([0.0, 0.0], dtype=np.float32)
fullscreen = False

win_name = "Viewer"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(win_name, 1000, 800)

# Mouse controls
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

        if flags > 0:
            zoom_change = 1.2
        else:
            zoom_change = 0.8

        new_zoom = np.clip(zoom * zoom_change, 0.2, 5.0)
        zoom = new_zoom
        pan = mouse_pos - image_coord_before * zoom

cv2.setMouseCallback(win_name, mouse_event)

while True:
    # Get actual window size
    _, _, canvas_w, canvas_h = cv2.getWindowImageRect(win_name)
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    # Compute visible area in original image coordinates
    x0 = int(max(0, (-pan[0]) / zoom))
    y0 = int(max(0, (-pan[1]) / zoom))
    x1 = int(min(img_color.shape[1], x0 + canvas_w / zoom))
    y1 = int(min(img_color.shape[0], y0 + canvas_h / zoom))

    # Crop from image
    crop = img_color[y0:y1, x0:x1]
    if crop.size == 0:
        cv2.imshow(win_name, canvas)
        if cv2.waitKey(10) == 27:
            break
        continue

    # Resize cropped area
    resized = cv2.resize(crop, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_LINEAR)

    # Calculate where to paste on canvas
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

    # Show result
    cv2.imshow(win_name, canvas)
    key = cv2.waitKey(10)

    if key == 27:  # ESC
        break
    elif key == ord('f'):
        fullscreen = not fullscreen
        cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)

cv2.destroyAllWindows()