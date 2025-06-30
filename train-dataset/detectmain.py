import cv2
import os
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from ultralytics import YOLO
from PIL import Image, ImageTk

tile_size = 640
overlap = 100
model_path = 'project-1-at-2025-06-11-10-38-4d23685d/runs/detect/train7/weights/best.pt'  # Update if needed

model = YOLO(model_path)

def split_image(image_path):
    img = cv2.imread(image_path)
    height, width = img.shape[:2]
    tiles, coords = [], []

    for y in range(0, height, tile_size - overlap):
        for x in range(0, width, tile_size - overlap):
            tile = img[y:y + tile_size, x:x + tile_size]
            if tile.shape[0] != tile_size or tile.shape[1] != tile_size:
                pad_y = tile_size - tile.shape[0]
                pad_x = tile_size - tile.shape[1]
                tile = cv2.copyMakeBorder(tile, 0, pad_y, 0, pad_x, cv2.BORDER_CONSTANT, value=(114, 114, 114))
            tiles.append(tile)
            coords.append((x, y))
    return img, tiles, coords

def draw_boxes_on_image(base_image, detections, coords):
    for det, (x_offset, y_offset) in zip(detections, coords):
        if det and det.boxes:
            for box in det.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = box[:4]
                x1, x2 = int(x1 + x_offset), int(x2 + x_offset)
                y1, y2 = int(y1 + y_offset), int(y2 + y_offset)
                cv2.rectangle(base_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return base_image

def process_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if not file_path:
        return

    base_image, tiles, coords = split_image(file_path)
    detections = [model(tile, verbose=False)[0] for tile in tiles]
    result = draw_boxes_on_image(base_image, detections, coords)

    save_path = os.path.splitext(file_path)[0] + "_detected.jpg"
    cv2.imwrite(save_path, result)
    messagebox.showinfo("Success", f"Detection completed and saved as:\n{save_path}")

    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(result_rgb)
    img_pil.thumbnail((600, 600))
    img_tk = ImageTk.PhotoImage(img_pil)
    label_img.configure(image=img_tk)
    label_img.image = img_tk

root = tk.Tk()
root.title("YOLOv8 Large Image Object Detection")
root.geometry("700x700")

btn = tk.Button(root, text="Upload & Detect", command=process_image, font=("Arial", 14))
btn.pack(pady=10)

label_img = tk.Label(root)
label_img.pack()

root.mainloop()



# import cv2
# import os
# import numpy as np
# from ultralytics import YOLO

# def split_image(image_path, tile_size=640, overlap=100):
#     img = cv2.imread(image_path)
#     height, width = img.shape[:2]
#     tiles, coords = [], []

#     for y in range(0, height, tile_size - overlap):
#         for x in range(0, width, tile_size - overlap):
#             tile = img[y:y + tile_size, x:x + tile_size]
#             # Pad smaller tiles
#             if tile.shape[0] != tile_size or tile.shape[1] != tile_size:
#                 pad_y = tile_size - tile.shape[0]
#                 pad_x = tile_size - tile.shape[1]
#                 tile = cv2.copyMakeBorder(tile, 0, pad_y, 0, pad_x,
#                                           cv2.BORDER_CONSTANT, value=(114, 114, 114))
#             tiles.append(tile)
#             coords.append((x, y))
#     return img, tiles, coords

# def draw_boxes_on_image(base_image, detections, coords):
#     for det, (x_offset, y_offset) in zip(detections, coords):
#         if det and det.boxes:
#             for box in det.boxes.xyxy.cpu().numpy():
#                 x1, y1, x2, y2 = box[:4]
#                 x1, x2 = int(x1 + x_offset), int(x2 + x_offset)
#                 y1, y2 = int(y1 + y_offset), int(y2 + y_offset)
#                 cv2.rectangle(base_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
#     return base_image

# # === Edit these paths ===
# large_image_path = 'your_image.jpg'
# model_path = 'runs/detect/train5/weights/best.pt'

# # Load YOLO model
# model = YOLO(model_path)

# # Step 1: Split large image
# base_image, tiles, coords = split_image(large_image_path, tile_size=640, overlap=100)

# # Step 2: Predict each tile
# detections = [model(tile, verbose=False)[0] for tile in tiles]

# # Step 3: Draw boxes on the original large image
# result = draw_boxes_on_image(base_image, detections, coords)

# # Step 4: Save final output
# cv2.imwrite('final_detection_result.jpg', result)
# print("Detection complete. Result saved as 'final_detection_result.jpg'")
