import cv2
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

tile_size = 1280

def split_images_in_folder(folder_path):
    output_folder = os.path.join(folder_path, "tiles")
    os.makedirs(output_folder, exist_ok=True)
    count = 0

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(folder_path, filename)
            img = cv2.imread(image_path)
            if img is None:
                continue
            height, width = img.shape[:2]
            for y in range(0, height, tile_size):
                for x in range(0, width, tile_size):
                    tile = img[y:y+tile_size, x:x+tile_size]
                    tile_name = f"{output_folder}/{os.path.splitext(filename)[0]}_tile_{count}.jpg"
                    cv2.imwrite(tile_name, tile)
                    count += 1
    return count

def browse_folder():
    folder_selected = filedialog.askdirectory()
    if folder_selected:
        count = split_images_in_folder(folder_selected)
        messagebox.showinfo("Done", f"{count} tiles saved to 'tiles' folder.")

# Create simple GUI
root = tk.Tk()
root.title("Image Splitter 1280X1280")
root.geometry("300x150")

label = tk.Label(root, text="Click below to choose folder:")
label.pack(pady=20)

btn = tk.Button(root, text="Choose Folder and Split", command=browse_folder)
btn.pack()

root.mainloop()
