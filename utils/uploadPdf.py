import tkinter as tk
from tkinter import filedialog, messagebox
from pdf2image import convert_from_path
from PIL import Image, ImageTk
import os

# --- Configuration ---
POPLER_PATH = r"C:\poppler\poppler-24.08.0\Library\bin"  # Make sure this is correct

# --- Main Window ---
window = tk.Tk()
window.title("PDF Viewer with Zoom")
window.geometry("1000x800")

# --- Canvas with Scrollbars ---
frame = tk.Frame(window)
frame.pack(fill=tk.BOTH, expand=True)

canvas = tk.Canvas(frame, bg='black')
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

scroll_y = tk.Scrollbar(frame, orient=tk.VERTICAL, command=canvas.yview)
scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
canvas.configure(yscrollcommand=scroll_y.set)

# --- Globals ---
original_image = None  # PIL image
zoom_level = 1.0
canvas_image_id = None

# --- Update Canvas ---
def update_canvas():
    global canvas_image_id, zoom_level
    if original_image:
        w, h = original_image.size
        resized = original_image.resize((int(w * zoom_level), int(h * zoom_level)), Image.LANCZOS)
        tk_image = ImageTk.PhotoImage(resized)

        canvas.delete("all")
        canvas_image_id = canvas.create_image(0, 0, anchor=tk.NW, image=tk_image)
        canvas.image = tk_image  # keep reference
        canvas.config(scrollregion=canvas.bbox(tk.ALL))

# --- Mouse Wheel Zoom ---
def on_mousewheel(event):
    global zoom_level
    if event.delta > 0:
        zoom_level *= 1.1
    else:
        zoom_level /= 1.1
    zoom_level = max(0.3, min(zoom_level, 5.0))
    update_canvas()

# For Linux support
canvas.bind("<Button-4>", lambda e: on_mousewheel(type('Event', (), {'delta': 120})))
canvas.bind("<Button-5>", lambda e: on_mousewheel(type('Event', (), {'delta': -120})))
canvas.bind("<MouseWheel>", on_mousewheel)

# --- PDF Upload ---
def select_pdf():
    global original_image, zoom_level
    file_path = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf")])
    if not file_path:
        return

    try:
        images = convert_from_path(file_path, dpi=200, poppler_path=POPLER_PATH)
        original_image = images[0]  # Only first page for now
        zoom_level = 1.0
        update_canvas()
    except Exception as e:
        messagebox.showerror("Error", f"Could not open PDF:\n{e}")

# --- Upload Button ---
btn = tk.Button(window, text="Upload PDF", command=select_pdf, font=("Arial", 12), bg="#007ACC", fg="white")
btn.pack(pady=10)

# --- Start ---
window.mainloop()
