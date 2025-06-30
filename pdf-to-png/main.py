import tkinter as tk
from tkinter import filedialog, messagebox, Toplevel, Label
from pdf2image import convert_from_path
import os
from PIL import Image
import threading

POPLER_PATH = r"C:\poppler\poppler-24.08.0\Library\bin"  # Change this to your Poppler path

# Show loading popup with dynamic label
def show_loading_popup():
    popup = Toplevel(window)
    popup.title("Processing...")
    popup.geometry("250x120")
    popup.transient(window)
    popup.grab_set()
    label = Label(popup, text="convert to image please wait...", pady=20)
    label.pack()
    return popup, label

def convert_pdf_to_images(pdf_path, output_folder, update_label=None):
    images = convert_from_path(pdf_path, poppler_path=POPLER_PATH)
    total = len(images)
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]

    for i, img in enumerate(images):
        image_path = os.path.join(output_folder, f"{base_name}_page_{i+1}.png")
        img.save(image_path, 'PNG')
        if update_label:
            percent = int(((i+1)/total)*100)
            update_label(f"Converting {i+1}/{total} pages ({percent}%)")

def process_file_thread():
    file_path = filedialog.askopenfilename(
        title="Select a PDF file",
        filetypes=[("PDF Files", "*.pdf")]
    )
    if not file_path:
        return

    output_folder = filedialog.askdirectory(title="Select Folder to Save Images")
    if not output_folder:
        return

    popup, label = show_loading_popup()

    def update_label(text):
        label.config(text=text)
        label.update_idletasks()

    def task():
        try:
            convert_pdf_to_images(file_path, output_folder, update_label)
            popup.destroy()
            messagebox.showinfo("Success", f"PDF converted and saved in:\n{output_folder}")
        except Exception as e:
            popup.destroy()
            messagebox.showerror("Error", str(e))

    threading.Thread(target=task).start()

def process_folder_thread():
    folder_path = filedialog.askdirectory(title="Select Folder Containing PDFs")
    if not folder_path:
        return

    output_folder = filedialog.askdirectory(title="Select Folder to Save Images")
    if not output_folder:
        return

    pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]
    if not pdf_files:
        messagebox.showwarning("No PDFs", "No PDF files found in the selected folder.")
        return

    popup, label = show_loading_popup()

    def update_label(text):
        label.config(text=text)
        label.update_idletasks()

    def task():
        try:
            for index, pdf_file in enumerate(pdf_files):
                update_label(f"Processing file {index+1}/{len(pdf_files)}")
                full_path = os.path.join(folder_path, pdf_file)
                convert_pdf_to_images(full_path, output_folder, update_label)
            popup.destroy()
            messagebox.showinfo("Success", f"All PDFs converted and saved in:\n{output_folder}")
        except Exception as e:
            popup.destroy()
            messagebox.showerror("Error", str(e))

    threading.Thread(target=task).start()

# GUI
window = tk.Tk()
window.title("PDF to Image Converter")
window.geometry("400x200")

tk.Button(window, text="Upload PDF File", command=process_file_thread, width=30).pack(pady=20)
tk.Button(window, text="Upload PDF Folder", command=process_folder_thread, width=30).pack(pady=10)

window.mainloop()