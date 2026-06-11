import tkinter as tk
from tkinter import filedialog, messagebox
import os
from PIL import Image, ImageTk
from ocr.image_loader import ImageLoader
from utils.image_utils import resize_keep_aspect, to_pil

class UploadScreen(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg="#1e1e24")
        self.controller = controller
        self.selected_file_path = None
        self.preview_image_ref = None  # Reference to avoid garbage collection
        self._init_ui()
        
        # Pre-populate if we are returning to this screen
        if self.controller.current_packet:
            self._prepopulate_packet(self.controller.current_packet)

    def _init_ui(self):
        main_frame = tk.Frame(self, bg="#1e1e24", padx=40, pady=30)
        main_frame.pack(fill="both", expand=True)

        # 1. Navigation Header
        nav_frame = tk.Frame(main_frame, bg="#1e1e24")
        nav_frame.pack(fill="x", pady=(0, 20))

        back_btn = tk.Label(nav_frame, text="← BACK TO DASHBOARD", fg="#00bcd4", bg="#1e1e24", font=("Segoe UI Bold", 10), cursor="hand2")
        back_btn.pack(side="left")
        back_btn.bind("<Button-1>", lambda e: self.controller.show_frame("HomeScreen"))

        # 2. Main content split layout (Left: Upload Zone & Preview | Right: Instructions)
        content_frame = tk.Frame(main_frame, bg="#1e1e24")
        content_frame.pack(fill="both", expand=True)

        # Left Column
        left_col = tk.Frame(content_frame, bg="#282830", padx=25, pady=25, highlightbackground="#3f3f46", highlightthickness=1)
        left_col.pack(side="left", fill="both", expand=True, padx=(0, 15))

        self.title_lbl = tk.Label(left_col, text="UPLOAD IDENTITY DOCUMENT", fg="#ffffff", bg="#282830", font=("Segoe UI Bold", 14))
        self.title_lbl.pack(anchor="w")

        # Interactive drop area / click button
        self.upload_box = tk.Frame(left_col, bg="#1e1e24", bd=2, relief="solid", highlightthickness=0, height=220)
        self.upload_box.pack(fill="x", pady=20)
        self.upload_box.pack_propagate(False)

        self.upload_icon_lbl = tk.Label(
            self.upload_box, 
            text="📁", 
            fg="#00bcd4", 
            bg="#1e1e24", 
            font=("Segoe UI", 36)
        )
        self.upload_icon_lbl.pack(expand=True, pady=(25, 0))

        self.upload_text_lbl = tk.Label(
            self.upload_box, 
            text="Click to select JPEG, PNG, TIFF, or PDF file", 
            fg="#a0aec0", 
            bg="#1e1e24", 
            font=("Segoe UI", 10)
        )
        self.upload_text_lbl.pack(expand=True, pady=(0, 25))

        # Event bind to upload box for click
        self.upload_box.bind("<Button-1>", lambda e: self._browse_file())
        self.upload_icon_lbl.bind("<Button-1>", lambda e: self._browse_file())
        self.upload_text_lbl.bind("<Button-1>", lambda e: self._browse_file())

        # Path label
        self.path_lbl = tk.Label(left_col, text="No file selected", fg="#a0aec0", bg="#282830", font=("Segoe UI Italic", 9), wraplength=450, justify="left")
        self.path_lbl.pack(anchor="w", pady=(0, 15))

        # Action Buttons
        self.btn_frame = tk.Frame(left_col, bg="#282830")
        self.btn_frame.pack(fill="x", side="bottom")

        self.process_btn = tk.Label(
            self.btn_frame,
            text="RUN OCR & VERIFICATION",
            fg="#1e1e24",
            bg="#3f3f46",  # Greyed out until uploaded
            font=("Segoe UI Bold", 10),
            padx=20,
            pady=10,
            cursor="arrow"
        )
        self.process_btn.pack(side="left")

        # Right Column
        right_col = tk.Frame(content_frame, bg="#282830", padx=25, pady=25, highlightbackground="#3f3f46", highlightthickness=1, width=340)
        right_col.pack(side="right", fill="both")

        hints_title = tk.Label(right_col, text="SCANNING BEST PRACTICES", fg="#00bcd4", bg="#282830", font=("Segoe UI Bold", 11))
        hints_title.pack(anchor="w", pady=(0, 15))

        hints = [
            ("📄 Image Format", "Ensure the document is in focus, aligned horizontally, and has no perspective warp."),
            ("💡 Lighting & Glares", "Avoid direct flash reflections or glares that obscure text (e.g. over plastic holograms)."),
            ("📷 High Resolution", "For mobile photos, use macro mode. Resolution should be at least 150 DPI for optimal OCR."),
            ("🛂 Passport MRZ Lines", "Ensure the 2 lines at the bottom of the Passport page 1 are completely visible and clean."),
            ("💳 Aadhaar Hindi Fields", "Aadhaar cards have bilingual details. Standard Tesseract models process both eng and hin.")
        ]

        for title, desc in hints:
            h_frame = tk.Frame(right_col, bg="#282830", pady=8)
            h_frame.pack(fill="x")
            
            h_title = tk.Label(h_frame, text=title, fg="#ffffff", bg="#282830", font=("Segoe UI Semibold", 10))
            h_title.pack(anchor="w")

            h_desc = tk.Label(h_frame, text=desc, fg="#a0aec0", bg="#282830", font=("Segoe UI", 9), wraplength=290, justify="left")
            h_desc.pack(anchor="w", pady=(2, 0))

    def _browse_file(self):
        file_types = [
            ("Supported Documents", "*.jpg;*.jpeg;*.png;*.tiff;*.tif;*.pdf"),
            ("Images", "*.jpg;*.jpeg;*.png;*.tiff;*.tif"),
            ("PDF Documents", "*.pdf"),
            ("All Files", "*.*")
        ]
        path = filedialog.askopenfilename(title="Select Document File", filetypes=file_types)
        if not path:
            return

        self.selected_file_path = path
        self.path_lbl.configure(text=f"Selected: {path}", fg="#00bcd4", font=("Segoe UI Semibold", 9))

        # Show thumbnail preview
        try:
            # 1. Load image (handles PDF pages automatically)
            img = ImageLoader.load(path)
            
            # 2. Downscale for preview box
            preview_img = resize_keep_aspect(img, max_width=380, max_height=180)
            
            # 3. Convert to ImageTk
            self.preview_image_ref = ImageTk.PhotoImage(preview_img)

            # 4. Swap upload icon/text with the image preview
            self.upload_icon_lbl.pack_forget()
            self.upload_text_lbl.pack_forget()
            
            # Set image to a new label inside upload box
            for child in self.upload_box.winfo_children():
                if child not in [self.upload_icon_lbl, self.upload_text_lbl]:
                    child.destroy()
                    
            preview_label = tk.Label(self.upload_box, image=self.preview_image_ref, bg="#1e1e24")
            preview_label.pack(expand=True, fill="both")
            
            # Bind click events to preview label too so users can re-browse
            preview_label.bind("<Button-1>", lambda e: self._browse_file())

            # Enable the process button
            self.process_btn.configure(bg="#00bcd4", fg="#1e1e24", cursor="hand2")
            self.process_btn.bind("<Button-1>", lambda e: self._start_verification())

        except Exception as e:
            messagebox.showerror("Loading Error", f"Could not generate preview for document:\n{e}")
            self.selected_file_path = None
            self.path_lbl.configure(text="No file selected", fg="#a0aec0", font=("Segoe UI Italic", 9))

    def _prepopulate_packet(self, packet):
        if not packet or not packet.image_path:
            return
        self.selected_file_path = packet.image_path
        self.path_lbl.configure(text=f"Selected: {packet.image_path}", fg="#00bcd4", font=("Segoe UI Semibold", 9))
        try:
            img = ImageLoader.load(packet.image_path)
            preview_img = resize_keep_aspect(img, max_width=380, max_height=180)
            self.preview_image_ref = ImageTk.PhotoImage(preview_img)
            self.upload_icon_lbl.pack_forget()
            self.upload_text_lbl.pack_forget()
            for child in self.upload_box.winfo_children():
                if child not in [self.upload_icon_lbl, self.upload_text_lbl]:
                    child.destroy()
            preview_label = tk.Label(self.upload_box, image=self.preview_image_ref, bg="#1e1e24")
            preview_label.pack(expand=True, fill="both")
            preview_label.bind("<Button-1>", lambda e: self._browse_file())
            self.process_btn.configure(bg="#00bcd4", fg="#1e1e24", cursor="hand2")
            self.process_btn.bind("<Button-1>", lambda e: self._start_verification())
        except Exception:
            pass

    def _start_verification(self):
        if not self.selected_file_path:
            return
        self.controller.start_processing(self.selected_file_path)
