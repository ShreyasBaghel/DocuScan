import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from utils.image_utils import resize_keep_aspect, to_pil

class OCRResultScreen(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg="#1e1e24")
        self.controller = controller
        self.preprocessed_image_ref = None
        self._init_ui()

    def _init_ui(self):
        self.main_frame = tk.Frame(self, bg="#1e1e24", padx=40, pady=30)
        self.main_frame.pack(fill="both", expand=True)

        # 1. Title Header
        title_frame = tk.Frame(self.main_frame, bg="#1e1e24")
        title_frame.pack(fill="x", pady=(0, 20))
        
        self.title_lbl = tk.Label(title_frame, text="STAGE 1 & 2: IMAGE PREPROCESSING & OCR EXTRACTION", fg="#ffffff", bg="#1e1e24", font=("Segoe UI Bold", 13))
        self.title_lbl.pack(side="left")

        # 2. Loading State / Animation Panel (Packed on top when processing)
        self.loading_frame = tk.Frame(self.main_frame, bg="#282830", highlightbackground="#3f3f46", highlightthickness=1)
        self.loading_frame.pack(fill="both", expand=True)

        self.spinner_canvas = tk.Canvas(self.loading_frame, width=60, height=60, bg="#282830", highlightthickness=0)
        self.spinner_canvas.pack(expand=True, pady=(100, 10))
        
        self.loading_lbl = tk.Label(self.loading_frame, text="Processing...", fg="#ffffff", bg="#282830", font=("Segoe UI Semibold", 12))
        self.loading_lbl.pack(expand=True, pady=(0, 150))
        
        self.spinner_angle = 0
        self._animate_spinner()

        # 3. Main Data Panels (Hidden until populate_ocr_data is called)
        self.data_frame = tk.Frame(self.main_frame, bg="#1e1e24")

        # Left Column: Preprocessed Image Preview
        self.left_col = tk.Frame(self.data_frame, bg="#282830", padx=15, pady=15, highlightbackground="#3f3f46", highlightthickness=1)
        self.left_col.pack(side="left", fill="both", expand=True, padx=(0, 10))

        img_title = tk.Label(self.left_col, text="PREPROCESSED IMAGE (Deskewed & Binarized)", fg="#00bcd4", bg="#282830", font=("Segoe UI Bold", 10))
        img_title.pack(anchor="w", pady=(0, 10))

        self.img_lbl = tk.Label(self.left_col, bg="#1e1e24")
        self.img_lbl.pack(fill="both", expand=True)

        # Right Column: Raw OCR Text Box
        self.right_col = tk.Frame(self.data_frame, bg="#282830", padx=15, pady=15, highlightbackground="#3f3f46", highlightthickness=1)
        self.right_col.pack(side="right", fill="both", width=420)

        txt_title = tk.Label(self.right_col, text="RAW EXTRACTED OCR TEXT", fg="#00bcd4", bg="#282830", font=("Segoe UI Bold", 10))
        txt_title.pack(anchor="w", pady=(0, 10))

        # Text Box + Scrollbar
        txt_container = tk.Frame(self.right_col, bg="#282830")
        txt_container.pack(fill="both", expand=True)

        self.ocr_txt = tk.Text(txt_container, bg="#1e1e24", fg="#ffffff", insertbackground="white", font=("Courier New", 10), bd=0, padx=10, pady=10)
        scrollbar = ttk.Scrollbar(txt_container, orient="vertical", command=self.ocr_txt.yview)
        self.ocr_txt.configure(yscrollcommand=scrollbar.set)

        self.ocr_txt.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="ns")

        # 4. Bottom Confidence & Actions Row
        self.bottom_frame = tk.Frame(self.main_frame, bg="#1e1e24")
        self.bottom_frame.pack(fill="x", side="bottom", pady=(15, 0))

        # Confidence Bar
        self.conf_frame = tk.Frame(self.bottom_frame, bg="#1e1e24")
        self.conf_frame.pack(side="left", fill="x", expand=True)

        self.conf_lbl = tk.Label(self.conf_frame, text="Aggregate OCR Confidence: 0.0%", fg="#a0aec0", bg="#1e1e24", font=("Segoe UI Semibold", 9))
        self.conf_lbl.pack(anchor="w")

        # Custom progressbar style
        style = ttk.Style()
        style.configure("Cyan.Horizontal.TProgressbar", background="#00bcd4", troughcolor="#282830")
        
        self.conf_progress = ttk.Progressbar(self.conf_frame, orient="horizontal", style="Cyan.Horizontal.TProgressbar", length=300, mode="determinate")
        self.conf_progress.pack(anchor="w", pady=(4, 0))

        # Action Buttons
        self.btn_frame = tk.Frame(self.bottom_frame, bg="#1e1e24")
        self.btn_frame.pack(side="right")

        self.continue_btn = tk.Label(
            self.btn_frame,
            text="CONTINUE TO CLASSIFICATION →",
            fg="#1e1e24",
            bg="#3f3f46",
            font=("Segoe UI Bold", 10),
            padx=20,
            pady=10,
            cursor="arrow"
        )
        self.continue_btn.pack()

    def set_loading(self, message: str):
        self.loading_lbl.configure(text=message)
        self.data_frame.pack_forget()
        self.loading_frame.pack(fill="both", expand=True)

    def _animate_spinner(self):
        """Animates a spinning cyan arc inside the loading window."""
        if not self.winfo_exists():
            return
        
        self.spinner_canvas.delete("spinner")
        # Draw spinning arc
        self.spinner_canvas.create_arc(
            5, 5, 55, 55, 
            start=self.spinner_angle, 
            extent=90, 
            outline="#00bcd4", 
            width=4, 
            style="arc",
            tags="spinner"
        )
        self.spinner_angle = (self.spinner_angle + 10) % 360
        self.after(20, self._animate_spinner)

    def populate_ocr_data(self, packet):
        """Hides the loader and populates OCR result data in the split frames."""
        self.loading_frame.pack_forget()
        self.data_frame.pack(fill="both", expand=True, pady=10)

        # 1. Populate Preprocessed Image Thumbnail
        if packet.preprocessed_image is not None:
            try:
                # Resize to fit left panel
                cv_img = packet.preprocessed_image
                preview_img = resize_keep_aspect(cv_img, max_width=450, max_height=380)
                pil_img = to_pil(preview_img)
                self.preprocessed_image_ref = ImageTk.PhotoImage(pil_img)
                self.img_lbl.configure(image=self.preprocessed_image_ref)
            except Exception as e:
                self.img_lbl.configure(text=f"Failed to show image preview: {e}", fg="#ef4444")
        else:
            self.img_lbl.configure(text="No preprocessed image data", fg="#ef4444")

        # 2. Populate Text Box
        self.ocr_txt.delete("1.0", tk.END)
        self.ocr_txt.insert(tk.END, packet.ocr_raw_text)

        # 3. Update Confidence Score
        conf_val = packet.ocr_confidence
        self.conf_lbl.configure(text=f"Aggregate OCR Confidence: {conf_val*100:.1f}%")
        self.conf_progress['value'] = conf_val * 100

        # Color confidence text appropriately
        if conf_val >= 0.80:
            self.conf_lbl.configure(fg="#10b981")
        elif conf_val >= 0.60:
            self.conf_lbl.configure(fg="#f59e0b")
        else:
            self.conf_lbl.configure(fg="#ef4444")

        # 4. Enable Continue Button
        self.continue_btn.configure(bg="#00bcd4", fg="#1e1e24", cursor="hand2")
        # Bind transition to Stage 3 / Classification
        self.continue_btn.bind("<Button-1>", lambda e: self._go_to_classification())

    def _go_to_classification(self):
        packet = self.controller.current_packet
        if packet:
            # Trigger dynamic flow for Stage 3/4
            # We first run the classifier on the packet
            # If the classifier predicts UNKNOWN, we pause on ClassificationScreen to ask user
            # Otherwise we directly show ClassificationScreen with the automatic prediction
            if packet.document_type == "UNKNOWN" or packet.document_type == tk.StringVar(): 
                # wait, let's look at controller logic
                pass
            
            # Transition to Classification display screen
            self.controller.show_frame("ClassificationScreen")
            self.controller.current_frame.populate_initial_classification(packet)
