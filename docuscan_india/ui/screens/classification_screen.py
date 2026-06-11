import tkinter as tk
from tkinter import ttk
from utils.document_packet import DocumentType, DocumentPacket
from ui.components.confidence_badge import ConfidenceBadge

class ClassificationScreen(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg="#1e1e24")
        self.controller = controller
        self.packet = None
        self._init_ui()

    def _init_ui(self):
        self.main_frame = tk.Frame(self, bg="#1e1e24", padx=40, pady=30)
        self.main_frame.pack(fill="both", expand=True)

        # Header
        self.title_lbl = tk.Label(
            self.main_frame, 
            text="STAGE 3: DOCUMENT CLASSIFICATION", 
            fg="#ffffff", 
            bg="#1e1e24", 
            font=("Segoe UI Bold", 13)
        )
        self.title_lbl.pack(anchor="w", pady=(0, 20))

        # 1. Loading State Frame (Visible during background execution of Stages 4-7)
        self.loading_frame = tk.Frame(self.main_frame, bg="#282830", highlightbackground="#3f3f46", highlightthickness=1)
        self.spinner_canvas = tk.Canvas(self.loading_frame, width=60, height=60, bg="#282830", highlightthickness=0)
        self.loading_lbl = tk.Label(self.loading_frame, text="Verifying...", fg="#ffffff", bg="#282830", font=("Segoe UI Semibold", 12))
        
        self.spinner_angle = 0
        self._animate_spinner()

        # 2. Main content container
        self.content_card = tk.Frame(self.main_frame, bg="#282830", padx=30, pady=30, highlightbackground="#3f3f46", highlightthickness=1)
        self.content_card.pack(fill="both", expand=True)

        # 2a. Auto-Classification View Components
        self.auto_frame = tk.Frame(self.content_card, bg="#282830")
        
        self.detected_type_lbl = tk.Label(self.auto_frame, text="Document Type Detected:", fg="#a0aec0", bg="#282830", font=("Segoe UI Bold", 11))
        self.detected_type_lbl.pack(anchor="w")

        self.doc_name_lbl = tk.Label(self.auto_frame, text="UNKNOWN", fg="#00bcd4", bg="#282830", font=("Segoe UI Bold", 26))
        self.doc_name_lbl.pack(anchor="w", pady=(5, 10))

        # Placeholder for badge
        self.badge_container = tk.Frame(self.auto_frame, bg="#282830")
        self.badge_container.pack(anchor="w", pady=(0, 25))
        self.conf_badge = None

        self.verify_btn = tk.Label(
            self.auto_frame,
            text="CONFIRM & RUN VERIFICATION",
            fg="#1e1e24",
            bg="#00bcd4",
            font=("Segoe UI Bold", 10),
            padx=25,
            pady=11,
            cursor="hand2"
        )
        self.verify_btn.pack(anchor="w", pady=(0, 25))
        self.verify_btn.bind("<Button-1>", lambda e: self._on_confirm_type())

        # Divider
        self.div_lbl = tk.Label(self.auto_frame, text="Not correct? Override and select the correct type below:", fg="#a0aec0", bg="#282830", font=("Segoe UI Italic", 9))
        self.div_lbl.pack(anchor="w", pady=(10, 8))

        # 2b. Manual Selection View Components (Used as override, or main view if UNKNOWN)
        self.manual_frame = tk.Frame(self.content_card, bg="#282830")
        
        self.manual_title_lbl = tk.Label(
            self.manual_frame, 
            text="Please Select Document Type Manually to Continue:", 
            fg="#ffffff", 
            bg="#282830", 
            font=("Segoe UI Semibold", 12)
        )
        self.manual_title_lbl.pack(anchor="w", pady=(0, 15))

        btn_grid = tk.Frame(self.manual_frame, bg="#282830")
        btn_grid.pack(anchor="w")

        # Create 4 type buttons
        self.type_buttons = []
        types = [
            ("Aadhaar Card", DocumentType.AADHAAR),
            ("PAN Card", DocumentType.PAN),
            ("Indian Passport", DocumentType.PASSPORT),
            ("Driving Licence", DocumentType.DRIVING_LICENCE)
        ]
        
        for idx, (label, dtype) in enumerate(types):
            btn = tk.Label(
                btn_grid, 
                text=label, 
                fg="#ffffff", 
                bg="#1e1e24", 
                font=("Segoe UI Semibold", 10), 
                padx=20, 
                pady=12, 
                highlightbackground="#3f3f46", 
                highlightthickness=1, 
                cursor="hand2"
            )
            btn.grid(row=0, column=idx, padx=(0 if idx==0 else 12, 0))
            # Bind click
            btn.bind("<Button-1>", lambda e, dt=dtype: self._on_manual_select(dt))
            btn.bind("<Enter>", lambda e, b=btn: b.configure(bg="#3f3f46"))
            btn.bind("<Leave>", lambda e, b=btn: b.configure(bg="#1e1e24"))
            self.type_buttons.append(btn)

        # 3. Footer Navigation (Visible when stages 4-7 complete)
        self.footer_frame = tk.Frame(self.main_frame, bg="#1e1e24")
        
        # Back Button
        self.back_btn = tk.Label(
            self.footer_frame,
            text="← BACK",
            fg="#ffffff",
            bg="#3f3f46",
            font=("Segoe UI Bold", 10),
            padx=20,
            pady=10,
            cursor="hand2"
        )
        self.back_btn.pack(side="left")
        self.back_btn.bind("<Button-1>", lambda e: self._go_back())

        self.next_btn = tk.Label(
            self.footer_frame,
            text="PROCEED TO FRAUD ANALYSIS →",
            fg="#1e1e24",
            bg="#00bcd4",
            font=("Segoe UI Bold", 10),
            padx=20,
            pady=10,
            cursor="hand2"
        )
        self.next_btn.pack(side="right")
        self.next_btn.bind("<Button-1>", lambda e: self.controller.show_frame("FraudAnalysisScreen"))

    def _go_back(self):
        # Go back to OCR Results Screen and repopulate the current packet details
        self.controller.show_frame("OCRResultScreen")
        if self.controller.current_packet:
            self.controller.current_frame.populate_ocr_data(self.controller.current_packet)

    def populate_initial_classification(self, packet: DocumentPacket):
        """Called when first swapping from the OCR screen."""
        self.packet = packet
        self.footer_frame.pack_forget()
        self.loading_frame.pack_forget()
        self.content_card.pack(fill="both", expand=True)

        # Determine display mode based on document classification
        if packet.document_type == DocumentType.UNKNOWN:
            self.auto_frame.pack_forget()
            self.manual_frame.pack(fill="both", expand=True)
            self.title_lbl.configure(text="STAGE 3: CLASSIFICATION PAUSED (UNKNOWN DOCUMENT)")
        else:
            self.manual_frame.pack_forget()
            self.auto_frame.pack(fill="both", expand=True)
            self.title_lbl.configure(text="STAGE 3: DOCUMENT CLASSIFICATION DETECTED")

            # Update Labels
            doc_labels = {
                DocumentType.AADHAAR: "Aadhaar Card",
                DocumentType.PAN: "PAN Card",
                DocumentType.PASSPORT: "Indian Passport",
                DocumentType.DRIVING_LICENCE: "Driving Licence"
            }
            self.doc_name_lbl.configure(text=doc_labels.get(packet.document_type, "UNKNOWN"))

            # Clear badge and redraw
            for widget in self.badge_container.winfo_children():
                widget.destroy()
            self.conf_badge = ConfidenceBadge(self.badge_container, packet.classification_confidence)
            self.conf_badge.pack()

            # Pack manual override section inside auto-frame bottom
            self.manual_frame.pack(fill="both", expand=True, in_=self.auto_frame, pady=(30, 0))

    def set_loading(self, message: str):
        """Displays full screen spinner during Stages 4-7."""
        self.loading_lbl.configure(text=message)
        self.content_card.pack_forget()
        self.loading_frame.pack(fill="both", expand=True)
        self.spinner_canvas.pack(expand=True, pady=(120, 10))
        self.loading_lbl.pack(expand=True, pady=(0, 150))

    def _animate_spinner(self):
        if not self.winfo_exists():
            return
        self.spinner_canvas.delete("spinner")
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

    def _on_confirm_type(self):
        """User clicked 'Confirm & Run Verification'."""
        self.controller.continue_verification(self.packet.document_type)

    def _on_manual_select(self, dtype: DocumentType):
        """User selected a document type manually."""
        self.controller.continue_verification(dtype)

    def populate_classification_data(self, packet: DocumentPacket):
        """Called by AppController queue poller when background verification completes."""
        self.packet = packet
        self.loading_frame.pack_forget()
        
        # Reload content card and show complete
        self.content_card.pack(fill="both", expand=True)
        self.auto_frame.pack(fill="both", expand=True)
        
        # Update Labels
        doc_labels = {
            DocumentType.AADHAAR: "Aadhaar Card",
            DocumentType.PAN: "PAN Card",
            DocumentType.PASSPORT: "Indian Passport",
            DocumentType.DRIVING_LICENCE: "Driving Licence"
        }
        self.doc_name_lbl.configure(text=doc_labels.get(packet.document_type, "UNKNOWN"))

        # Clear badge and redraw
        for widget in self.badge_container.winfo_children():
            widget.destroy()
        self.conf_badge = ConfidenceBadge(self.badge_container, packet.classification_confidence)
        self.conf_badge.pack()
        
        # Hide manual override panel and confirm buttons since we are done
        self.manual_frame.pack_forget()
        self.verify_btn.pack_forget()
        self.div_lbl.pack_forget()
        
        # Display completed notice
        # Check if already has done label to prevent duplicates
        for child in self.auto_frame.winfo_children():
            if getattr(child, "is_done_frame", False):
                child.destroy()
                
        done_frame = tk.Frame(self.auto_frame, bg="#282830", pady=10)
        done_frame.is_done_frame = True
        done_frame.pack(anchor="w")
        
        lbl_done = tk.Label(done_frame, text="✓ Verification, Validation & Fraud analysis complete!", fg="#10b981", bg="#282830", font=("Segoe UI Semibold", 12))
        lbl_done.pack(side="left")

        # Show next navigation footer
        self.footer_frame.pack(fill="x", side="bottom", pady=(15, 0))
