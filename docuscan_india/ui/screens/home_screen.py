import tkinter as tk
from tkinter import ttk
from reports.audit_logger import AuditLogger

class HomeScreen(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg="#1e1e24")
        self.controller = controller
        self.audit_logger = AuditLogger()
        self._init_ui()

    def _init_ui(self):
        # 1. Main layout frame
        main_frame = tk.Frame(self, bg="#1e1e24", padx=40, pady=40)
        main_frame.pack(fill="both", expand=True)

        # 2. Header / Logo Banner
        banner_frame = tk.Frame(main_frame, bg="#282830", bd=0, highlightthickness=0)
        banner_frame.pack(fill="x", pady=(0, 25))
        
        # Subtle horizontal cyan gradient strip
        cyan_strip = tk.Frame(banner_frame, bg="#00bcd4", height=4)
        cyan_strip.pack(fill="x", side="top")

        banner_content = tk.Frame(banner_frame, bg="#282830", padx=20, pady=25)
        banner_content.pack(fill="x")

        title_lbl = tk.Label(
            banner_content, 
            text="DOCUSCAN INDIA", 
            fg="#ffffff", 
            bg="#282830", 
            font=("Segoe UI Bold", 26)
        )
        title_lbl.pack(anchor="w")

        subtitle_lbl = tk.Label(
            banner_content, 
            text="Intelligent Offline Government Document Verification System", 
            fg="#00bcd4", 
            bg="#282830", 
            font=("Segoe UI Semibold", 12)
        )
        subtitle_lbl.pack(anchor="w", pady=(5, 0))

        # 3. Statistics Overview Cards
        stats = self._load_statistics()
        
        stats_frame = tk.Frame(main_frame, bg="#1e1e24")
        stats_frame.pack(fill="x", pady=15)
        
        stats_items = [
            ("TOTAL SCANNED", stats["total"], "#ffffff"),
            ("VERIFIED PASS", stats["pass"], "#10b981"),
            ("WARNINGS FOUND", stats["warn"], "#f59e0b"),
            ("CRITICAL REJECT", stats["fail"], "#ef4444")
        ]

        for idx, (label, val, color) in enumerate(stats_items):
            card = tk.Frame(stats_frame, bg="#282830", padx=15, pady=18, highlightbackground="#3f3f46", highlightthickness=1)
            card.pack(side="left", fill="both", expand=True, padx=(0 if idx==0 else 10, 0))

            lbl = tk.Label(card, text=label, fg="#a0aec0", bg="#282830", font=("Segoe UI Bold", 9))
            lbl.pack(anchor="center")

            val_lbl = tk.Label(card, text=str(val), fg=color, bg="#282830", font=("Segoe UI Bold", 24))
            val_lbl.pack(anchor="center", pady=(8, 0))

        # 4. Action Card
        action_card = tk.Frame(main_frame, bg="#282830", padx=30, pady=30, highlightbackground="#3f3f46", highlightthickness=1)
        action_card.pack(fill="both", expand=True, pady=25)

        info_title = tk.Label(
            action_card, 
            text="Ready to Scan and Verify a Document?", 
            fg="#ffffff", 
            bg="#282830", 
            font=("Segoe UI Semibold", 16)
        )
        info_title.pack(anchor="w")

        info_desc = tk.Label(
            action_card, 
            text="DocuScan processes Aadhaar Cards, PAN Cards, Passports, and Driving Licences locally.\n"
                 "The system executes a 7-stage pipeline covering image preprocessing, OCR, layout extraction, "
                 "checksum validation, and advanced fraud score estimation.", 
            fg="#a0aec0", 
            bg="#282830", 
            font=("Segoe UI", 10),
            justify="left"
        )
        info_desc.pack(anchor="w", pady=(10, 25))

        # Start Button with modern hover effect
        start_btn = tk.Label(
            action_card,
            text="START SCANNING NEW DOCUMENT",
            fg="#1e1e24",
            bg="#00bcd4",
            font=("Segoe UI Bold", 11),
            padx=25,
            pady=12,
            cursor="hand2"
        )
        start_btn.pack(anchor="w")
        
        # Hover events
        start_btn.bind("<Enter>", lambda e: start_btn.configure(bg="#00e5ff"))
        start_btn.bind("<Leave>", lambda e: start_btn.configure(bg="#00bcd4"))
        start_btn.bind("<Button-1>", lambda e: self.controller.show_frame("UploadScreen"))

    def _load_statistics(self) -> dict:
        """Queries the SQLite audit logger to extract stats."""
        logs = self.audit_logger.fetch_all()
        stats = {"total": len(logs), "pass": 0, "warn": 0, "fail": 0}
        for log in logs:
            verdict = log.get("verdict", "PASS").upper()
            if verdict == "PASS":
                stats["pass"] += 1
            elif verdict == "WARN":
                stats["warn"] += 1
            else:
                stats["fail"] += 1
        return stats
