import tkinter as tk
from tkinter import ttk, messagebox
import os
import subprocess
from ui.components.field_table import FieldTable
from utils.document_packet import DocumentPacket

class AuditReportScreen(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg="#1e1e24")
        self.controller = controller
        self.packet = controller.current_packet
        self._init_ui()

    def _init_ui(self):
        main_frame = tk.Frame(self, bg="#1e1e24", padx=40, pady=30)
        main_frame.pack(fill="both", expand=True)

        # Title Header
        title_frame = tk.Frame(main_frame, bg="#1e1e24")
        title_frame.pack(fill="x", pady=(0, 20))
        
        title_lbl = tk.Label(title_frame, text="STAGE 7: AUDIT REPORT & EXPORTS", fg="#ffffff", bg="#1e1e24", font=("Segoe UI Bold", 13))
        title_lbl.pack(side="left")

        # Layout Split (Top: Field Table | Bottom: Export & Action Panel)
        top_frame = tk.Frame(main_frame, bg="#282830", padx=20, pady=20, highlightbackground="#3f3f46", highlightthickness=1)
        top_frame.pack(fill="both", expand=True, pady=(0, 15))

        table_title = tk.Label(top_frame, text="EXTRACTED FIELDS & VALIDATION DETAILS", fg="#00bcd4", bg="#282830", font=("Segoe UI Bold", 10))
        table_title.pack(anchor="w", pady=(0, 10))

        # Embed FieldTable component
        self.table = FieldTable(top_frame)
        self.table.pack(fill="both", expand=True)

        if self.packet:
            self.table.populate(self.packet.extracted_fields, self.packet.validation_results)

        # Bottom Frame: Actions and Export Directory Status
        bottom_frame = tk.Frame(main_frame, bg="#282830", padx=20, pady=20, highlightbackground="#3f3f46", highlightthickness=1)
        bottom_frame.pack(fill="x")

        # Left side: Export paths summary
        export_info_frame = tk.Frame(bottom_frame, bg="#282830")
        export_info_frame.pack(side="left", fill="both", expand=True)

        info_title = tk.Label(export_info_frame, text="COMPLETED EXPORTS", fg="#10b981", bg="#282830", font=("Segoe UI Bold", 11))
        info_title.pack(anchor="w")

        # Determine file names
        if self.packet:
            base_name = os.path.splitext(os.path.basename(self.packet.image_path))[0]
            pdf_name = f"{base_name}_audit.pdf"
            json_name = f"{base_name}_audit.json"
        else:
            pdf_name = "N/A"
            json_name = "N/A"

        lbl_pdf_path = tk.Label(export_info_frame, text=f"• PDF Report: {pdf_name}", fg="#a0aec0", bg="#282830", font=("Segoe UI", 9))
        lbl_pdf_path.pack(anchor="w", pady=(5, 0))

        lbl_json_path = tk.Label(export_info_frame, text=f"• JSON Metadata: {json_name}", fg="#a0aec0", bg="#282830", font=("Segoe UI", 9))
        lbl_json_path.pack(anchor="w")

        lbl_log_path = tk.Label(export_info_frame, text="• Persistent SQLite Database: data/db/audit.db", fg="#a0aec0", bg="#282830", font=("Segoe UI", 9))
        lbl_log_path.pack(anchor="w")

        # Right side: Action buttons
        actions_grid = tk.Frame(bottom_frame, bg="#282830")
        actions_grid.pack(side="right", fill="y", anchor="center")

        # Row 1 Buttons
        open_pdf_btn = tk.Label(
            actions_grid, 
            text="OPEN PDF REPORT", 
            fg="#1e1e24", 
            bg="#00bcd4", 
            font=("Segoe UI Bold", 9), 
            padx=15, 
            pady=8, 
            cursor="hand2"
        )
        open_pdf_btn.grid(row=0, column=0, padx=6)
        open_pdf_btn.bind("<Button-1>", lambda e: self._open_pdf())
        open_pdf_btn.bind("<Enter>", lambda e: open_pdf_btn.configure(bg="#00e5ff"))
        open_pdf_btn.bind("<Leave>", lambda e: open_pdf_btn.configure(bg="#00bcd4"))

        open_json_btn = tk.Label(
            actions_grid, 
            text="OPEN JSON FILE", 
            fg="#ffffff", 
            bg="#1e1e24", 
            font=("Segoe UI Semibold", 9), 
            padx=15, 
            pady=8, 
            highlightbackground="#3f3f46", 
            highlightthickness=1, 
            cursor="hand2"
        )
        open_json_btn.grid(row=0, column=1, padx=6)
        open_json_btn.bind("<Button-1>", lambda e: self._open_json())
        open_json_btn.bind("<Enter>", lambda e: open_json_btn.configure(bg="#3f3f46"))
        open_json_btn.bind("<Leave>", lambda e: open_json_btn.configure(bg="#1e1e24"))

        finish_btn = tk.Label(
            actions_grid, 
            text="FINISH & RETURN", 
            fg="#1e1e24", 
            bg="#10b981", 
            font=("Segoe UI Bold", 9), 
            padx=18, 
            pady=8, 
            cursor="hand2"
        )
        finish_btn.grid(row=0, column=2, padx=6)
        finish_btn.bind("<Button-1>", lambda e: self.controller.show_frame("HomeScreen"))
        finish_btn.bind("<Enter>", lambda e: finish_btn.configure(bg="#059669"))
        finish_btn.bind("<Leave>", lambda e: finish_btn.configure(bg="#10b981"))

    def _open_pdf(self):
        if not self.packet or not self.packet.report_path:
            messagebox.showerror("Error", "PDF report path not found.")
            return

        path = self.packet.report_path
        if not os.path.exists(path):
            messagebox.showerror("Error", f"PDF report file does not exist at: {path}")
            return

        try:
            # os.startfile is Windows specific, which matches the user's OS info
            os.startfile(path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open PDF file: {e}")

    def _open_json(self):
        if not self.packet or not self.packet.image_path:
            return
            
        base_name = os.path.splitext(os.path.basename(self.packet.image_path))[0]
        app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        json_path = os.path.join(app_dir, "data", "exports", f"{base_name}_audit.json")

        if not os.path.exists(json_path):
            messagebox.showerror("Error", f"JSON file does not exist at: {json_path}")
            return

        try:
            os.startfile(json_path)
        except Exception as e:
            try:
                # Fallback to notepad
                subprocess.Popen(["notepad.exe", json_path])
            except Exception as ex:
                messagebox.showerror("Error", f"Failed to open JSON file: {ex}")
