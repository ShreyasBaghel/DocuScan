import tkinter as tk
from tkinter import ttk
from typing import Dict, List, Any
from utils.document_packet import FieldResult, ValidationResult

class FieldTable(tk.Frame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg="#1e1e24", **kwargs)
        self._init_ui()

    def _init_ui(self):
        # Configure scrollbars and layout
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Style Treeview
        style = ttk.Style()
        style.theme_use("clam")
        
        # Configure modern colors for treeview
        style.configure(
            "Custom.Treeview",
            background="#282830",
            foreground="#ffffff",
            fieldbackground="#282830",
            rowheight=30,
            font=("Segoe UI", 10)
        )
        style.configure(
            "Custom.Treeview.Heading",
            background="#3f3f46",
            foreground="#ffffff",
            font=("Segoe UI Bold", 10),
            borderwidth=0
        )
        style.map(
            "Custom.Treeview.Heading",
            background=[('active', '#52525b')]
        )
        style.map(
            "Custom.Treeview",
            background=[('selected', '#00bcd4')],
            foreground=[('selected', '#1e1e24')]
        )

        # Create Treeview
        self.tree = ttk.Treeview(
            self, 
            columns=("field", "value", "check", "status"), 
            show="headings", 
            style="Custom.Treeview"
        )
        
        self.tree.heading("field", text="Field Name")
        self.tree.heading("value", text="Extracted Value")
        self.tree.heading("check", text="Validation Check Detail")
        self.tree.heading("status", text="Status")

        self.tree.column("field", width=120, anchor="w")
        self.tree.column("value", width=150, anchor="w")
        self.tree.column("check", width=250, anchor="w")
        self.tree.column("status", width=80, anchor="center")

        # Scrollbar
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)

        self.tree.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        # Setup tags for color-coded status rows
        self.tree.tag_configure("pass", foreground="#10b981")  # Emerald
        self.tree.tag_configure("warn", foreground="#f59e0b")  # Amber
        self.tree.tag_configure("fail", foreground="#ef4444")  # Crimson
        self.tree.tag_configure("oddrow", background="#212128")
        self.tree.tag_configure("evenrow", background="#282830")

    def populate(self, fields: Dict[str, FieldResult], validations: List[ValidationResult]):
        """Populates the table with fields and their validation statuses."""
        # Clear existing entries
        for item in self.tree.get_children():
            self.tree.delete(item)

        # Map validation results by field name/check name for convenient lookup
        val_map: Dict[str, ValidationResult] = {}
        for r in validations:
            # Group formats, dates, and consistency checks by field name
            f_name = r.field_name.lower()
            val_map[f_name] = r

        # Add fields
        idx = 0
        for name, field in fields.items():
            disp_name = name.replace('_', ' ').title()
            val_str = field.value
            
            # Find status & validation detail
            status = "PASS"
            detail = "Format validated"
            tag = "pass"
            
            # Check validation map
            matched_val = None
            for v_key, v_val in val_map.items():
                if name.lower() in v_key:
                    matched_val = v_val
                    break
                    
            if matched_val:
                status = matched_val.status
                detail = f"Expected: {matched_val.expected} | Actual: {matched_val.actual}"
                tag = status.lower()
            elif field.value == "NOT_FOUND":
                status = "FAIL"
                detail = "Required field missing"
                tag = "fail"

            row_bg_tag = "oddrow" if idx % 2 == 0 else "evenrow"
            self.tree.insert(
                "", 
                "end", 
                values=(disp_name, val_str, detail, status), 
                tags=(tag, row_bg_tag)
            )
            idx += 1

        # Also add any standalone checks (like overall checksum or cross-field) not directly mapped above
        for r in validations:
            # Check if this rule is already shown in the table
            shown = False
            for child in self.tree.get_children():
                row_val = self.tree.item(child)["values"]
                if r.field_name.replace('_', ' ').title() == row_val[0]:
                    shown = True
                    break
            
            if not shown:
                disp_name = r.field_name.replace('_', ' ').title()
                status = r.status
                detail = f"Expected: {r.expected} | Actual: {r.actual}"
                tag = status.lower()
                
                row_bg_tag = "oddrow" if idx % 2 == 0 else "evenrow"
                self.tree.insert(
                    "", 
                    "end", 
                    values=(disp_name, "N/A", detail, status), 
                    tags=(tag, row_bg_tag)
                )
                idx += 1
