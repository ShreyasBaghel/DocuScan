import tkinter as tk
from tkinter import ttk

class ConfidenceBadge(tk.Frame):
    def __init__(self, parent, confidence: float, **kwargs):
        """
        confidence: Float value between 0.0 and 1.0.
        """
        super().__init__(parent, bg="#282830", **kwargs)
        self.confidence = confidence
        self._init_ui()

    def _init_ui(self):
        conf_pct = self.confidence * 100
        
        # Select color palette based on confidence value
        if conf_pct >= 80:
            bg_color = "#10b981"  # Emerald Green
            fg_color = "#ffffff"
        elif conf_pct >= 60:
            bg_color = "#f59e0b"  # Amber Orange
            fg_color = "#1e1e24"
        else:
            bg_color = "#ef4444"  # Crimson Red
            fg_color = "#ffffff"

        # Frame container for padding (simulates rounded border-radius card)
        inner_frame = tk.Frame(
            self, 
            bg=bg_color, 
            highlightbackground="#3f3f46", 
            highlightthickness=1, 
            padx=10, 
            pady=4
        )
        inner_frame.pack(fill="both", expand=True)

        label_text = f"Confidence: {conf_pct:.1f}%"
        label = tk.Label(
            inner_frame, 
            text=label_text, 
            bg=bg_color, 
            fg=fg_color, 
            font=("Segoe UI Semibold", 10)
        )
        label.pack()
        
    def update_confidence(self, new_confidence: float):
        self.confidence = new_confidence
        for widget in self.winfo_children():
            widget.destroy()
        self._init_ui()
