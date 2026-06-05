import tkinter as tk
import math

class RiskGauge(tk.Canvas):
    def __init__(self, parent, score: int = 0, width: int = 240, height: int = 180, **kwargs):
        super().__init__(parent, width=width, height=height, bg="#1e1e24", highlightthickness=0, **kwargs)
        self.score = score
        self.width = width
        self.height = height
        
        self.center_x = width // 2
        self.center_y = int(height * 0.75)
        self.radius = int(width * 0.4)
        
        self.draw_gauge(0)
        if score > 0:
            self.animate_to_score(0, score)

    def draw_gauge(self, score_val: int):
        self.delete("all")
        
        # 1. Draw outer background arc (grey track)
        # Bounding box for arc
        x0 = self.center_x - self.radius
        y0 = self.center_y - self.radius
        x1 = self.center_x + self.radius
        y1 = self.center_y + self.radius
        
        # Arc from 180 degrees (left) to 0 degrees (right)
        self.create_arc(x0, y0, x1, y1, start=0, extent=180, outline="#3f3f46", width=18, style="arc")
        
        # 2. Draw colored foreground arc based on score
        # The angle is proportional to the score (from 0 to 100 corresponds to 180 to 0)
        # extent = -(score_val / 100.0) * 180.0
        # In Tkinter, positive extent goes counter-clockwise, negative goes clockwise.
        # Since we want to start from the left (180 deg) and go clockwise to the right,
        # we start at 180 and draw a clockwise arc (negative extent).
        extent = -(score_val / 100.0) * 180.0
        
        # Select color based on current value
        if score_val <= 20:
            color = "#10b981"  # Emerald
            tier_text = "LOW RISK"
        elif score_val <= 50:
            color = "#f59e0b"  # Amber
            tier_text = "MODERATE"
        elif score_val <= 75:
            color = "#f97316"  # Orange
            tier_text = "HIGH RISK"
        else:
            color = "#ef4444"  # Crimson
            tier_text = "CRITICAL"

        if score_val > 0:
            self.create_arc(x0, y0, x1, y1, start=180, extent=extent, outline=color, width=18, style="arc")
            
        # 3. Draw needle
        # Convert score to angle in radians.
        # Score 0 -> Left (180 deg = pi), Score 100 -> Right (0 deg = 0)
        angle_deg = 180 - (score_val / 100.0) * 180
        angle_rad = math.radians(angle_deg)
        
        # Needle endpoint
        needle_length = self.radius - 8
        nx = self.center_x + needle_length * math.cos(angle_rad)
        ny = self.center_y - needle_length * math.sin(angle_rad)
        
        # Draw needle line
        self.create_line(self.center_x, self.center_y, nx, ny, fill="#ffffff", width=4, arrow="last", arrowshape=(10,12,4))
        
        # Draw pivot circle
        self.create_oval(self.center_x - 8, self.center_y - 8, self.center_x + 8, self.center_y + 8, fill="#ffffff", outline="#1e1e24", width=2)
        
        # 4. Score text and tier text
        self.create_text(self.center_x, self.center_y + 20, text=f"{int(score_val)} / 100", fill="#ffffff", font=("Segoe UI Bold", 18))
        self.create_text(self.center_x, self.center_y + 40, text=tier_text, fill=color, font=("Segoe UI Semibold", 10))

    def animate_to_score(self, current: float, target: int):
        """Micro-animation that sweeps the dial needle from 'current' to 'target'."""
        if current < target:
            # Increments step proportional to remaining distance for easing effect
            step = max(1.0, (target - current) * 0.15)
            next_val = min(target, current + step)
            self.draw_gauge(next_val)
            self.after(15, self.animate_to_score, next_val, target)
        else:
            self.draw_gauge(target)
            
    def set_score(self, score: int):
        self.score = score
        self.animate_to_score(0, score)
