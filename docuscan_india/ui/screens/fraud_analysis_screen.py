import tkinter as tk
from tkinter import ttk
from ui.components.risk_gauge import RiskGauge
from utils.document_packet import DocumentPacket

class FraudAnalysisScreen(tk.Frame):
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
        
        title_lbl = tk.Label(title_frame, text="STAGE 6: FRAUD DETECTION & RISK SCORE ANALYSIS", fg="#ffffff", bg="#1e1e24", font=("Segoe UI Bold", 13))
        title_lbl.pack(side="left")

        # Main horizontal split layout (Left: Risk Gauge Dial | Right: Fraud Signal List)
        content_frame = tk.Frame(main_frame, bg="#1e1e24")
        content_frame.pack(fill="both", expand=True)

        # Left Column: Risk Dial & Verdict Card
        left_col = tk.Frame(content_frame, bg="#282830", padx=25, pady=25, highlightbackground="#3f3f46", highlightthickness=1, width=340)
        left_col.pack(side="left", fill="both", padx=(0, 15))

        dial_title = tk.Label(left_col, text="AGGREGATED RISK METER", fg="#00bcd4", bg="#282830", font=("Segoe UI Bold", 10))
        dial_title.pack(anchor="w", pady=(0, 10))

        # Risk Score Dial
        score = self.packet.fraud_risk_score if self.packet else 0
        self.gauge = RiskGauge(left_col, score=score)
        self.gauge.pack(anchor="center", pady=10)

        # Color coded verdict card
        verdict_card = tk.Frame(left_col, bg="#1e1e24", padx=15, pady=15, highlightthickness=0)
        verdict_card.pack(fill="x", side="bottom", pady=(15, 0))

        # Verdict logic
        from utils.score_formatter import ScoreFormatter
        auth_score = self.packet.authenticity_score if self.packet else 0
        ext_reliability = self.packet.extraction_reliability if self.packet else 0
        
        if self.packet:
            v_title, v_desc = ScoreFormatter.get_verdict(auth_score, score)
        else:
            v_title, v_desc = "Needs Manual Review", "No document processed."
            
        if v_title == "Genuine":
            v_color = "#10b981"
        elif v_title == "Suspicious":
            v_color = "#ef4444"
        else:
            v_color = "#f59e0b"

        lbl_v_title = tk.Label(verdict_card, text=f"VERDICT: {v_title.upper()}", fg=v_color, bg="#1e1e24", font=("Segoe UI Bold", 11))
        lbl_v_title.pack(anchor="w")

        lbl_auth_score = tk.Label(verdict_card, text=f"Authenticity Score: {auth_score}/100", fg="#ffffff", bg="#1e1e24", font=("Segoe UI Semibold", 9))
        lbl_auth_score.pack(anchor="w", pady=(4, 0))

        lbl_ext_score = tk.Label(verdict_card, text=f"Extraction Reliability: {ext_reliability}/100", fg="#ffffff", bg="#1e1e24", font=("Segoe UI Semibold", 9))
        lbl_ext_score.pack(anchor="w")

        lbl_v_desc = tk.Label(verdict_card, text=v_desc, fg="#a0aec0", bg="#1e1e24", font=("Segoe UI", 9), wraplength=200, justify="left")
        lbl_v_desc.pack(anchor="w", pady=(6, 0))


        # Right Column: List of logged fraud signals / rules failed
        right_col = tk.Frame(content_frame, bg="#282830", padx=20, pady=20, highlightbackground="#3f3f46", highlightthickness=1)
        right_col.pack(side="right", fill="both", expand=True)

        list_title = tk.Label(right_col, text="VERIFICATION SIGNALS LOG", fg="#00bcd4", bg="#282830", font=("Segoe UI Bold", 10))
        list_title.pack(anchor="w", pady=(0, 10))

        # Scrollable container for fraud items
        container = tk.Frame(right_col, bg="#282830")
        container.pack(fill="both", expand=True)

        scroll_canvas = tk.Canvas(container, bg="#282830", highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=scroll_canvas.yview)
        self.signal_list_frame = tk.Frame(scroll_canvas, bg="#282830")

        self.signal_list_frame.bind(
            "<Configure>",
            lambda e: scroll_canvas.configure(
                scrollregion=scroll_canvas.bbox("all")
            )
        )

        scroll_canvas.create_window((0, 0), window=self.signal_list_frame, anchor="nw", width=420)
        scroll_canvas.configure(yscrollcommand=scrollbar.set)

        scroll_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self._populate_signals()

        # Footer Actions Row
        footer_frame = tk.Frame(main_frame, bg="#1e1e24")
        footer_frame.pack(fill="x", side="bottom", pady=(15, 0))

        # Back Button
        back_btn = tk.Label(
            footer_frame,
            text="← BACK",
            fg="#ffffff",
            bg="#3f3f46",
            font=("Segoe UI Bold", 10),
            padx=20,
            pady=10,
            cursor="hand2"
        )
        back_btn.pack(side="left")
        back_btn.bind("<Button-1>", lambda e: self._go_back())

        next_btn = tk.Label(
            footer_frame,
            text="VIEW AUDIT REPORT & EXPORT →",
            fg="#1e1e24",
            bg="#00bcd4",
            font=("Segoe UI Bold", 10),
            padx=20,
            pady=10,
            cursor="hand2"
        )
        next_btn.pack(side="right")
        next_btn.bind("<Button-1>", lambda e: self.controller.show_frame("AuditReportScreen"))

    def _go_back(self):
        # Go back to Classification screen and populate it
        self.controller.show_frame("ClassificationScreen")
        if self.controller.current_packet:
            self.controller.current_frame.populate_classification_data(self.controller.current_packet)

    def _populate_signals(self):
        # 1. Clear current frame
        for widget in self.signal_list_frame.winfo_children():
            widget.destroy()

        signals = self.packet.fraud_signals if self.packet else []
        
        # Also construct signals for failed checksum validations for listing here
        validation_fails = []
        if self.packet:
            for v in self.packet.validation_results:
                if v.status == "FAIL":
                    validation_fails.append(v)

        if not signals and not validation_fails:
            empty_lbl = tk.Label(
                self.signal_list_frame, 
                text="✓ No security or formatting anomalies flagged.", 
                fg="#10b981", 
                bg="#282830", 
                font=("Segoe UI Semibold", 11)
            )
            empty_lbl.pack(pady=40)
            return

        # List validation fails
        for f_val in validation_fails:
            item = tk.Frame(self.signal_list_frame, bg="#1e1e24", padx=12, pady=10, highlightbackground="#3f3f46", highlightthickness=1)
            item.pack(fill="x", pady=(0, 8), ipadx=5)
            
            header = tk.Frame(item, bg="#1e1e24")
            header.pack(fill="x")
            
            lbl_name = tk.Label(header, text=f"Rule Failure: {f_val.field_name.replace('_', ' ').title()}", fg="#ef4444", bg="#1e1e24", font=("Segoe UI Bold", 10))
            lbl_name.pack(side="left")

            lbl_score = tk.Label(header, text="+20 Risk", fg="#ef4444", bg="#1e1e24", font=("Segoe UI Bold", 10))
            lbl_score.pack(side="right")

            lbl_desc = tk.Label(item, text=f"Expected: {f_val.expected} | Actual: {f_val.actual}", fg="#a0aec0", bg="#1e1e24", font=("Segoe UI", 9), justify="left", wraplength=400)
            lbl_desc.pack(anchor="w", pady=(4, 0))

        # List metadata/layout fraud signals
        for sig in signals:
            item = tk.Frame(self.signal_list_frame, bg="#1e1e24", padx=12, pady=10, highlightbackground="#3f3f46", highlightthickness=1)
            item.pack(fill="x", pady=(0, 8), ipadx=5)
            
            header = tk.Frame(item, bg="#1e1e24")
            header.pack(fill="x")
            
            lbl_name = tk.Label(header, text=sig.name.replace('_', ' ').title(), fg="#ef4444", bg="#1e1e24", font=("Segoe UI Bold", 10))
            lbl_name.pack(side="left")

            lbl_score = tk.Label(header, text=f"+{sig.score} Risk", fg="#ef4444", bg="#1e1e24", font=("Segoe UI Bold", 10))
            lbl_score.pack(side="right")

            lbl_desc = tk.Label(item, text=sig.description, fg="#a0aec0", bg="#1e1e24", font=("Segoe UI", 9), justify="left", wraplength=400)
            lbl_desc.pack(anchor="w", pady=(4, 0))
