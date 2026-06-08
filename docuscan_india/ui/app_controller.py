import tkinter as tk
from tkinter import messagebox
import threading
import queue
import os
from typing import Type
from ocr.pipeline import VerificationPipeline
from utils.document_packet import DocumentPacket, DocumentType
from utils.logger import get_logger

logger = get_logger("app_controller")

class AppController(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("DocuScan India — Government Document OCR & Verification")
        self.geometry("1100x750")
        self.configure(bg="#1e1e24")
        self.resizable(True, True)

        # 1. Pipeline State
        self.pipeline = VerificationPipeline()
        self.current_packet = None
        self.queue = queue.Queue()

        # 2. Main Layout Container
        self.container = tk.Frame(self, bg="#1e1e24")
        self.container.pack(fill="both", expand=True)

        self.current_frame = None
        self.show_frame("HomeScreen")

        # 3. Start background queue polling
        self._poll_queue()

    def show_frame(self, screen_class_name: str):
        """Destroys current frame and loads the requested screen frame."""
        logger.info(f"Swapping screen to: {screen_class_name}")
        
        # Late import to prevent circular import loops
        from ui.screens.home_screen import HomeScreen
        from ui.screens.upload_screen import UploadScreen
        from ui.screens.ocr_result_screen import OCRResultScreen
        from ui.screens.classification_screen import ClassificationScreen
        from ui.screens.fraud_analysis_screen import FraudAnalysisScreen
        from ui.screens.audit_report_screen import AuditReportScreen

        screens = {
            "HomeScreen": HomeScreen,
            "UploadScreen": UploadScreen,
            "OCRResultScreen": OCRResultScreen,
            "ClassificationScreen": ClassificationScreen,
            "FraudAnalysisScreen": FraudAnalysisScreen,
            "AuditReportScreen": AuditReportScreen
        }

        if screen_class_name not in screens:
            logger.error(f"Screen class {screen_class_name} not found.")
            return

        if self.current_frame is not None:
            self.current_frame.destroy()

        screen_class = screens[screen_class_name]
        self.current_frame = screen_class(parent=self.container, controller=self)
        self.current_frame.pack(fill="both", expand=True)

    def start_processing(self, image_path: str):
        """Starts the pipeline thread for Stages 1-3 (Image Loading, Preprocessing, OCR, Classification)."""
        self.show_frame("OCRResultScreen")
        # Update loading state in OCR Screen
        self.current_frame.set_loading("Processing image and running OCR...")
        
        thread = threading.Thread(target=self._run_classification_stage, args=(image_path,), daemon=True)
        thread.start()

    def continue_verification(self, doc_type: DocumentType):
        """Updates classification and triggers background thread for Stages 4-7."""
        self.current_packet.document_type = doc_type
        
        # Transition to Classification display screen if not already on it
        from ui.screens.classification_screen import ClassificationScreen
        if not isinstance(self.current_frame, ClassificationScreen):
            self.show_frame("ClassificationScreen")
            
        self.current_frame.set_loading("Running extraction, validation, and fraud checks...")

        thread = threading.Thread(target=self._run_verification_stage, daemon=True)
        thread.start()

    def _run_classification_stage(self, image_path: str):
        """Target for background thread executing Stages 1-3."""
        try:
            packet = self.pipeline.process_classification(image_path)
            self.queue.put(("CLASSIFICATION_DONE", packet))
        except Exception as e:
            logger.error(f"Background classification error: {e}")
            self.queue.put(("ERROR", str(e)))

    def _run_verification_stage(self):
        """Target for background thread executing Stages 4-7."""
        try:
            packet = self.pipeline.process_verification(self.current_packet)
            self.queue.put(("VERIFICATION_DONE", packet))
        except Exception as e:
            logger.error(f"Background verification error: {e}")
            self.queue.put(("ERROR", str(e)))

    def _poll_queue(self):
        """Polls the thread queue for updates to display on the UI thread."""
        try:
            while True:
                msg_type, payload = self.queue.get_nowait()
                logger.info(f"UI Queue message received: {msg_type}")
                
                if msg_type == "CLASSIFICATION_DONE":
                    self.current_packet = payload
                    # Update OCR screen with raw text
                    if hasattr(self.current_frame, "populate_ocr_data"):
                        self.current_frame.populate_ocr_data(payload)
                    
                elif msg_type == "VERIFICATION_DONE":
                    self.current_packet = payload
                    # Once verification is fully done, swap to classification feedback
                    if hasattr(self.current_frame, "populate_classification_data"):
                        self.current_frame.populate_classification_data(payload)
                        
                elif msg_type == "ERROR":
                    messagebox.showerror("Verification Pipeline Error", f"An error occurred in the pipeline:\n{payload}")
                    self.show_frame("UploadScreen")
                
                self.queue.task_done()
        except queue.Empty:
            pass
        finally:
            self.after(100, self._poll_queue)
