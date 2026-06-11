import pytest
from unittest.mock import MagicMock
import tkinter as tk
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.document_packet import DocumentPacket, DocumentType, FieldResult
from ui.screens.ocr_result_screen import OCRResultScreen

def test_get_clean_ocr_display():
    # Initialize a dummy Tk root to allow widget creation
    root = tk.Tk()
    root.withdraw()  # Hide the main window during tests
    
    controller = MagicMock()
    screen = OCRResultScreen(root, controller)
    
    # Create a packet with a mix of classification keywords, extracted fields, and noise
    packet = DocumentPacket(image_path="dummy.jpg")
    packet.document_type = DocumentType.PAN
    packet.classification_confidence = 0.95
    packet.ocr_raw_text = (
        "~~ NOISY HEADER ~~\n"
        "INCOME TAX DEPARTMENT\n"
        "Permanent Account Number Card\n"
        "HXDPB8430A\n"
        "Name: SHREYAS BAGHEL\n"
        "Father's Name: SANJEEV KUMAR\n"
        "Date of Birth: 01/01/2006\n"
        "=== GARBAGE NOISE LINE ===\n"
        "some random garbage text\n"
    )
    packet.extracted_fields = {
        "pan_number": FieldResult(value="HXDPB8430A", raw_text="HXDPB8430A", confidence=0.98),
        "name": FieldResult(value="SHREYAS BAGHEL", raw_text="Name: SHREYAS BAGHEL", confidence=0.90),
        "father_name": FieldResult(value="SANJEEV KUMAR", raw_text="Father's Name: SANJEEV KUMAR", confidence=0.88),
        "dob": FieldResult(value="2006-01-01", raw_text="Date of Birth: 01/01/2006", confidence=0.92)
    }
    
    clean_text = screen._get_clean_ocr_display(packet)
    
    # Check classification info displays properly
    assert "CLASSIFIED DOCUMENT: PAN" in clean_text
    assert "CLASSIFICATION CONFIDENCE: 95.0%" in clean_text
    
    # Check extracted details are rendered
    assert "Pan Number: HXDPB8430A" in clean_text
    assert "Name: SHREYAS BAGHEL" in clean_text
    assert "Father Name: SANJEEV KUMAR" in clean_text
    assert "Dob: 2006-01-01" in clean_text
    
    # Check that garbage/noise lines are successfully removed
    assert "=== GARBAGE NOISE LINE ===" not in clean_text
    assert "~~ NOISY HEADER ~~" not in clean_text
    assert "some random garbage text" not in clean_text
    
    # Check that raw/useful OCR text lines are not retained (ensuring 100% noise-free output)
    assert "INCOME TAX DEPARTMENT" not in clean_text
    assert "Permanent Account Number Card" not in clean_text
    
    root.destroy()
