import pytest
import os
import sys
import tkinter as tk
from unittest.mock import MagicMock

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.document_packet import DocumentType, FieldResult, DocumentPacket
from extractors.base_extractor import BaseExtractor
from extractors.aadhaar_extractor import AadhaarExtractor
from extractors.passport_extractor import PassportExtractor
from extractors.dl_extractor import DLExtractor
from ui.screens.ocr_result_screen import OCRResultScreen

# 1. Test Tolerant Highlight Matching
def test_tolerant_highlight_matching():
    root = tk.Tk()
    root.withdraw()
    controller = MagicMock()
    screen = OCRResultScreen(root, controller)

    # Identical
    assert screen._is_highlight_match_tolerant("123456789012", "123456789012") is True
    # Spacing and normalization
    assert screen._is_highlight_match_tolerant("123456789012", "1234 5678 9012") is True
    # Punctuation/casing
    assert screen._is_highlight_match_tolerant("KARTIK KAPOOR", "Kartik, Kapoor.") is True
    # Subset matching
    assert screen._is_highlight_match_tolerant("ABHISHEK JITENDRABHAI SOLANKI", "ABHISHEK JITENDRABHAI") is True
    # False cases
    assert screen._is_highlight_match_tolerant("KARTIK KAPOOR", "SDMW") is False
    assert screen._is_highlight_match_tolerant("123456789012", "987654321098") is False
    
    root.destroy()

# 2. Test Multi-Box Merging with details in BaseExtractor
def test_merge_bounding_boxes_with_details():
    class DummyExtractor(BaseExtractor):
        def extract(self, raw_text, word_map):
            return {}

    ext = DummyExtractor()
    word_map = [
        {"text": "2086", "left": 10, "top": 50, "width": 40, "height": 20, "conf": 0.95},
        {"text": "8322", "left": 60, "top": 50, "width": 40, "height": 20, "conf": 0.90},
        {"text": "2522", "left": 110, "top": 50, "width": 40, "height": 20, "conf": 0.85}
    ]

    # Test exact concatenated target match
    bbox, constituent = ext.merge_bounding_boxes_with_details("208683222522", word_map)
    assert bbox is not None
    assert bbox['x'] == 10
    assert bbox['w'] == 140 # (110 + 40) - 10
    assert len(constituent) == 3
    assert constituent[0]['text'] == "2086"
    assert constituent[2]['text'] == "2522"

# 3. Test Aadhaar fragmented number reconstruction
def test_aadhaar_fragmented_reconstruction():
    ext = AadhaarExtractor()
    text = "Unique Identification Authority of India\n2086 8322 2522\n"
    word_map = [
        {"text": "2086", "left": 10, "top": 50, "width": 40, "height": 20, "conf": 0.95},
        {"text": "8322", "left": 60, "top": 50, "width": 40, "height": 20, "conf": 0.90},
        {"text": "2522", "left": 110, "top": 50, "width": 40, "height": 20, "conf": 0.85}
    ]
    res = ext.extract(text, word_map)
    
    assert res["aadhaar_number"].value == "208683222522"
    assert res["aadhaar_number"].constituent_boxes is not None
    assert len(res["aadhaar_number"].constituent_boxes) == 3

# 4. Test Passport name candidate ranking and city filtering
def test_passport_name_ranking():
    ext = PassportExtractor()
    text = "REPUBLIC OF INDIA\nPlace of Birth\nAHMEDABAD\nABHISHEK JITENDRABHAI SOLANKI\n"
    word_map = [
        {"text": "AHMEDABAD", "left": 10, "top": 40, "width": 80, "height": 20, "conf": 0.98},
        {"text": "ABHISHEK", "left": 10, "top": 60, "width": 60, "height": 20, "conf": 0.95},
        {"text": "JITENDRABHAI", "left": 80, "top": 60, "width": 90, "height": 20, "conf": 0.95},
        {"text": "SOLANKI", "left": 180, "top": 60, "width": 60, "height": 20, "conf": 0.95}
    ]
    res = ext.extract(text, word_map)
    
    # AHMEDABAD should be filtered out by city check, leaving ABHISHEK JITENDRABHAI SOLANKI
    assert res["name"].value == "ABHISHEK JITENDRABHAI SOLANKI"

# 5. Test DL label filtering and holder name selection
def test_dl_name_label_filtering():
    ext = DLExtractor()
    text = "DRIVING LICENCE\nSDMW\nKARTIK KAPOOR\nDL NO: MH0320140123456\n"
    word_map = [
        {"text": "SDMW", "left": 10, "top": 30, "width": 40, "height": 20, "conf": 0.95},
        {"text": "KARTIK", "left": 10, "top": 50, "width": 60, "height": 20, "conf": 0.90},
        {"text": "KAPOOR", "left": 80, "top": 50, "width": 60, "height": 20, "conf": 0.90},
        {"text": "MH0320140123456", "left": 10, "top": 70, "width": 120, "height": 20, "conf": 0.95}
    ]
    res = ext.extract(text, word_map)
    
    # SDMW must be filtered out as a label, and KARTIK KAPOOR selected
    assert res["name"].value == "KARTIK KAPOOR"

# 6. Test Field-level confidence calculation formula
def test_field_level_confidence():
    ext = AadhaarExtractor()
    text = "2086 8322 2522"
    word_map = [
        {"text": "2086", "left": 10, "top": 50, "width": 40, "height": 20, "conf": 0.95},
        {"text": "8322", "left": 60, "top": 50, "width": 40, "height": 20, "conf": 0.90},
        {"text": "2522", "left": 110, "top": 50, "width": 40, "height": 20, "conf": 0.85}
    ]
    res = ext.extract(text, word_map)
    conf = res["aadhaar_number"].confidence
    
    # Confidence should be a float between 0.0 and 1.0
    assert isinstance(conf, float)
    assert 0.0 <= conf <= 1.0
    # For a perfect matching Aadhaar number with high OCR confidence, the score should be high (>= 0.70)
    assert conf >= 0.70
