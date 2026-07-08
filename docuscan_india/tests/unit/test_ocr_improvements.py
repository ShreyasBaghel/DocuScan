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


# 7. Test OCR Confidence Thresholding and status field
def test_ocr_confidence_thresholding():
    ext = AadhaarExtractor()
    text = "Unique Identification Authority of India\n2086 8322 2522\n"
    # Low confidence words
    word_map = [
        {"text": "2086", "left": 10, "top": 50, "width": 40, "height": 20, "conf": 0.50},
        {"text": "8322", "left": 60, "top": 50, "width": 40, "height": 20, "conf": 0.40},
        {"text": "2522", "left": 110, "top": 50, "width": 40, "height": 20, "conf": 0.35}
    ]
    res = ext.extract(text, word_map)
    # The confidence should be below 0.65 for numeric ID, so status should be 'low_confidence'
    assert res["aadhaar_number"].value == "208683222522"
    assert res["aadhaar_number"].status == "low_confidence"


# 8. Test Neighborhood Scoring Alignment Boost
def test_neighborhood_scoring_alignment():
    ext = DLExtractor()
    # Anchor at line 0, candidate 1 at same line (line 0), candidate 2 at line 2
    raw_text = "DOB: 14/01/1998  OtherText  19/10/2016"
    word_map = [
        {"text": "DOB:", "left": 10, "top": 50, "width": 40, "height": 20, "conf": 0.95},
        {"text": "14/01/1998", "left": 60, "top": 50, "width": 80, "height": 20, "conf": 0.90},
        {"text": "19/10/2016", "left": 60, "top": 150, "width": 80, "height": 20, "conf": 0.95}
    ]
    res = ext.extract(raw_text, word_map)
    # The one on the same line as DOB anchor (14/01/1998) should win due to neighborhood scoring alignment boost
    assert res["dob"].value == "1998-01-14"


# 9. Test OCR Normalization Layer
def test_ocr_normalization_layer():
    # Numeric fields should replace O->0, I->1, S->5, B->8
    assert BaseExtractor.normalize_ocr_text("2086832225S2", True) == "208683222552"
    assert BaseExtractor.normalize_ocr_text("MH032Ol40I23456", True) == "MH0320140123456"
    # Name fields should NOT replace characters
    assert BaseExtractor.normalize_ocr_text("SHREYAS BAGHEL", False) == "SHREYAS BAGHEL"


# 10. Test Name Merging Stops at Field Label
def test_name_merging_stops_at_label():
    ext = DLExtractor()
    raw_text = "Name: BANMEET SINGH\nFather Name: HARPREET SINGH"
    word_map = [
        {"text": "Name:", "left": 10, "top": 30, "width": 40, "height": 20, "conf": 0.95},
        {"text": "BANMEET", "left": 60, "top": 30, "width": 60, "height": 20, "conf": 0.95},
        {"text": "SINGH", "left": 130, "top": 30, "width": 40, "height": 20, "conf": 0.95},
        {"text": "Father", "left": 10, "top": 60, "width": 40, "height": 20, "conf": 0.95},
        {"text": "Name:", "left": 60, "top": 60, "width": 40, "height": 20, "conf": 0.95},
        {"text": "HARPREET", "left": 110, "top": 60, "width": 60, "height": 20, "conf": 0.95},
        {"text": "SINGH", "left": 180, "top": 60, "width": 40, "height": 20, "conf": 0.95}
    ]
    res = ext.extract(raw_text, word_map)
    # The name extractor must not merge "HARPREET SINGH" into the name
    assert res["name"].value == "BANMEET SINGH"


# 11. Test Driving Licence Vehicle Class Exact Matching
def test_dl_vehicle_class_exact_matching():
    ext = DLExtractor()
    raw_text = "TRANSPORT DEPARTMENT DELHI\nAuthorised Vehicles: LMV, MCWG"
    word_map = [
        {"text": "TRANSPORT", "left": 10, "top": 30, "width": 80, "height": 20, "conf": 0.95},
        {"text": "DEPARTMENT", "left": 100, "top": 30, "width": 80, "height": 20, "conf": 0.95},
        {"text": "LMV,", "left": 10, "top": 60, "width": 40, "height": 20, "conf": 0.95},
        {"text": "MCWG", "left": 60, "top": 60, "width": 40, "height": 20, "conf": 0.95}
    ]
    res = ext.extract(raw_text, word_map)
    # It must not match TRANS from TRANSPORT, only LMV, MCWG
    assert "TRANS" not in res["vehicle_class"].value.split(", ")
    assert "LMV" in res["vehicle_class"].value
    assert "MCWG" in res["vehicle_class"].value

# 12. Test passport failing case name extraction (from user issue)
def test_passport_failing_case_extraction():
    ext = PassportExtractor()
    raw_text = (
        "SINT TRAE / REPUBLIC OF INDIA\n"
        "\n"
        "=a 6 Tye ory GS eee\n"
        "\n"
        "e IND wreellal / iINOIAN\n"
        "\n"
        "Per Sena\n"
        "SOLANKI\n"
        "\n"
        "or ow 2) Cawor Meets)\n"
        "ABHISHEK JITENDRABHAI\n"
        "wat Date ot eth |e)\n"
        "21/11/2000 \"\n"
        "\n"
        "gem ST Pree of Gartn\n"
        "VERAVAL,GUJARAT\n"
        "wee = Mace fae\n"
        "\n"
        "4 AHMEDABAD\n"
        "pe eer oe a ed\n"
        "27/06/2023\n"
        "\n"
        "P<INDSOLANKI<<ABHISHEK<JITENDRABHAL KKK <6 KKK\n"
        "\n"
        "VOMRGRSRAINNBINAIANENE5062645068006162 423<36\n"
        "\n"
        "YS401360\n"
        "\n"
        "os. oop? ews & er ere of f ee | oe\n"
        "JITENDRABHAI LAVCHAND SOLANKI\n"
        "ar ee pte 6 Mote\n"
        "\n"
        "SANGITA JITENDRABHAIL SOLANKI\n"
        "\n"
        "Seta es were 1 oom\n"
        "\n"
        "HEAR GEETA SCHOOL, JALARAM SOCIETY\n"
        "VERAVAL,GIR SOMNATH\n"
        "PIN:362265,GUJARAT, INDIA\n"
    )
    word_map = [
        {"text": "SOLANKI", "left": 10, "top": 50, "width": 40, "height": 10, "conf": 0.95},
        {"text": "ABHISHEK", "left": 10, "top": 70, "width": 40, "height": 10, "conf": 0.95},
        {"text": "JITENDRABHAI", "left": 60, "top": 70, "width": 60, "height": 10, "conf": 0.95}
    ]
    res = ext.extract(raw_text, word_map)
    assert res["name"].value == "ABHISHEK JITENDRABHAI SOLANKI"


