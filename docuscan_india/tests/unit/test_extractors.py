import pytest
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.document_packet import DocumentType
from extractors.extractor_registry import ExtractorRegistry
from extractors.aadhaar_extractor import AadhaarExtractor
from extractors.pan_extractor import PANExtractor
from extractors.passport_extractor import PassportExtractor
from extractors.dl_extractor import DLExtractor

def test_extractor_registry():
    ExtractorRegistry.initialize()
    
    ext = ExtractorRegistry.get_extractor(DocumentType.AADHAAR)
    assert isinstance(ext, AadhaarExtractor)
    
    ext = ExtractorRegistry.get_extractor(DocumentType.PAN)
    assert isinstance(ext, PANExtractor)
    
    ext = ExtractorRegistry.get_extractor(DocumentType.PASSPORT)
    assert isinstance(ext, PassportExtractor)
    
    ext = ExtractorRegistry.get_extractor(DocumentType.DRIVING_LICENCE)
    assert isinstance(ext, DLExtractor)

def test_aadhaar_extractor():
    ext = AadhaarExtractor()
    text = (
        "GOVERNMENT OF INDIA\n"
        "SHREYANSH KUMAR\n"
        "DOB: 15/08/1990\n"
        "MALE\n"
        "1234 5678 9012\n"
    )
    # Mock word map to avoid errors in bounding boxes
    word_map = [
        {"text": "1234", "left": 10, "top": 50, "width": 10, "height": 5, "conf": 0.90},
        {"text": "5678", "left": 25, "top": 50, "width": 10, "height": 5, "conf": 0.90},
        {"text": "9012", "left": 40, "top": 50, "width": 10, "height": 5, "conf": 0.90},
        {"text": "SHREYANSH", "left": 10, "top": 20, "width": 20, "height": 5, "conf": 0.85},
        {"text": "KUMAR", "left": 35, "top": 20, "width": 15, "height": 5, "conf": 0.85},
        {"text": "MALE", "left": 10, "top": 40, "width": 10, "height": 5, "conf": 0.95},
        {"text": "DOB:", "left": 10, "top": 30, "width": 10, "height": 5, "conf": 0.90},
        {"text": "15/08/1990", "left": 25, "top": 30, "width": 20, "height": 5, "conf": 0.90}
    ]
    res = ext.extract(text, word_map)
    
    assert res["aadhaar_number"].value == "123456789012"
    assert res["dob"].value == "1990-08-15"
    assert res["gender"].value == "MALE"
    assert res["name"].value == "SHREYANSH KUMAR"

def test_pan_extractor():
    ext = PANExtractor()
    text = (
        "INCOME TAX DEPARTMENT\n"
        "GOVT. OF INDIA\n"
        "NAME\n"
        "RAJESH SHARMA\n"
        "FATHER'S NAME\n"
        "OM PRAKASH SHARMA\n"
        "DATE OF BIRTH\n"
        "01-01-1985\n"
        "PAN CARD NUMBER\n"
        "ABCDE1234F\n"
    )
    word_map = [
        {"text": "ABCDE1234F", "left": 10, "top": 100, "width": 30, "height": 5, "conf": 0.98},
        {"text": "RAJESH", "left": 10, "top": 40, "width": 15, "height": 5, "conf": 0.90},
        {"text": "SHARMA", "left": 30, "top": 40, "width": 15, "height": 5, "conf": 0.90},
        {"text": "OM", "left": 10, "top": 60, "width": 10, "height": 5, "conf": 0.90},
        {"text": "PRAKASH", "left": 25, "top": 60, "width": 15, "height": 5, "conf": 0.90},
        {"text": "SHARMA", "left": 45, "top": 60, "width": 15, "height": 5, "conf": 0.90},
        {"text": "01-01-1985", "left": 10, "top": 80, "width": 25, "height": 5, "conf": 0.92}
    ]
    res = ext.extract(text, word_map)
    
    assert res["pan_number"].value == "ABCDE1234F"
    assert res["name"].value == "RAJESH SHARMA"
    assert res["father_name"].value == "OM PRAKASH SHARMA"
    assert res["dob"].value == "1985-01-01"

def test_passport_extractor():
    ext = PassportExtractor()
    # Mock MRZ strings for Indian Passport
    # Line 1: P<IND[SURNAME]<<[GIVEN_NAMES]
    # Line 2: [PASS_NO] < [CHECK] [NAT] [DOB] [CHECK] [SEX] [EXPIRY] [CHECK] ...
    # Z1234567 is passport no, IND is nationality, 901231 is DOB (31 Dec 1990), M is male, 301231 is expiry (31 Dec 2030)
    text = (
        "PASSPORT\n"
        "P<INDSHARMA<<RAJESH<<<<<<<<<<<<<<<<<<<<<<<<<\n"
        "Z1234567<8IND9012312M3012315<<<<<<<<<<<<<<02\n"
    )
    res = ext.extract(text, [])
    
    assert res["passport_number"].value == "Z1234567"
    assert res["nationality"].value == "IND"
    assert res["dob"].value == "1990-12-31"
    assert res["expiry"].value == "2030-12-31"
    assert res["name"].value == "RAJESH SHARMA"
    assert res["mrz_line1"].value == "P<INDSHARMA<<RAJESH<<<<<<<<<<<<<<<<<<<<<<<<<"
    assert res["mrz_line2"].value == "Z1234567<8IND9012312M3012315<<<<<<<<<<<<<<02"

def test_passport_extractor_fallback_and_cleaning():
    ext = PassportExtractor()
    # Mock text similar to indianpp1 with:
    # - MRZ line 1 having P<<SPECIMEN (missing country code) and trailing noise (S66SKSSE)
    # - MRZ line 2 having invalid expiry 230000 (meaning invalid month/day 00)
    # - Visual text having:
    #   * Place of Birth: MUMBAI, MAHARASHTRA
    #   * Place of Issue: BANGALORE
    #   * Date of Expiry: 01/01/2023 (valid expiry fallback)
    text = (
        "Place of Birth\n"
        "MUMBAI, MAHARASHTRA\n"
        "Place of issue\n"
        "BANGALORE\n"
        "Date of Expiry\n"
        "01/01/2023\n"
        "P<<SPECIMEN<<KUMAR<G<<<<<<<<<<<S66SKSSE<<<<<\n"
        "Z9999999<0IND8505246M2300000<<<<<<<<<<<<<<S4\n"
    )
    res = ext.extract(text, [])
    
    # Assert MRZ prefix robust skip + noise filtering works
    assert res["name"].value == "KUMAR G SPECIMEN"
    
    # Assert passport number is extracted
    assert res["passport_number"].value == "Z9999999"
    
    # Assert DOB is correct
    assert res["dob"].value == "1985-05-24"
    
    # Assert Sex is correct
    assert res["sex"].value == "M"
    
    # Assert Expiry falls back to visual date because 230000 in MRZ is invalid
    assert res["expiry"].value == "2023-01-01"
    
    # Assert Places are extracted from visual zone
    assert res["place_of_birth"].value == "MUMBAI, MAHARASHTRA"
    assert res["place_of_issue"].value == "BANGALORE"

def test_dl_extractor():
    ext = DLExtractor()
    text = (
        "DRIVING LICENCE\n"
        "NAME: KARTIK KAPOOR\n"
        "DL NO: MH-03-2014-0123456\n"
        "DOB: 10/10/1992\n"
        "VALID TILL: 09/10/2034\n"
        "VEHICLE CLASSES: MCWG, LMV\n"
    )
    word_map = [
        {"text": "MH-03-2014-0123456", "left": 10, "top": 40, "width": 40, "height": 5, "conf": 0.95},
        {"text": "KARTIK", "left": 10, "top": 20, "width": 15, "height": 5, "conf": 0.88},
        {"text": "KAPOOR", "left": 30, "top": 20, "width": 15, "height": 5, "conf": 0.88},
        {"text": "10/10/1992", "left": 10, "top": 60, "width": 20, "height": 5, "conf": 0.90},
        {"text": "09/10/2034", "left": 10, "top": 80, "width": 20, "height": 5, "conf": 0.88},
        {"text": "MCWG", "left": 10, "top": 100, "width": 10, "height": 5, "conf": 0.90},
        {"text": "LMV", "left": 25, "top": 100, "width": 10, "height": 5, "conf": 0.90}
    ]
    res = ext.extract(text, word_map)
    
    assert res["dl_number"].value == "MH0320140123456"
    assert res["dob"].value == "1992-10-10"
    assert res["validity"].value == "2034-10-09"
    assert res["name"].value == "KARTIK KAPOOR"
    assert "MCWG" in res["vehicle_class"].value
    assert "LMV" in res["vehicle_class"].value

def test_dl_extractor_address_filtering():
    ext = DLExtractor()
    text = (
        "DRIVING LICENCE\n"
        "DL NO: MH-03-2014-0123456\n"
        "Name: NIVRUTTI BODAKE\n"
        "S/D/W of: KRUSHNA\n"
        "Add: A/P MAZGAON\n"
        "TAL: BHOR\n"
        "DIST: PUNE\n"
    )
    word_map = [
        {"text": "MH-03-2014-0123456", "left": 10, "top": 40, "width": 40, "height": 5, "conf": 0.95},
        {"text": "NIVRUTTI", "left": 10, "top": 60, "width": 15, "height": 5, "conf": 0.90},
        {"text": "BODAKE", "left": 30, "top": 60, "width": 15, "height": 5, "conf": 0.90},
        {"text": "KRUSHNA", "left": 10, "top": 80, "width": 20, "height": 5, "conf": 0.90},
        {"text": "A/P", "left": 10, "top": 100, "width": 10, "height": 5, "conf": 0.88},
        {"text": "MAZGAON", "left": 25, "top": 100, "width": 20, "height": 5, "conf": 0.88},
        {"text": "BHOR", "left": 25, "top": 120, "width": 20, "height": 5, "conf": 0.88},
        {"text": "PUNE", "left": 25, "top": 140, "width": 20, "height": 5, "conf": 0.88}
    ]
    res = ext.extract(text, word_map)
    assert res["name"].value == "NIVRUTTI BODAKE"

def test_dl_extractor_multi_line_name():
    ext = DLExtractor()
    text = (
        "DRIVING LICENCE\n"
        "Name: NIVRUTTI\n"
        "BODAKE\n"
        "DL NO: MH-03-2014-0123456\n"
        "Add: A/P MAZGAON\n"
    )
    word_map = [
        {"text": "MH-03-2014-0123456", "left": 10, "top": 80, "width": 40, "height": 5, "conf": 0.95},
        {"text": "NIVRUTTI", "left": 10, "top": 40, "width": 15, "height": 5, "conf": 0.90},
        {"text": "BODAKE", "left": 10, "top": 60, "width": 15, "height": 5, "conf": 0.90},
        {"text": "MAZGAON", "left": 25, "top": 100, "width": 20, "height": 5, "conf": 0.88}
    ]
    res = ext.extract(text, word_map)
    assert res["name"].value == "NIVRUTTI BODAKE"

