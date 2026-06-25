import pytest
import os
import sys
from PIL import Image
from unittest.mock import MagicMock, patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ocr.passport_splitter import PassportSplitter
from utils.document_packet import DocumentPacket, DocumentType, FieldResult

def test_count_mrz_indicators():
    # Bio-data page raw text containing MRZ lines
    mrz_text = (
        "REPUBLIC OF INDIA\n"
        "P<INDSHARMA<<RAJESH<<<<<<<<<<<<<<<<<<<<<<<<<\n"
        "Z1234567<8IND9012312M3012315<<<<<<<<<<<<<<02\n"
    )
    # Address page text without MRZ
    address_text = (
        "FATHER'S NAME: OM PRAKASH SHARMA\n"
        "MOTHER'S NAME: SHANTI SHARMA\n"
        "ADDRESS: 123 STREET, NEW DELHI, INDIA\n"
    )
    
    score_mrz = PassportSplitter.count_mrz_indicators(mrz_text)
    score_address = PassportSplitter.count_mrz_indicators(address_text)
    
    assert score_mrz > 15
    assert score_address < 5

def test_detect_split_vertical():
    # Create a dummy image (800x400)
    img = Image.new("RGB", (800, 400), (255, 255, 255))
    
    # Word map with a clear vertical gap between x=300 and x=500
    word_map = [
        {"text": "Name", "left": 50, "top": 50, "width": 50, "height": 20, "conf": 0.9},
        {"text": "Rajesh", "left": 120, "top": 50, "width": 80, "height": 20, "conf": 0.9},
        {"text": "Address", "left": 550, "top": 60, "width": 80, "height": 20, "conf": 0.9},
        {"text": "Delhi", "left": 650, "top": 60, "width": 60, "height": 20, "conf": 0.9},
    ]
    
    split_type, split_coord = PassportSplitter.detect_split(img, word_map)
    
    assert split_type == "vertical"
    # The split should be around the center W // 2 = 400
    assert 350 <= split_coord <= 450

def test_detect_split_horizontal():
    # Create a dummy image (400x800)
    img = Image.new("RGB", (400, 800), (255, 255, 255))
    
    # Word map with a clear horizontal gap between y=300 and y=500
    word_map = [
        {"text": "Name", "left": 50, "top": 50, "width": 50, "height": 20, "conf": 0.9},
        {"text": "Rajesh", "left": 50, "top": 100, "width": 80, "height": 20, "conf": 0.9},
        {"text": "Address", "left": 50, "top": 600, "width": 80, "height": 20, "conf": 0.9},
        {"text": "Delhi", "left": 50, "top": 650, "width": 60, "height": 20, "conf": 0.9},
    ]
    
    split_type, split_coord = PassportSplitter.detect_split(img, word_map)
    
    assert split_type == "horizontal"
    # The split should be around the center H // 2 = 400
    assert 350 <= split_coord <= 450

def test_detect_no_split():
    # Standard single page size (400x300)
    img = Image.new("RGB", (400, 300), (255, 255, 255))
    
    # Words crossing the center line (x=200)
    word_map = [
        {"text": "REPUBLIC", "left": 50, "top": 50, "width": 300, "height": 20, "conf": 0.9},
        {"text": "OF", "left": 50, "top": 100, "width": 300, "height": 20, "conf": 0.9},
        {"text": "INDIA", "left": 50, "top": 150, "width": 300, "height": 20, "conf": 0.9},
        {"text": "PASSPORT", "left": 50, "top": 200, "width": 300, "height": 20, "conf": 0.9},
        {"text": "CARD", "left": 50, "top": 250, "width": 300, "height": 20, "conf": 0.9},
    ]
    
    split_type, split_coord = PassportSplitter.detect_split(img, word_map)
    assert split_type is None
    assert split_coord is None

@patch("ocr.ocr_engine.OCREngine.extract")
def test_split_and_extract_flow(mock_extract):
    # Setup mock OCR extracts
    # Region A (Page 1) has MRZ lines
    mrz_text = (
        "REPUBLIC OF INDIA\n"
        "P<INDSHARMA<<RAJESH<<<<<<<<<<<<<<<<<<<<<<<<<\n"
        "Z1234567<8IND9012312M3012315<<<<<<<<<<<<<<02\n"
    )
    word_map_A = [
        {"text": "RAJESH", "left": 50, "top": 50, "width": 50, "height": 10, "conf": 0.95},
        {"text": "Z1234567", "left": 50, "top": 100, "width": 60, "height": 10, "conf": 0.95}
    ]
    
    # Region B (Page 2) has address/parents
    address_text = (
        "PLACE OF BIRTH: NEW DELHI\n"
        "PLACE OF ISSUE: DELHI\n"
    )
    word_map_B = [
        {"text": "DELHI", "left": 100, "top": 50, "width": 40, "height": 10, "conf": 0.92}
    ]
    
    # Return in order of execution: A (first), B (second)
    extract_returns = [
        (mrz_text, 0.95, word_map_A),
        (address_text, 0.92, word_map_B)
    ]
    extract_iter = iter(extract_returns)
    def extract_side_effect(img, doc_type="PASSPORT"):
        return next(extract_iter)

    mock_extract.side_effect = extract_side_effect
    
    # Create packet
    packet = DocumentPacket(image_path="dummy.jpg")
    packet.document_type = DocumentType.PASSPORT
    packet.preprocessed_image = Image.new("RGB", (800, 400), (255, 255, 255))
    
    # Word map indicating vertical split at center
    packet.ocr_word_map = [
        {"text": "RAJESH", "left": 50, "top": 50, "width": 50, "height": 10, "conf": 0.95},
        {"text": "DELHI", "left": 500, "top": 50, "width": 40, "height": 10, "conf": 0.92}
    ]
    
    # Mock extractor
    mock_extractor = MagicMock()
    
    # Page 1 extraction results
    fields_1 = {
        "name": FieldResult(value="RAJESH SHARMA", raw_text="RAJESH", confidence=0.95, bounding_box={"x": 50, "y": 50, "w": 50, "h": 10}),
        "passport_number": FieldResult(value="Z1234567", raw_text="Z1234567", confidence=0.95, bounding_box={"x": 50, "y": 100, "w": 60, "h": 10}),
        "nationality": FieldResult(value="IND", raw_text="IND", confidence=0.95, bounding_box=None),
        "dob": FieldResult(value="1990-12-31", raw_text="901231", confidence=0.95, bounding_box=None),
        "place_of_birth": FieldResult(value="NEW DELHI", raw_text="NEW DELHI", confidence=0.95, bounding_box=None)
    }
    
    # Page 2 extraction results
    fields_2 = {
        "place_of_issue": FieldResult(value="DELHI", raw_text="DELHI", confidence=0.92, bounding_box={"x": 100, "y": 50, "w": 40, "h": 10}),
        "place_of_birth": FieldResult(value="NOT_FOUND", raw_text="", confidence=0.0, bounding_box=None)
    }
    
    def extractor_extract_side_effect(text, word_map):
        if "P<IND" in text:
            return fields_1
        else:
            return fields_2
            
    mock_extractor.extract.side_effect = extractor_extract_side_effect
    
    # Run splitter
    from ocr.ocr_engine import OCREngine
    ocr_engine = OCREngine({})
    ocr_engine.extract = mock_extract
    
    split_occurred = PassportSplitter.split_and_extract(packet, ocr_engine, mock_extractor)
    
    assert split_occurred is True
    assert packet.pipeline_metadata["split_occurred"] is True
    assert packet.pipeline_metadata["split_type"] == "vertical"
    
    # Region B offset was (390, 0) due to hybrid split at x=390. So place_of_issue x should be 100 + 390 = 490
    assert packet.extracted_fields["place_of_issue"].bounding_box["x"] == 490
    assert packet.extracted_fields["place_of_issue"].bounding_box["y"] == 50
    
    # Region A offset was (0, 0), so name x remains 50
    assert packet.extracted_fields["name"].bounding_box["x"] == 50
    
    # Check field merge values
    assert packet.extracted_fields["passport_number"].value == "Z1234567"
    assert packet.extracted_fields["place_of_birth"].value == "NEW DELHI"
    assert packet.extracted_fields["place_of_issue"].value == "DELHI"

@patch("pytesseract.image_to_data")
def test_detect_hybrid_split_single_page(mock_image_to_data):
    # Single page layout (400x300), no distinct gaps
    img = Image.new("RGB", (400, 300), (255, 255, 255))
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    # Draw text-like blocks crossing the middle
    draw.rectangle([50, 50, 350, 70], fill=(0, 0, 0))
    draw.rectangle([50, 100, 350, 120], fill=(0, 0, 0))
    draw.rectangle([50, 150, 350, 170], fill=(0, 0, 0))
    
    # Mock image_to_data output
    mock_data = {
        'level': [1, 2, 3, 4, 5, 5, 5],
        'left': [0, 0, 0, 50, 50, 150, 250],
        'top': [0, 0, 0, 50, 50, 50, 50],
        'width': [400, 400, 400, 300, 80, 80, 80],
        'height': [300, 300, 300, 20, 20, 20, 20],
        'conf': [-1, -1, -1, -1, 90, 90, 90],
        'text': ["", "", "", "", "REPUBLIC", "OF", "INDIA"]
    }
    mock_image_to_data.return_value = mock_data
    
    split_type, split_coord = PassportSplitter.detect_hybrid_split(img)
    assert split_type is None
    assert split_coord is None

@patch("pytesseract.image_to_data")
def test_detect_hybrid_split_side_by_side(mock_image_to_data):
    # Two pages side-by-side (800x400), vertical gap around center (400)
    img = Image.new("RGB", (800, 400), (255, 255, 255))
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    # Draw text-like blocks on Left page (Region A) up to 300
    draw.rectangle([50, 50, 300, 350], fill=(0, 0, 0))
    # Draw text-like blocks on Right page (Region B) from 450
    draw.rectangle([450, 50, 750, 350], fill=(0, 0, 0))
    
    # Mock image_to_data output
    mock_data = {
        'level': [1, 2, 3, 4, 5, 5, 5, 5],
        'left': [0, 0, 0, 50, 200, 200, 450, 600],
        'top': [0, 0, 0, 50, 50, 50, 50, 50],
        'width': [800, 800, 800, 700, 100, 100, 100, 100],
        'height': [400, 400, 400, 20, 20, 20, 20, 20],
        'conf': [-1, -1, -1, -1, 90, 90, 90, 90],
        'text': ["", "", "", "", "PAGE", "ONE", "PAGE", "TWO"]
    }
    mock_image_to_data.return_value = mock_data
    
    split_type, split_coord = PassportSplitter.detect_hybrid_split(img)
    assert split_type == "vertical"
    # Vertical gap is [300, 450]. Center is 375.
    assert 370 <= split_coord <= 380

@patch("pytesseract.image_to_data")
def test_detect_hybrid_split_stacked(mock_image_to_data):
    # Two pages stacked or MRZ separated (600x800), horizontal gap around 450
    img = Image.new("RGB", (600, 800), (255, 255, 255))
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    # Draw text-like blocks on Top page (Region A)
    draw.rectangle([50, 290, 550, 400], fill=(0, 0, 0))
    # Draw text-like blocks on Bottom page / MRZ (Region B)
    draw.rectangle([50, 500, 550, 670], fill=(0, 0, 0))
    
    # Mock image_to_data output aligned with drawing:
    mock_data = {
        'level': [1, 2, 3, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
        'left': [0, 0, 0, 50, 50, 200, 50, 200, 50, 200, 50, 200, 50, 200],
        'top': [0, 0, 0, 50, 290, 290, 380, 380, 500, 500, 590, 590, 650, 650],
        'width': [600, 600, 600, 500, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
        'height': [800, 800, 800, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20],
        'conf': [-1, -1, -1, -1, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90],
        'text': ["", "", "", "", "TOP1", "TOP2", "INFO", "PAGE", "MRZ1", "MRZ2", "BOT1", "BOT2", "END1", "END2"]
    }
    mock_image_to_data.return_value = mock_data
    
    split_type, split_coord = PassportSplitter.detect_hybrid_split(img)
    assert split_type == "horizontal"
    # Horizontal gap is [400, 500]. Center is 450.
    assert 440 <= split_coord <= 460
