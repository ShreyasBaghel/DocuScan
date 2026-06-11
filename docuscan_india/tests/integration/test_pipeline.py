import pytest
import os
import sys
import numpy as np
from unittest.mock import patch, MagicMock

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.document_packet import DocumentType, DocumentPacket
from ocr.pipeline import VerificationPipeline

@patch("ocr.image_loader.ImageLoader.load")
@patch("ocr.preprocessor.Preprocessor.preprocess")
@patch("ocr.ocr_engine.OCREngine.extract")
def test_full_pipeline_aadhaar(mock_extract, mock_preprocess, mock_load):
    """
    Integration test validating that a mock image flows through all 7 stages
    and produces database records, PDF reports, and JSON exports.
    """
    # 1. Setup mock returns for ImageLoader & Preprocessor
    from PIL import Image
    mock_load.return_value = Image.new("RGB", (100, 100), 0)
    mock_preprocess.return_value = Image.new("L", (100, 100), 0)


    # 2. Setup mock OCR return representing a valid Aadhaar Card
    # Let's use a valid Verhoeff Aadhaar: 366874830214
    mock_raw_text = (
        "GOVERNMENT OF INDIA\n"
        "MERA AADHAAR, MERI PEHCHAN\n"
        "KARTIK KAPOOR\n"
        "DOB: 15/08/1990\n"
        "MALE\n"
        "3668 7483 0214\n"
    )
    mock_word_map = [
        {"text": "3668", "left": 10, "top": 50, "width": 10, "height": 5, "conf": 0.95},
        {"text": "7483", "left": 25, "top": 50, "width": 10, "height": 5, "conf": 0.95},
        {"text": "0214", "left": 40, "top": 50, "width": 10, "height": 5, "conf": 0.95},
        {"text": "KARTIK", "left": 10, "top": 20, "width": 15, "height": 5, "conf": 0.90},
        {"text": "KAPOOR", "left": 30, "top": 20, "width": 15, "height": 5, "conf": 0.90},
        {"text": "MALE", "left": 10, "top": 40, "width": 10, "height": 5, "conf": 0.95},
        {"text": "DOB:", "left": 10, "top": 30, "width": 10, "height": 5, "conf": 0.90},
        {"text": "15/08/1990", "left": 25, "top": 30, "width": 20, "height": 5, "conf": 0.90}
    ]
    mock_extract.return_value = (mock_raw_text, 0.92, mock_word_map)

    # 3. Create dummy file to run tests on
    app_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dummy_img_path = os.path.join(app_dir, "tests", "fixtures", "sample_docs", "mock_aadhaar.jpg")
    os.makedirs(os.path.dirname(dummy_img_path), exist_ok=True)
    with open(dummy_img_path, "w") as f:
        f.write("dummy image data")

    try:
        # 4. Initialize and run the pipeline
        pipeline = VerificationPipeline()
        
        # Stages 1-3
        packet = pipeline.process_classification(dummy_img_path)
        
        assert packet.document_type == DocumentType.AADHAAR
        assert packet.classification_confidence > 0.70
        assert packet.ocr_confidence == 0.92
        
        # Stages 4-7
        completed_packet = pipeline.process_verification(packet)
        
        # Assert fields extracted correctly
        assert completed_packet.extracted_fields["aadhaar_number"].value == "366874830214"
        assert completed_packet.extracted_fields["dob"].value == "1990-08-15"
        assert completed_packet.extracted_fields["gender"].value == "MALE"
        assert completed_packet.extracted_fields["name"].value == "KARTIK KAPOOR"

        # Assert validations performed
        assert len(completed_packet.validation_results) > 0
        # Verhoeff should PASS
        verhoeff_results = [r for r in completed_packet.validation_results if r.field_name == "aadhaar_checksum_verhoeff"]
        assert len(verhoeff_results) == 1
        assert verhoeff_results[0].status == "PASS"

        # Assert fraud risk score computed
        assert 0 <= completed_packet.fraud_risk_score <= 100

        # Assert reports exported and exist
        assert completed_packet.report_path != ""
        assert os.path.exists(completed_packet.report_path)
        
        base_name = os.path.splitext(os.path.basename(dummy_img_path))[0]
        json_path = os.path.join(app_dir, "data", "exports", f"{base_name}_audit.json")
        html_path = os.path.join(app_dir, "data", "exports", f"{base_name}_audit.html")
        assert os.path.exists(json_path)
        assert os.path.exists(html_path)

        # Assert audit log saved in database
        db_path = os.path.join(app_dir, "data", "db", "audit.db")
        assert os.path.exists(db_path)

    finally:
        # Clean up dummy image fixture
        if os.path.exists(dummy_img_path):
            os.remove(dummy_img_path)
