import pytest
from unittest.mock import patch, MagicMock
from utils.document_packet import DocumentPacket, DocumentType, FieldResult
from ocr.pipeline import VerificationPipeline

@patch("ocr.image_loader.ImageLoader.load")
@patch("ocr.preprocessor.Preprocessor.preprocess")
@patch("ocr.ocr_engine.OCREngine.extract")
def test_pipeline_extraction_flow(mock_extract, mock_preprocess, mock_load):
    """
    Tests that:
    1. Early extraction is run in process_classification.
    2. process_verification skips re-extraction if document type matches.
    3. process_verification runs extraction if document type changes.
    """
    from PIL import Image
    mock_load.return_value = Image.new("RGB", (10, 10))
    mock_preprocess.return_value = Image.new("L", (10, 10))

    # Mock primary extract: return Aadhaar-like text but missing aadhaar_number to trigger fallback
    # In fallback return, return the complete text
    def side_effect(img, doc_type="UNKNOWN"):
        if doc_type == "UNKNOWN":
            raw_text = (
                "GOVERNMENT OF INDIA\n"
                "MERA AADHAAR, MERI PEHCHAN\n"
                "KARTIK KAPOOR\n"
                "DOB: 15/08/1990\n"
                "MALE\n"
            )
            word_map = [
                {"text": "GOVERNMENT", "left": 10, "top": 5, "width": 40, "height": 5, "conf": 0.95},
                {"text": "OF", "left": 55, "top": 5, "width": 10, "height": 5, "conf": 0.95},
                {"text": "INDIA", "left": 70, "top": 5, "width": 20, "height": 5, "conf": 0.95},
                {"text": "MERA", "left": 10, "top": 12, "width": 20, "height": 5, "conf": 0.95},
                {"text": "AADHAAR,", "left": 35, "top": 12, "width": 30, "height": 5, "conf": 0.95},
                {"text": "MERI", "left": 70, "top": 12, "width": 20, "height": 5, "conf": 0.95},
                {"text": "PEHCHAN", "left": 95, "top": 12, "width": 30, "height": 5, "conf": 0.95},
                {"text": "KARTIK", "left": 10, "top": 20, "width": 15, "height": 5, "conf": 0.90},
                {"text": "KAPOOR", "left": 30, "top": 20, "width": 15, "height": 5, "conf": 0.90},
                {"text": "MALE", "left": 10, "top": 40, "width": 10, "height": 5, "conf": 0.95},
                {"text": "DOB:", "left": 10, "top": 30, "width": 10, "height": 5, "conf": 0.90},
                {"text": "15/08/1990", "left": 25, "top": 30, "width": 20, "height": 5, "conf": 0.90}
            ]
            return raw_text, 0.92, word_map
        elif doc_type == "AADHAAR_FALLBACK":
            raw_text = (
                "GOVERNMENT OF INDIA\n"
                "MERA AADHAAR, MERI PEHCHAN\n"
                "KARTIK KAPOOR\n"
                "DOB: 15/08/1990\n"
                "MALE\n"
                "3668 7483 0214\n"
            )
            word_map = [
                {"text": "GOVERNMENT", "left": 10, "top": 5, "width": 40, "height": 5, "conf": 0.95},
                {"text": "OF", "left": 55, "top": 5, "width": 10, "height": 5, "conf": 0.95},
                {"text": "INDIA", "left": 70, "top": 5, "width": 20, "height": 5, "conf": 0.95},
                {"text": "MERA", "left": 10, "top": 12, "width": 20, "height": 5, "conf": 0.95},
                {"text": "AADHAAR,", "left": 35, "top": 12, "width": 30, "height": 5, "conf": 0.95},
                {"text": "MERI", "left": 70, "top": 12, "width": 20, "height": 5, "conf": 0.95},
                {"text": "PEHCHAN", "left": 95, "top": 12, "width": 30, "height": 5, "conf": 0.95},
                {"text": "KARTIK", "left": 10, "top": 20, "width": 15, "height": 5, "conf": 0.90},
                {"text": "KAPOOR", "left": 30, "top": 20, "width": 15, "height": 5, "conf": 0.90},
                {"text": "MALE", "left": 10, "top": 40, "width": 10, "height": 5, "conf": 0.95},
                {"text": "DOB:", "left": 10, "top": 30, "width": 10, "height": 5, "conf": 0.90},
                {"text": "15/08/1990", "left": 25, "top": 30, "width": 20, "height": 5, "conf": 0.90},
                {"text": "3668", "left": 10, "top": 50, "width": 10, "height": 5, "conf": 0.95},
                {"text": "7483", "left": 25, "top": 50, "width": 10, "height": 5, "conf": 0.95},
                {"text": "0214", "left": 40, "top": 50, "width": 10, "height": 5, "conf": 0.95}
            ]
            return raw_text, 0.95, word_map
        elif doc_type == "PAN_FALLBACK":
            raw_text = (
                "INCOME TAX DEPARTMENT\n"
                "Permanent Account Number Card\n"
                "HXDPB8430A\n"
                "Name: SHREYAS BAGHEL\n"
                "Father's Name: SANJEEV KUMAR\n"
                "Date of Birth: 01/01/2006\n"
            )
            word_map = [
                {"text": "HXDPB8430A", "left": 10, "top": 50, "width": 30, "height": 5, "conf": 0.95},
                {"text": "SHREYAS", "left": 10, "top": 20, "width": 15, "height": 5, "conf": 0.90},
                {"text": "BAGHEL", "left": 30, "top": 20, "width": 15, "height": 5, "conf": 0.90},
                {"text": "SANJEEV", "left": 10, "top": 30, "width": 15, "height": 5, "conf": 0.90},
                {"text": "KUMAR", "left": 30, "top": 30, "width": 15, "height": 5, "conf": 0.90},
                {"text": "01/01/2006", "left": 10, "top": 40, "width": 20, "height": 5, "conf": 0.95}
            ]
            return raw_text, 0.95, word_map
        return "", 0.0, []

    mock_extract.side_effect = side_effect

    pipeline = VerificationPipeline()
    
    # 1. Classification (should run early extraction & fallback OCR for Aadhaar)
    packet = pipeline.process_classification("dummy_path.jpg")
    
    assert packet.document_type == DocumentType.AADHAAR
    # Fallback OCR should have been called during classification!
    assert packet.extracted_fields["aadhaar_number"].value == "366874830214"
    assert packet.extracted_fields["dob"].value == "1990-08-15"
    assert packet.extracted_fields["gender"].value == "MALE"
    assert packet.extracted_fields["name"].value == "KARTIK KAPOOR"
    assert packet.pipeline_metadata.get("extracted_doc_type") == DocumentType.AADHAAR

    # 2. Verification with same document type
    # We clear calls from mock_extract to verify it is not called again during verification
    mock_extract.reset_mock()
    
    completed_packet = pipeline.process_verification(packet)
    
    # Verify mock_extract was NOT called during process_verification because we skipped extraction
    # (Since fields were already extracted for Aadhaar)
    for call in mock_extract.call_args_list:
        assert "FALLBACK" not in call[0][1]

    # The fields should remain correctly set
    assert completed_packet.extracted_fields["aadhaar_number"].value == "366874830214"
    assert completed_packet.pipeline_metadata.get("extracted_doc_type") == DocumentType.AADHAAR

    # 3. Changing the document type manually should trigger re-extraction
    packet.document_type = DocumentType.PAN
    mock_extract.reset_mock()
    
    completed_packet_pan = pipeline.process_verification(packet)
    
    # Check that it executed PAN fallback OCR or PAN extractor
    # And changed the extracted doc type to PAN
    assert completed_packet_pan.pipeline_metadata.get("extracted_doc_type") == DocumentType.PAN
    assert completed_packet_pan.extracted_fields["pan_number"].value == "HXDPB8430A"
