import pytest
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.document_packet import DocumentPacket, DocumentType, FieldResult
from utils.string_utils import normalize_date, is_valid_name
from ocr.pipeline import VerificationPipeline
from extractors.base_extractor import BaseExtractor
from extractors.passport_extractor import PassportExtractor

def test_date_normalization_OCR_corrections():
    # Test normalization of specific OCR anomalies
    assert normalize_date("19D8-50-52") == "1985-05-24"
    assert normalize_date("20M2-30-00") == "2023-01-01"

    # Test standard OCR character corrections
    assert normalize_date("19O8-O5-2Z") == "1908-05-22"
    assert normalize_date("20l8-10-15") == "2018-10-15"

def test_name_validation_filters():
    # Verify that slogan phrases and invalid names are rejected
    assert is_valid_name("MERA AADHAAR MERI PEHCHAN") is False
    assert is_valid_name("GOVERNMENT OF INDIA") is False
    assert is_valid_name("KARTIK KAPOOR") is True

def test_future_dob_penalty():
    # Test that future DOB is penalized
    class DummyExtractor(BaseExtractor):
        def extract(self, raw_text, word_map):
            return {}
            
    ext = DummyExtractor()
    cands = [
        {"text": "2035-10-10", "raw_text": "2035-10-10", "ocr_confidence": 0.90, "page_source": "visual"},
        {"text": "1995-10-10", "raw_text": "1995-10-10", "ocr_confidence": 0.80, "page_source": "visual"}
    ]
    
    # 2035-10-10 should get penalized by 5 points because it's a future DOB
    score_future = ext.score_candidate("dob", cands[0], DocumentType.AADHAAR, [], "")
    score_past = ext.score_candidate("dob", cands[1], DocumentType.AADHAAR, [], "")
    
    assert score_future < score_past

def test_classification_fallback(monkeypatch):
    pipeline = VerificationPipeline()
    
    # Mock OCR engine to return raw text with a valid PAN pattern and high confidence (> 60%)
    class MockOCREngine:
        config = {}
        def extract(self, img, doc_type="UNKNOWN"):
            raw_text = "INCOME TAX DEPARTMENT\nPERMANENT ACCOUNT NUMBER CARD\nABCDE1234F"
            word_map = [{"text": "ABCDE1234F", "left": 10, "top": 10, "width": 10, "height": 10, "conf": 0.90}]
            return raw_text, 0.95, word_map
            
    # Mock ClassifierEnsemble to return UNKNOWN class (as if standard keyword/layout matching failed)
    class MockClassifier:
        def classify(self, text, word_map):
            return DocumentType.UNKNOWN, 0.20

    monkeypatch.setattr(pipeline, "ocr_engine", MockOCREngine())
    monkeypatch.setattr(pipeline, "classifier", MockClassifier())
    
    from PIL import Image
    # Set packet OCR confidence manually in mock pipeline or bypass image loading
    class MockImageLoader:
        @staticmethod
        def load(path):
            return Image.new("RGB", (100, 100), (255, 255, 255))
            
    class MockPreprocessor:
        def preprocess(self, img):
            return Image.new("RGB", (100, 100), (255, 255, 255))

    monkeypatch.setattr("ocr.pipeline.ImageLoader", MockImageLoader)
    monkeypatch.setattr(pipeline, "preprocessor", MockPreprocessor())

    packet = pipeline.process_classification("dummy.jpg")
    
    # Classification fallback should have run and set type to PAN
    assert packet.document_type == DocumentType.PAN
