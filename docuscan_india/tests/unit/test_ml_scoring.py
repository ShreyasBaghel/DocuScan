import pytest
import os
import sys
import numpy as np

# Ensure the root directory of the project is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.document_packet import DocumentPacket, DocumentType, FieldResult, ValidationResult, FraudSignal
from ml.feature_engineering import FeatureEngineering, FEATURE_NAMES
from ml.inference import ScoringInference
from ml.training import generate_mock_dataset
from fraud_detection.risk_score_engine import RiskScoreEngine

def test_feature_engineering_extraction():
    """Verifies that features are extracted correctly and match the feature names list."""
    packet = DocumentPacket(
        image_path="dummy_path.jpg",
        ocr_raw_text="GOVERNMENT OF INDIA Aadhaar No: 1234 5678 9012 DOB: 01/01/1990",
        ocr_confidence=0.88,
        ocr_word_map=[
            {"text": "GOVERNMENT", "left": 10, "top": 10, "width": 50, "height": 10, "conf": 0.90},
            {"text": "Aadhaar", "left": 10, "top": 30, "width": 40, "height": 10, "conf": 0.85}
        ],
        document_type=DocumentType.AADHAAR,
        classification_confidence=0.95
    )
    
    fe = FeatureEngineering()
    feat_dict = fe.extract_features(packet)
    
    # Assert all feature names are present in dict
    for name in FEATURE_NAMES:
        assert name in feat_dict, f"Missing feature: {name}"
        
    feat_arr = fe.to_array(feat_dict)
    assert len(feat_arr) == len(FEATURE_NAMES)
    assert isinstance(feat_arr, np.ndarray)

def test_mock_dataset_generation():
    """Verifies mock training dataset generation dimensions and target values."""
    X, ys = generate_mock_dataset(n_samples=50)
    assert X.shape == (50, len(FEATURE_NAMES))
    assert "ocr" in ys
    assert "authenticity" in ys
    assert len(ys["ocr"]) == 50

def test_scoring_inference():
    """Verifies model loading, inference predictions, and fallback functionality."""
    packet = DocumentPacket(
        image_path="dummy_path.jpg",
        ocr_raw_text="GOVERNMENT OF INDIA Aadhaar No: 3668 7483 0214 MALE",
        ocr_confidence=0.92,
        ocr_word_map=[
            {"text": "3668", "conf": 0.95},
            {"text": "7483", "conf": 0.95},
            {"text": "0214", "conf": 0.95}
        ],
        document_type=DocumentType.AADHAAR,
        classification_confidence=0.90
    )
    
    # Predict using inference
    res = ScoringInference.predict(packet)
    
    # Verify the structure of returned results
    assert "ocr_confidence" in res
    assert "classification_confidence" in res
    assert "extraction_reliability" in res
    assert "fraud_risk_score" in res
    assert "authenticity_score" in res
    assert "final_decision" in res
    
    # Value range assertions
    assert 0 <= res["ocr_confidence"] <= 100
    assert 0 <= res["classification_confidence"] <= 100
    assert 0 <= res["extraction_reliability"] <= 100
    assert 0 <= res["fraud_risk_score"] <= 100
    assert 0 <= res["authenticity_score"] <= 100
    assert res["final_decision"] in ["Genuine", "Suspicious", "Needs Manual Review"]

def test_fallback_behavior_when_models_missing():
    """Verifies that the fallback engine correctly estimates scores without crashing."""
    packet = DocumentPacket(
        image_path="dummy.jpg",
        ocr_raw_text="Aadhaar Card",
        ocr_confidence=0.55,  # low conf penalty should apply
        validation_results=[
            ValidationResult(status="FAIL", field_name="dob", expected="valid date", actual="empty")
        ],
        fraud_signals=[
            FraudSignal(name="editing_software_detected", score=30, description="exif edit", source="MetadataAnalyser")
        ]
    )
    
    # Force fallback prediction
    res = ScoringInference._fallback_predict(packet)
    
    assert res["ocr_confidence"] == 55
    # low conf penalty (0.60-0.55)*50 = 2.5 (rounded to 2)
    # exif software = 30
    # failed validation = 20
    # total risk score should be: 30 + 20 + 2 = 52
    assert res["fraud_risk_score"] == 52
    assert res["authenticity_score"] == 48
