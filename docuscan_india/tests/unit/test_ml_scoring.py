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

def test_dynamic_thresholds_and_attribution():
    """Verifies that dynamic thresholds are loaded and explain_fraud_risk produces scores."""
    # Ensure models are loaded
    loaded = ScoringInference.load_models()
    assert loaded
    
    thresholds = ScoringInference.get_thresholds()
    assert "ocr_threshold" in thresholds
    assert "classification_threshold" in thresholds
    assert "authenticity_genuine" in thresholds
    
    weights = ScoringInference.get_ensemble_weights()
    assert "keyword" in weights
    assert "regex" in weights
    assert "layout" in weights
    
    packet = DocumentPacket(
        image_path="dummy.jpg",
        ocr_raw_text="Aadhaar Card with editing software detected",
        ocr_confidence=0.85,
        validation_results=[
            ValidationResult(status="FAIL", field_name="dob", expected="valid date", actual="empty")
        ],
        fraud_signals=[
            FraudSignal(name="editing_software_detected", score=30, description="exif edit", source="MetadataAnalyser")
        ]
    )
    
    res = ScoringInference.predict(packet)
    fraud_prob = res.get("fraud_prob", 0.0)
    
    val_scores, sig_scores = ScoringInference.explain_fraud_risk(packet, fraud_prob)
    assert len(val_scores) == len(packet.validation_results)
    assert len(sig_scores) == len(packet.fraud_signals)
    assert val_scores[0] >= 10
    assert sig_scores[0] >= 15


def test_new_features_extraction():
    """Validates the correct calculation of the 6 newly introduced OCR quality and layout features."""
    packet = DocumentPacket(
        image_path="dummy_path.jpg",
        ocr_raw_text="GOVERNMENT OF INDIA Aadhaar Card. Unique ID: 1234-5678-9012.",
        ocr_confidence=0.85,
        ocr_word_map=[
            {"text": "GOVERNMENT", "conf": 0.90, "top": 10, "left": 10, "width": 50, "height": 10},
            {"text": "OF", "conf": 0.80, "top": 10, "left": 70, "width": 20, "height": 10},
            {"text": "INDIA", "conf": 0.85, "top": 10, "left": 100, "width": 40, "height": 10},
            {"text": "Aadhaar", "conf": 0.70, "top": 30, "left": 10, "width": 50, "height": 10},
            {"text": "Card.", "conf": 0.95, "top": 30, "left": 70, "width": 30, "height": 10}
        ],
        document_type=DocumentType.AADHAAR,
        classification_confidence=0.95,
        extracted_fields={
            "aadhaar_no": FieldResult(value="1234-5678-9012", raw_text="1234-5678-9012", confidence=0.95),
            "dob": FieldResult(value="NOT_FOUND", raw_text="", confidence=0.0),
            "name": FieldResult(value="   ", raw_text="   ", confidence=0.0)
        }
    )
    
    fe = FeatureEngineering()
    feat_dict = fe.extract_features(packet)
    
    # 1. ocr_median_conf: median of [0.90, 0.80, 0.85, 0.70, 0.95] is 0.85
    assert abs(feat_dict["ocr_median_conf"] - 0.85) < 1e-5
    
    # 2. valid_words_ratio: words with alphanumeric characters.
    # Words: "GOVERNMENT", "OF", "INDIA", "Aadhaar", "Card."
    # All 5 contain alphanumeric characters, so 5/5 = 1.0.
    assert feat_dict["valid_words_ratio"] == 1.0
    
    # 3. alphanumeric_ratio: ratio of alphanumeric characters to total characters in ocr_raw_text
    raw_text = packet.ocr_raw_text
    total_chars = len(raw_text)
    alnum_chars = sum(1 for c in raw_text if c.isalnum())
    expected_alnum_ratio = float(alnum_chars / total_chars)
    assert abs(feat_dict["alphanumeric_ratio"] - expected_alnum_ratio) < 1e-5
    
    # 4. dictionary_match_ratio: words in COMMON_KEYWORDS: "government", "india", "aadhaar"
    # Matches: GOVERNMENT, INDIA, Aadhaar (3 words out of 5) -> 0.60
    assert abs(feat_dict["dictionary_match_ratio"] - 0.60) < 1e-5
    
    # 5. field_extraction_success_rate: 
    # extracted_fields: "aadhaar_no" is valid (non-empty, not NOT_FOUND)
    # "dob" is NOT_FOUND
    # "name" is whitespace-only
    # Success: 1 out of 3 -> 1/3
    assert abs(feat_dict["field_extraction_success_rate"] - (1.0 / 3.0)) < 1e-5
    
    # 6. text_density: total character count divided by image area (1000*1000 = 1000000 by default)
    expected_density = float(len(raw_text) / 1000000.0)
    assert abs(feat_dict["text_density"] - expected_density) < 1e-5


def test_ocr_prediction_stability():
    """Validates that OCR model predictions are stable between Stage 3 (no validations/fraud) and Stage 6 (validations/fraud present)."""
    # Stage 3 state packet: only OCR results, no validations or fraud signals
    packet_stage3 = DocumentPacket(
        image_path="dummy_path.jpg",
        ocr_raw_text="GOVERNMENT OF INDIA Aadhaar No: 3668 7483 0214 MALE",
        ocr_confidence=0.92,
        ocr_word_map=[
            {"text": "GOVERNMENT", "conf": 0.90, "top": 10, "left": 10, "width": 50, "height": 10},
            {"text": "Aadhaar", "conf": 0.95, "top": 30, "left": 10, "width": 40, "height": 10},
            {"text": "3668", "conf": 0.95, "top": 50, "left": 10, "width": 30, "height": 10},
            {"text": "7483", "conf": 0.95, "top": 50, "left": 50, "width": 30, "height": 10},
            {"text": "0214", "conf": 0.95, "top": 50, "left": 90, "width": 30, "height": 10}
        ],
        document_type=DocumentType.AADHAAR,
        classification_confidence=0.90
    )
    
    # Stage 6 state packet: same OCR, but now has validation results and fraud signals
    packet_stage6 = DocumentPacket(
        image_path="dummy_path.jpg",
        ocr_raw_text="GOVERNMENT OF INDIA Aadhaar No: 3668 7483 0214 MALE",
        ocr_confidence=0.92,
        ocr_word_map=[
            {"text": "GOVERNMENT", "conf": 0.90, "top": 10, "left": 10, "width": 50, "height": 10},
            {"text": "Aadhaar", "conf": 0.95, "top": 30, "left": 10, "width": 40, "height": 10},
            {"text": "3668", "conf": 0.95, "top": 50, "left": 10, "width": 30, "height": 10},
            {"text": "7483", "conf": 0.95, "top": 50, "left": 50, "width": 30, "height": 10},
            {"text": "0214", "conf": 0.95, "top": 50, "left": 90, "width": 30, "height": 10}
        ],
        document_type=DocumentType.AADHAAR,
        classification_confidence=0.90,
        validation_results=[
            ValidationResult(status="FAIL", field_name="dob", expected="valid date", actual="empty"),
            ValidationResult(status="FAIL", field_name="aadhaar_no_checksum", expected="valid", actual="invalid")
        ],
        fraud_signals=[
            FraudSignal(name="editing_software_detected", score=30, description="exif edit", source="MetadataAnalyser"),
            FraudSignal(name="overlapping_fields_detected", score=25, description="overlap", source="LayoutAnalyser")
        ]
    )
    
    res_stage3 = ScoringInference.predict(packet_stage3)
    res_stage6 = ScoringInference.predict(packet_stage6)
    
    # OCR confidence MUST be identical between both stages
    assert res_stage3["ocr_confidence"] == res_stage6["ocr_confidence"]

