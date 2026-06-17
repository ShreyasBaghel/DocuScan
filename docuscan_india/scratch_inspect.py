import os
import sys
import pickle
import json
import pandas as pd
import numpy as np
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.document_packet import DocumentPacket, DocumentType
from ml.feature_engineering import FeatureEngineering, FEATURE_NAMES
from ml.inference import ScoringInference
from fraud_detection.tamper_features import TamperFeatures

# Mock image quality metrics return values dynamically
mock_blur = 2000.0
mock_noise = 60.0
mock_edge = 0.15

TamperFeatures.calculate_blur_score = lambda img: mock_blur
TamperFeatures.calculate_noise_score = lambda img: mock_noise
TamperFeatures.calculate_edge_density = lambda img: mock_edge

# Load models
ScoringInference.load_models()

def generate_mock_words(count: int, high_conf_count: int, high_conf_val: float, low_conf_count: int, low_conf_val: float, dict_match_count: int, valid_words_count: int, height_std: float):
    words = []
    common_keywords = ["aadhaar", "unique", "identification", "india", "government", "name", "dob", "gender", "address", "tax"]
    
    for i in range(count):
        # Determine text
        if i < dict_match_count:
            text = common_keywords[i % len(common_keywords)]
        elif i < valid_words_count:
            text = "validWord"
        else:
            text = "!!!"  # Non-alphanumeric
            
        # Determine conf
        if i < high_conf_count:
            conf = high_conf_val
        elif i < high_conf_count + low_conf_count:
            conf = low_conf_val
        else:
            conf = 0.55
            
        # Determine layout height
        height = int(np.random.normal(10.0, height_std))
        height = max(1, height)
        
        words.append({
            "text": text,
            "conf": conf,
            "top": 10 + i * 20,
            "left": 10,
            "width": 50,
            "height": height
        })
    return words

def evaluate_mock_packet(name: str, ocr_mean: float, ocr_word_map: list, blur: float, noise: float, edge: float):
    global mock_blur, mock_noise, mock_edge
    mock_blur = blur
    mock_noise = noise
    mock_edge = edge
    
    # Create dummy image to trigger calculations
    dummy_img = Image.new("RGB", (1000, 1000), color="white")
    
    packet = DocumentPacket(
        image_path="dummy_path.jpg",
        raw_image=dummy_img,
        ocr_raw_text="GOVERNMENT OF INDIA Aadhaar Card. Unique ID: 1234 5678 9012.",
        ocr_confidence=ocr_mean,
        ocr_word_map=ocr_word_map,
        document_type=DocumentType.AADHAAR,
        classification_confidence=0.95
    )
    
    res = ScoringInference.predict(packet)
    print(f"[{name}]")
    print(f"  Inputs -> OCR Mean: {ocr_mean:.2f}, Blur: {blur:.1f}, BBox H Std: {np.std([w.get('height', 10) for w in ocr_word_map]):.2f}")
    print(f"  Result -> Calibrated OCR Conf: {res['ocr_confidence']}%")

print("\n--- Verifying Target Quality Brackets ---")

# Excellent Image
# OCR Mean: 0.99, Blur: 3000.0, low BBox H std
excellent_words = generate_mock_words(30, 29, 0.99, 1, 0.90, 15, 30, 0.2)
evaluate_mock_packet("Excellent Quality", 0.99, excellent_words, 3000.0, 70.0, 0.18)

# Good Image
# OCR Mean: 0.82, Blur: 1200.0, low BBox H std
good_words = generate_mock_words(30, 25, 0.85, 5, 0.68, 11, 28, 1.0)
evaluate_mock_packet("Good Quality", 0.82, good_words, 1200.0, 50.0, 0.12)

# Average Image
# OCR Mean: 0.68, Blur: 650.0, moderate BBox H std
avg_words = generate_mock_words(30, 20, 0.74, 10, 0.54, 7, 25, 2.0)
evaluate_mock_packet("Average Quality", 0.68, avg_words, 650.0, 40.0, 0.10)

# Poor Image
# OCR Mean: 0.50, Blur: 380.0, high BBox H std
poor_words = generate_mock_words(30, 15, 0.62, 15, 0.42, 5, 20, 3.0)
evaluate_mock_packet("Poor Quality", 0.50, poor_words, 380.0, 25.0, 0.06)

# Unreadable Image
# OCR Mean: 0.24, Blur: 120.0, high BBox H std
unreadable_words = generate_mock_words(30, 2, 0.61, 28, 0.21, 1, 10, 6.0)
evaluate_mock_packet("Unreadable Quality", 0.24, unreadable_words, 120.0, 10.0, 0.02)

