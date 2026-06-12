import numpy as np
from typing import Dict, List, Any
import re

from utils.document_packet import DocumentPacket, DocumentType
from fraud_detection.tamper_features import TamperFeatures
from utils.validation_helpers import ValidationHelpers
from classifiers.keyword_classifier import KeywordClassifier
from classifiers.regex_classifier import RegexClassifier
from classifiers.layout_classifier import LayoutClassifier

# Pre-defined feature names in a stable order to ensure consistency between training and inference
FEATURE_NAMES = [
    # OCR Confidence Stats
    "ocr_mean_conf",
    "ocr_min_conf",
    "ocr_max_conf",
    "ocr_std_conf",
    "ocr_pct_low_conf",
    # Text summaries
    "word_count",
    "char_count",
    "avg_word_len",
    # Classifier Sub-scores
    "kw_score_aadhaar",
    "kw_score_pan",
    "kw_score_passport",
    "kw_score_dl",
    "rg_score_aadhaar",
    "rg_score_pan",
    "rg_score_passport",
    "rg_score_dl",
    "ly_score_aadhaar",
    "ly_score_pan",
    "ly_score_passport",
    "ly_score_dl",
    # Image Quality / Blur / Noise
    "blur_score",
    "noise_score",
    "edge_density",
    # Font/Bounding Box Statistics
    "bbox_height_std",
    "bbox_width_std",
    "bbox_area_ratio",
    # Validation & Fraud signals
    "validation_fails",
    "validation_warns",
    "checksum_failed",
    "missing_fields_count",
    "overlapping_fields_count",
    "layout_anomaly_count",
    "exif_editor_detected",
    "exif_timestamp_mismatch",
    # Regex Match Counts
    "regex_match_aadhaar",
    "regex_match_pan",
    "regex_match_passport",
    "regex_match_dl"
]

class FeatureEngineering:
    def __init__(self):
        self.keyword_clf = KeywordClassifier()
        self.regex_clf = RegexClassifier()
        self.layout_clf = LayoutClassifier()

    def extract_features(self, packet: DocumentPacket) -> Dict[str, float]:
        """
        Extracts a dictionary of features from a DocumentPacket.
        """
        features = {}

        # 1. OCR Confidence Statistics
        confidences = []
        word_lengths = []
        bbox_heights = []
        bbox_widths = []
        bbox_areas = []
        
        # Determine image area for ratio calculations
        img_w, img_h = 1000, 1000  # defaults
        if packet.preprocessed_image:
            img_w, img_h = packet.preprocessed_image.size
        elif packet.raw_image:
            img_w, img_h = packet.raw_image.size
        img_area = float(img_w * img_h)

        for w in packet.ocr_word_map:
            conf = w.get("conf", 0.0)
            confidences.append(conf)
            text = w.get("text", "")
            if text:
                word_lengths.append(len(text))
            
            w_h = w.get("height", 0)
            w_w = w.get("width", 0)
            bbox_heights.append(w_h)
            bbox_widths.append(w_w)
            bbox_areas.append(w_w * w_h)

        if confidences:
            features["ocr_mean_conf"] = float(np.mean(confidences))
            features["ocr_min_conf"] = float(np.min(confidences))
            features["ocr_max_conf"] = float(np.max(confidences))
            features["ocr_std_conf"] = float(np.std(confidences))
            features["ocr_pct_low_conf"] = float(sum(1 for c in confidences if c < 0.6) / len(confidences))
        else:
            features["ocr_mean_conf"] = float(packet.ocr_confidence)
            features["ocr_min_conf"] = float(packet.ocr_confidence)
            features["ocr_max_conf"] = float(packet.ocr_confidence)
            features["ocr_std_conf"] = 0.0
            features["ocr_pct_low_conf"] = 1.0 if packet.ocr_confidence < 0.6 else 0.0

        features["word_count"] = float(len(packet.ocr_word_map))
        features["char_count"] = float(len(packet.ocr_raw_text))
        features["avg_word_len"] = float(np.mean(word_lengths)) if word_lengths else 0.0

        # 2. Classifier Sub-scores
        # Keyword scores
        _, kw_a = self.keyword_clf.classify(packet.ocr_raw_text, packet.ocr_word_map)
        # Check specific classes
        kw_scores = {DocumentType.AADHAAR: 0.0, DocumentType.PAN: 0.0, DocumentType.PASSPORT: 0.0, DocumentType.DRIVING_LICENCE: 0.0}
        kw_doc, kw_score = self.keyword_clf.classify(packet.ocr_raw_text, packet.ocr_word_map)
        if kw_doc in kw_scores:
            kw_scores[kw_doc] = kw_score
        
        features["kw_score_aadhaar"] = kw_scores[DocumentType.AADHAAR]
        features["kw_score_pan"] = kw_scores[DocumentType.PAN]
        features["kw_score_passport"] = kw_scores[DocumentType.PASSPORT]
        features["kw_score_dl"] = kw_scores[DocumentType.DRIVING_LICENCE]

        # Regex scores
        rg_scores = {DocumentType.AADHAAR: 0.0, DocumentType.PAN: 0.0, DocumentType.PASSPORT: 0.0, DocumentType.DRIVING_LICENCE: 0.0}
        rg_doc, rg_score = self.regex_clf.classify(packet.ocr_raw_text, packet.ocr_word_map)
        if rg_doc in rg_scores:
            rg_scores[rg_doc] = rg_score
        
        features["rg_score_aadhaar"] = rg_scores[DocumentType.AADHAAR]
        features["rg_score_pan"] = rg_scores[DocumentType.PAN]
        features["rg_score_passport"] = rg_scores[DocumentType.PASSPORT]
        features["rg_score_dl"] = rg_scores[DocumentType.DRIVING_LICENCE]

        # Layout scores
        ly_scores = {DocumentType.AADHAAR: 0.0, DocumentType.PAN: 0.0, DocumentType.PASSPORT: 0.0, DocumentType.DRIVING_LICENCE: 0.0}
        ly_doc, ly_score = self.layout_clf.classify(packet.ocr_raw_text, packet.ocr_word_map)
        if ly_doc in ly_scores:
            ly_scores[ly_doc] = ly_score

        features["ly_score_aadhaar"] = ly_scores[DocumentType.AADHAAR]
        features["ly_score_pan"] = ly_scores[DocumentType.PAN]
        features["ly_score_passport"] = ly_scores[DocumentType.PASSPORT]
        features["ly_score_dl"] = ly_scores[DocumentType.DRIVING_LICENCE]

        # 3. Image Quality Metrics
        img_to_measure = packet.preprocessed_image or packet.raw_image
        if img_to_measure:
            features["blur_score"] = TamperFeatures.calculate_blur_score(img_to_measure)
            features["noise_score"] = TamperFeatures.calculate_noise_score(img_to_measure)
            features["edge_density"] = TamperFeatures.calculate_edge_density(img_to_measure)
        else:
            features["blur_score"] = 0.0
            features["noise_score"] = 0.0
            features["edge_density"] = 0.0

        # 4. Font/Spacing/Bounding Box stats
        features["bbox_height_std"] = float(np.std(bbox_heights)) if bbox_heights else 0.0
        features["bbox_width_std"] = float(np.std(bbox_widths)) if bbox_widths else 0.0
        features["bbox_area_ratio"] = float(sum(bbox_areas) / img_area) if bbox_areas else 0.0

        # 5. Validation & Fraud signals
        val_summary = ValidationHelpers.get_validation_summary(packet.validation_results)
        features["validation_fails"] = float(val_summary["validation_fails"])
        features["validation_warns"] = float(val_summary["validation_warns"])
        features["checksum_failed"] = float(val_summary["checksum_failed"])

        features["missing_fields_count"] = float(ValidationHelpers.get_missing_fields_count(packet.extracted_fields))

        # Check fraud signals count
        overlaps = 0
        layout_anoms = 0
        exif_edit = 0
        exif_time = 0
        for sig in packet.fraud_signals:
            if sig.name == "editing_software_detected":
                exif_edit = 1
            elif sig.name == "timestamp_mismatch":
                exif_time = 1
            elif "overlapping" in sig.name:
                overlaps += 1
            elif "layout_anomaly" in sig.name:
                layout_anoms += 1

        features["overlapping_fields_count"] = float(overlaps)
        features["layout_anomaly_count"] = float(layout_anoms)
        features["exif_editor_detected"] = float(exif_edit)
        features["exif_timestamp_mismatch"] = float(exif_time)

        # 6. Regex Match Counts
        text = packet.ocr_raw_text
        features["regex_match_aadhaar"] = float(len(re.findall(r"\b\d{4}\s+\d{4}\s+\d{4}\b", text)))
        features["regex_match_pan"] = float(len(re.findall(r"\b[A-Z]{5}\d{4}[A-Z]\b", text)))
        features["regex_match_passport"] = float(len(re.findall(r"\b[A-Z]\d{7}\b", text)))
        # DL pattern match count
        features["regex_match_dl"] = float(len(re.findall(r"\b[A-Z]{2}\d{2}\s*\d{11}\b|\b[A-Z]{2}-\d{2}-\d{11}\b", text)))

        return features

    def to_array(self, features: Dict[str, float]) -> np.ndarray:
        """Converts feature dictionary to a stable feature vector (numpy array)."""
        return np.array([features.get(name, 0.0) for name in FEATURE_NAMES])
