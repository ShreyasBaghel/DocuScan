import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

from utils.document_packet import DocumentPacket, DocumentType
from ml.feature_engineering import FeatureEngineering, FEATURE_NAMES
from utils.score_formatter import ScoreFormatter
from utils.logger import get_logger

logger = get_logger("inference")

class ScoringInference:
    _models = None
    _calib_info = None
    _loaded = False

    @classmethod
    def load_models(cls) -> bool:
        """Loads serialized models from the models directory."""
        if cls._loaded:
            return True

        app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        scoring_path = os.path.join(app_dir, "models", "scoring_model.pkl")
        calib_path = os.path.join(app_dir, "models", "calibration_model.pkl")

        if not os.path.exists(scoring_path):
            logger.warning(f"Scoring model pkl not found at {scoring_path}. Fallback scoring will be used.")
            return False

        try:
            with open(scoring_path, "rb") as f:
                cls._models = pickle.load(f)
            
            if os.path.exists(calib_path):
                with open(calib_path, "rb") as f:
                    cls._calib_info = pickle.load(f)
            
            cls._loaded = True
            logger.info("Scoring and calibration models loaded successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to load model files: {e}. Falling back to rule-based logic.", exc_info=True)
            return False

    @classmethod
    def predict(cls, packet: DocumentPacket) -> Dict[str, Any]:
        """
        Runs inference on a DocumentPacket to generate calibrated scoring values.
        Returns:
            Dict containing:
              - ocr_confidence: int (0-100)
              - classification_confidence: int (0-100)
              - extraction_reliability: int (0-100)
              - fraud_risk_score: int (0-100)
              - authenticity_score: int (0-100)
              - final_decision: str (Genuine, Suspicious, Needs Manual Review)
              - decision_description: str
        """
        # Ensure models are loaded, otherwise use fallback
        models_available = cls.load_models()

        if not models_available or not cls._models:
            return cls._fallback_predict(packet)

        try:
            fe = FeatureEngineering()
            feat_dict = fe.extract_features(packet)
            feat_vector = fe.to_array(feat_dict).reshape(1, -1)
            
            # Convert to DataFrame with correct column names to avoid LightGBM feature name warnings
            df_feat = pd.DataFrame(feat_vector, columns=FEATURE_NAMES)

            # Predict probabilities
            # CalibratedClassifierCV's predict_proba outputs [p_neg, p_pos]
            ocr_prob = float(cls._models["ocr"].predict_proba(df_feat)[0][1])
            class_prob = float(cls._models["classification"].predict_proba(df_feat)[0][1])
            extract_prob = float(cls._models["extraction"].predict_proba(df_feat)[0][1])
            fraud_prob = float(cls._models["fraud"].predict_proba(df_feat)[0][1])
            auth_prob = float(cls._models["authenticity"].predict_proba(df_feat)[0][1])

            # Convert to integer scores in [0, 100] range
            ocr_conf = ScoreFormatter.to_score_range(ocr_prob)
            class_conf = ScoreFormatter.to_score_range(class_prob)
            extract_conf = ScoreFormatter.to_score_range(extract_prob)
            fraud_risk = ScoreFormatter.to_score_range(fraud_prob)
            auth_score = ScoreFormatter.to_score_range(auth_prob)

            # Enforce consistency: If document type is UNKNOWN, classification confidence is 0
            if packet.document_type == DocumentType.UNKNOWN:
                class_conf = 0

            # Map to final decision
            decision, desc = ScoreFormatter.get_verdict(auth_score, fraud_risk)

            logger.info(f"Model Inference Results: OCR={ocr_conf}, Class={class_conf}, "
                        f"Extract={extract_conf}, FraudRisk={fraud_risk}, Auth={auth_score} -> Verdict={decision}")

            return {
                "ocr_confidence": ocr_conf,
                "classification_confidence": class_conf,
                "extraction_reliability": extract_conf,
                "fraud_risk_score": fraud_risk,
                "authenticity_score": auth_score,
                "final_decision": decision,
                "decision_description": desc
            }

        except Exception as e:
            logger.error(f"Inference prediction failed: {e}. Executing fallback behavior.", exc_info=True)
            return cls._fallback_predict(packet)

    @classmethod
    def _fallback_predict(cls, packet: DocumentPacket) -> Dict[str, Any]:
        """Provides safe, deterministic, rules-based calculations when models are missing."""
        logger.info("Executing safe fallback scoring calculations.")

        # 1. OCR Confidence
        ocr_conf = ScoreFormatter.to_score_range(packet.ocr_confidence)

        # 2. Classification Confidence
        class_conf = ScoreFormatter.to_score_range(packet.classification_confidence)
        if packet.document_type == DocumentType.UNKNOWN:
            class_conf = 0

        # 3. Extraction Reliability
        total_fields = len(packet.extracted_fields)
        if total_fields > 0:
            missing_fields = sum(1 for f in packet.extracted_fields.values() if not f or f.value == "NOT_FOUND")
            extract_ratio = 1.0 - (missing_fields / total_fields)
            extract_conf = ScoreFormatter.to_score_range(extract_ratio)
        else:
            extract_conf = 0

        # 4. Fraud Risk (emulate original RiskScoreEngine weights)
        fraud_score = 0
        
        # Check exif
        for sig in packet.fraud_signals:
            if sig.name == "editing_software_detected":
                fraud_score += 30
            elif sig.name == "timestamp_mismatch":
                fraud_score += 15
            elif "layout_anomaly" in sig.name or "overlapping" in sig.name:
                fraud_score += 25
            else:
                fraud_score += sig.score

        # Check validation failures
        checksum_failed = False
        for res in packet.validation_results:
            if res.status == "FAIL":
                if "checksum" in res.field_name or "verhoeff" in res.field_name:
                    checksum_failed = True
                else:
                    fraud_score += 20
            elif res.status == "WARN":
                fraud_score += 10

        if checksum_failed:
            fraud_score += 40

        # OCR penalty
        if packet.ocr_confidence < 0.60:
            fraud_score += int((0.60 - packet.ocr_confidence) * 50)

        fraud_risk = max(0, min(100, fraud_score))

        # 5. Authenticity Score
        auth_score = max(0, 100 - fraud_risk)
        if ocr_conf < 50:
            auth_score = int(auth_score * 0.5)

        # Map to final decision
        decision, desc = ScoreFormatter.get_verdict(auth_score, fraud_risk)

        return {
            "ocr_confidence": ocr_conf,
            "classification_confidence": class_conf,
            "extraction_reliability": extract_conf,
            "fraud_risk_score": fraud_risk,
            "authenticity_score": auth_score,
            "final_decision": decision,
            "decision_description": desc
        }
