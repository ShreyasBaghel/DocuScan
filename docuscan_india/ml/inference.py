import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List

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
    def get_thresholds(cls) -> dict:
        """Returns optimal thresholds, falling back to defaults if not present."""
        default_thresholds = {
            "ocr_threshold": 0.60,
            "classification_threshold": 0.70,
            "extraction_threshold": 0.50,
            "authenticity_genuine": 0.75,
            "authenticity_suspicious": 0.40,
            "fraud_suspicious": 0.60,
            "fraud_genuine": 0.25
        }
        if not cls._loaded or not cls._calib_info or not isinstance(cls._calib_info, dict):
            return default_thresholds
        return cls._calib_info.get("optimal_thresholds", default_thresholds)

    @classmethod
    def get_ensemble_weights(cls) -> dict:
        """Returns ensemble weights, falling back to defaults if not present."""
        default_weights = {"keyword": 0.40, "regex": 0.40, "layout": 0.20}
        if not cls._loaded or not cls._calib_info or not isinstance(cls._calib_info, dict):
            return default_weights
        return cls._calib_info.get("ensemble_weights", default_weights)

    @classmethod
    def explain_fraud_risk(cls, packet: DocumentPacket, baseline_prob: float) -> Tuple[List[int], List[int]]:
        """
        Calculates local model-driven feature attribution (risk impact) for each active
        fraud signal and validation failure using counterfactual probability shifts.
        Returns:
            - val_scores: List of scores corresponding to packet.validation_results
            - sig_scores: List of scores corresponding to packet.fraud_signals
        """
        if not cls._loaded or not cls._models or "fraud" not in cls._models:
            # Fallback to hardcoded scores
            val_scores = []
            for v in packet.validation_results:
                if v.status == "FAIL":
                    val_scores.append(40 if "checksum" in v.field_name or "verhoeff" in v.field_name else 20)
                elif v.status == "WARN":
                    val_scores.append(10)
                else:
                    val_scores.append(0)
            sig_scores = [s.score for s in packet.fraud_signals]
            return val_scores, sig_scores

        try:
            fe = FeatureEngineering()
            feat_dict = fe.extract_features(packet)
            
            # Map validation results
            val_scores = []
            for v in packet.validation_results:
                if v.status == "FAIL":
                    feat_copy = feat_dict.copy()
                    is_checksum = "checksum" in v.field_name or "verhoeff" in v.field_name
                    if is_checksum:
                        feat_copy["checksum_failed"] = max(0.0, feat_copy.get("checksum_failed", 0.0) - 1.0)
                    else:
                        feat_copy["validation_fails"] = max(0.0, feat_copy.get("validation_fails", 0.0) - 1.0)
                    
                    vec = fe.to_array(feat_copy).reshape(1, -1)
                    df_vec = pd.DataFrame(vec, columns=FEATURE_NAMES)
                    prob_cf = float(cls._models["fraud"].predict_proba(df_vec)[0][1])
                    diff = max(0.0, baseline_prob - prob_cf)
                    score = int(round(diff * 100))
                    val_scores.append(max(15 if is_checksum else 10, score))
                elif v.status == "WARN":
                    feat_copy = feat_dict.copy()
                    feat_copy["validation_warns"] = max(0.0, feat_copy.get("validation_warns", 0.0) - 1.0)
                    vec = fe.to_array(feat_copy).reshape(1, -1)
                    df_vec = pd.DataFrame(vec, columns=FEATURE_NAMES)
                    prob_cf = float(cls._models["fraud"].predict_proba(df_vec)[0][1])
                    diff = max(0.0, baseline_prob - prob_cf)
                    val_scores.append(max(5, int(round(diff * 100))))
                else:
                    val_scores.append(0)

            # Map fraud signals
            sig_scores = []
            for sig in packet.fraud_signals:
                feat_copy = feat_dict.copy()
                if sig.name == "editing_software_detected":
                    feat_copy["exif_editor_detected"] = 0.0
                elif sig.name == "timestamp_mismatch":
                    feat_copy["exif_timestamp_mismatch"] = 0.0
                elif "overlapping" in sig.name:
                    feat_copy["overlapping_fields_count"] = max(0.0, feat_copy.get("overlapping_fields_count", 0.0) - 1.0)
                elif "layout_anomaly" in sig.name:
                    feat_copy["layout_anomaly_count"] = max(0.0, feat_copy.get("layout_anomaly_count", 0.0) - 1.0)
                
                vec = fe.to_array(feat_copy).reshape(1, -1)
                df_vec = pd.DataFrame(vec, columns=FEATURE_NAMES)
                prob_cf = float(cls._models["fraud"].predict_proba(df_vec)[0][1])
                diff = max(0.0, baseline_prob - prob_cf)
                score = int(round(diff * 100))
                
                floors = {
                    "editing_software_detected": 15,
                    "timestamp_mismatch": 10,
                    "overlapping_fields_detected": 15,
                    "layout_anomaly": 10
                }
                floor_val = floors.get(sig.name, 10)
                if "layout_anomaly" in sig.name:
                    floor_val = 10
                sig_scores.append(max(floor_val, score))

            return val_scores, sig_scores

        except Exception as e:
            logger.error(f"Failed to explain fraud risk: {e}")
            val_scores = []
            for v in packet.validation_results:
                if v.status == "FAIL":
                    val_scores.append(40 if "checksum" in v.field_name or "verhoeff" in v.field_name else 20)
                elif v.status == "WARN":
                    val_scores.append(10)
                else:
                    val_scores.append(0)
            sig_scores = [s.score for s in packet.fraud_signals]
            return val_scores, sig_scores

    @classmethod
    def predict(cls, packet: DocumentPacket) -> Dict[str, Any]:
        """
        Runs inference on a DocumentPacket to generate calibrated scoring values.
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

            # Create a copy of feature dictionary to zero out post-OCR columns for OCR prediction
            ocr_feat_dict = feat_dict.copy()
            post_ocr_cols = [
                "validation_fails",
                "validation_warns",
                "checksum_failed",
                "exif_editor_detected",
                "exif_timestamp_mismatch",
                "overlapping_fields_count",
                "layout_anomaly_count"
            ]
            for col in post_ocr_cols:
                ocr_feat_dict[col] = 0.0
            
            ocr_feat_vector = fe.to_array(ocr_feat_dict).reshape(1, -1)
            df_ocr_feat = pd.DataFrame(ocr_feat_vector, columns=FEATURE_NAMES)

            # Predict probabilities
            # CalibratedClassifierCV's predict_proba outputs [p_neg, p_pos]
            ocr_prob = float(cls._models["ocr"].predict_proba(df_ocr_feat)[0][1])
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

            # Map to final decision using dynamic thresholds
            thresholds = cls.get_thresholds()
            auth_gen_score = int(round(thresholds.get("authenticity_genuine", 0.75) * 100))
            auth_susp_score = int(round(thresholds.get("authenticity_suspicious", 0.40) * 100))
            fraud_susp_score = int(round(thresholds.get("fraud_suspicious", 0.60) * 100))
            fraud_gen_score = int(round(thresholds.get("fraud_genuine", 0.25) * 100))
            
            verdict_thresholds = {
                "auth_genuine": auth_gen_score,
                "auth_suspicious": auth_susp_score,
                "fraud_suspicious": fraud_susp_score,
                "fraud_genuine": fraud_gen_score
            }
            decision, desc = ScoreFormatter.get_verdict(auth_score, fraud_risk, verdict_thresholds)

            logger.info(f"Model Inference Results: OCR={ocr_conf}, Class={class_conf}, "
                        f"Extract={extract_conf}, FraudRisk={fraud_risk}, Auth={auth_score} -> Verdict={decision}")

            return {
                "ocr_confidence": ocr_conf,
                "classification_confidence": class_conf,
                "extraction_reliability": extract_conf,
                "fraud_risk_score": fraud_risk,
                "authenticity_score": auth_score,
                "final_decision": decision,
                "decision_description": desc,
                "fraud_prob": fraud_prob
            }

        except Exception as e:
            logger.error(f"Inference prediction failed: {e}. Executing fallback behavior.", exc_info=True)
            return cls._fallback_predict(packet)

    @classmethod
    def _fallback_predict(cls, packet: DocumentPacket) -> Dict[str, Any]:
        """Provides safe, deterministic, rules-based calculations when models are missing."""
        logger.info("Executing safe fallback scoring calculations.")

        fe = FeatureEngineering()
        feat_dict = fe.extract_features(packet)

        # 1. OCR Confidence: rules-based combining mean, median, valid words, dictionary matching and blur.
        mean_conf = feat_dict.get("ocr_mean_conf", packet.ocr_confidence)
        median_conf = feat_dict.get("ocr_median_conf", packet.ocr_confidence)
        
        base_ocr = 0.5 * mean_conf + 0.5 * median_conf
        
        valid_ratio = feat_dict.get("valid_words_ratio", 0.0)
        alnum_ratio = feat_dict.get("alphanumeric_ratio", 0.0)
        dict_ratio = feat_dict.get("dictionary_match_ratio", 0.0)
        word_count = feat_dict.get("word_count", 0.0)
        
        if word_count == 0:
            ocr_prob = packet.ocr_confidence
        else:
            blur_score = feat_dict.get("blur_score", 1.0)
            scaled_blur = min(1.0, blur_score / 1500.0) if blur_score > 0 else 0.5
            
            # Quality factor based on layout and vocabulary quality features
            quality_factor = 0.4 * valid_ratio + 0.3 * alnum_ratio + 0.3 * dict_ratio
            
            # Weighted average of base OCR and quality factor
            ocr_prob = 0.75 * base_ocr + 0.25 * quality_factor
            
            # Apply low confidence word penalty
            pct_low = feat_dict.get("ocr_pct_low_conf", 0.0)
            ocr_prob = max(0.0, ocr_prob - 0.15 * pct_low)
            
            # Apply blur penalty
            if blur_score < 400.0 and blur_score > 0:
                ocr_prob = max(0.0, ocr_prob - 0.20 * ((400.0 - blur_score) / 400.0))
        
        ocr_conf = ScoreFormatter.to_score_range(ocr_prob)

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

        # Map to final decision using dynamic thresholds
        thresholds = cls.get_thresholds()
        auth_gen_score = int(round(thresholds.get("authenticity_genuine", 0.75) * 100))
        auth_susp_score = int(round(thresholds.get("authenticity_suspicious", 0.40) * 100))
        fraud_susp_score = int(round(thresholds.get("fraud_suspicious", 0.60) * 100))
        fraud_gen_score = int(round(thresholds.get("fraud_genuine", 0.25) * 100))
        
        verdict_thresholds = {
            "auth_genuine": auth_gen_score,
            "auth_suspicious": auth_susp_score,
            "fraud_suspicious": fraud_susp_score,
            "fraud_genuine": fraud_gen_score
        }
        decision, desc = ScoreFormatter.get_verdict(auth_score, fraud_risk, verdict_thresholds)

        return {
            "ocr_confidence": ocr_conf,
            "classification_confidence": class_conf,
            "extraction_reliability": extract_conf,
            "fraud_risk_score": fraud_risk,
            "authenticity_score": auth_score,
            "final_decision": decision,
            "decision_description": desc
        }
