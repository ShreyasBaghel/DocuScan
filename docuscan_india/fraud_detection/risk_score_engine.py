from typing import List, Dict, Any
from utils.document_packet import FraudSignal, ValidationResult, DocumentType
from utils.logger import get_logger

logger = get_logger("risk_score_engine")

class RiskScoreEngine:
    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the Risk Score Engine.
        config: Dict containing fraud_weights and risk_score_tiers.
        """
        self.config = config

    def calculate(self, 
                  ocr_confidence: float, 
                  validation_results: List[ValidationResult], 
                  fraud_signals: List[FraudSignal]) -> int:
        """
        Computes a final integer fraud risk score between 0 and 100.
        Uses weights configured in config.yaml.
        """
        # 1. Load weights
        fraud_config = self.config.get("fraud_weights", {})
        w_exif_software = fraud_config.get("exif_editing_software", 30)
        w_exif_timestamp = fraud_config.get("exif_timestamp_anomaly", 15)
        w_failed_checksum = fraud_config.get("failed_checksum", 40)
        w_format_inconsistency = fraud_config.get("format_inconsistency", 25)
        w_validation_fail = fraud_config.get("validation_failure", 20)
        w_validation_warn = w_validation_fail // 2

        score = 0

        # 2. Add EXIF signals
        for sig in fraud_signals:
            if sig.name == "editing_software_detected":
                score += w_exif_software
            elif sig.name == "timestamp_mismatch":
                score += w_exif_timestamp
            elif "layout_anomaly" in sig.name or "overlapping" in sig.name:
                score += w_format_inconsistency
            else:
                score += sig.score

        # 3. Add validation issues
        checksum_failed = False
        for res in validation_results:
            if res.status == "FAIL":
                # Check if it is a checksum failure
                if "checksum" in res.field_name or "verhoeff" in res.field_name:
                    checksum_failed = True
                else:
                    score += w_validation_fail
            elif res.status == "WARN":
                score += w_validation_warn

        if checksum_failed:
            score += w_failed_checksum

        # 4. Add OCR Confidence penalty (poor scan, blurry text)
        if ocr_confidence < 0.60:
            penalty = int((0.60 - ocr_confidence) * 50)
            score += penalty
            logger.info(f"Adding OCR low confidence penalty: +{penalty} (OCR Conf: {ocr_confidence:.2f})")

        # 5. Cap risk score to [0, 100]
        final_score = max(0, min(100, score))
        logger.info(f"Calculated fraud risk score: {final_score}/100")
        
        return final_score
