from typing import List, Dict, Any, Union
from utils.document_packet import FraudSignal, ValidationResult, DocumentType, DocumentPacket
from ml.inference import ScoringInference
from utils.logger import get_logger

logger = get_logger("risk_score_engine")

class RiskScoreEngine:
    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the Risk Score Engine.
        config: Dict containing config options.
        """
        self.config = config

    def calculate(self, 
                  packet_or_conf: Union[DocumentPacket, float], 
                  validation_results: List[ValidationResult] = None, 
                  fraud_signals: List[FraudSignal] = None) -> int:
        """
        Computes a final integer fraud risk score between 0 and 100 using ML model inference.
        """
        if isinstance(packet_or_conf, DocumentPacket):
            packet = packet_or_conf
        else:
            # Reconstruct dummy packet for backwards compatibility
            packet = DocumentPacket(
                image_path="",
                ocr_confidence=packet_or_conf,
                validation_results=validation_results or [],
                fraud_signals=fraud_signals or []
            )

        res = ScoringInference.predict(packet)
        # Update packet if it's a real DocumentPacket
        if isinstance(packet_or_conf, DocumentPacket):
            packet.fraud_risk_score = res["fraud_risk_score"]
            packet.authenticity_score = res["authenticity_score"]
            packet.extraction_reliability = res["extraction_reliability"]
            packet.final_decision = res["final_decision"]
            # Calibrate ocr and classification confidence
            packet.ocr_confidence = res["ocr_confidence"] / 100.0
            if packet.document_type != DocumentType.UNKNOWN:
                packet.classification_confidence = res["classification_confidence"] / 100.0

        return res["fraud_risk_score"]

