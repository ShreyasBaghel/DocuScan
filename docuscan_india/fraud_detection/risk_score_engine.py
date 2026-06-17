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
        
        # Calculate dynamic model-driven attributions
        fraud_prob = res.get("fraud_prob", float(res["fraud_risk_score"]) / 100.0)
        val_scores, sig_scores = ScoringInference.explain_fraud_risk(packet, fraud_prob)
        
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
                
            # Assign dynamic model-driven scores back to validation results and fraud signals
            for v, score in zip(packet.validation_results, val_scores):
                v.score = score
            for s, score in zip(packet.fraud_signals, sig_scores):
                s.score = score

        else:
            # For backward compatibility, update the input lists if present
            if validation_results:
                for v, score in zip(validation_results, val_scores):
                    v.score = score
            if fraud_signals:
                for s, score in zip(fraud_signals, sig_scores):
                    s.score = score

        return res["fraud_risk_score"]

