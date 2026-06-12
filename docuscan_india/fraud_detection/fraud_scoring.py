from utils.document_packet import DocumentPacket
from ml.inference import ScoringInference
from typing import Dict, Any

class FraudScorer:
    @staticmethod
    def calculate_scores(packet: DocumentPacket) -> Dict[str, Any]:
        """
        Calculates all model-driven scores and decisions for the document packet.
        Returns:
            Dict containing ocr_confidence, classification_confidence,
            extraction_reliability, fraud_risk_score, authenticity_score,
            final_decision, and decision_description.
        """
        # Retrieve predictions from the calibrated LightGBM scoring models
        results = ScoringInference.predict(packet)
        return results
