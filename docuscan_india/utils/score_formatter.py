from typing import Tuple

class ScoreFormatter:
    @staticmethod
    def to_score_range(prob: float) -> int:
        """Converts a probability float [0, 1] to an integer score [0, 100]."""
        # Ensure it is bounded
        val = max(0.0, min(1.0, prob))
        return int(round(val * 100))

    @staticmethod
    def get_verdict(authenticity_score: int, fraud_risk_score: int, thresholds: dict = None) -> Tuple[str, str]:
        """
        Maps the authenticity and fraud risk scores to a final decision label and description.
        """
        if thresholds is None:
            thresholds = {
                "auth_genuine": 75,
                "auth_suspicious": 40,
                "fraud_suspicious": 60,
                "fraud_genuine": 25
            }
        
        auth_gen = thresholds.get("auth_genuine", 75)
        auth_susp = thresholds.get("auth_suspicious", 40)
        fraud_susp = thresholds.get("fraud_suspicious", 60)
        fraud_gen = thresholds.get("fraud_genuine", 25)

        if authenticity_score >= auth_gen and fraud_risk_score <= fraud_gen:
            return "Genuine", "Document is verified as authentic. All validations passed."
        elif authenticity_score < auth_susp or fraud_risk_score >= fraud_susp:
            return "Suspicious", "High probability of tampering or critical validation failures. Rejected."
        else:
            return "Needs Manual Review", "Minor verification alerts flagged. Manual review required."
