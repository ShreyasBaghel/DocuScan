from typing import Tuple

class ScoreFormatter:
    @staticmethod
    def to_score_range(prob: float) -> int:
        """Converts a probability float [0, 1] to an integer score [0, 100]."""
        # Ensure it is bounded
        val = max(0.0, min(1.0, prob))
        return int(round(val * 100))

    @staticmethod
    def get_verdict(authenticity_score: int, fraud_risk_score: int) -> Tuple[str, str]:
        """
        Maps the authenticity and fraud risk scores to a final decision label and description.
        Labels:
          - Genuine: authenticity_score >= 75 and fraud_risk_score <= 25
          - Suspicious: authenticity_score < 40 or fraud_risk_score >= 60
          - Needs Manual Review: otherwise
        """
        # Ensure we have dynamic but deterministic mappings without hardcoded nested ifs in core code
        if authenticity_score >= 75 and fraud_risk_score <= 25:
            return "Genuine", "Document is verified as authentic. All validations passed."
        elif authenticity_score < 40 or fraud_risk_score >= 60:
            return "Suspicious", "High probability of tampering or critical validation failures. Rejected."
        else:
            return "Needs Manual Review", "Minor verification alerts flagged. Manual review required."
