from typing import Tuple, List, Dict, Any
from classifiers.base_classifier import BaseClassifier
from classifiers.keyword_classifier import KeywordClassifier
from classifiers.regex_classifier import RegexClassifier
from classifiers.layout_classifier import LayoutClassifier
from utils.document_packet import DocumentType
from utils.logger import get_logger

logger = get_logger("classifier_ensemble")

class ClassifierEnsemble(BaseClassifier):
    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the Classifier Ensemble.
        config: Dict containing classification weights and threshold settings.
        """
        self.config = config
        self.keyword_classifier = KeywordClassifier()
        self.regex_classifier = RegexClassifier()
        self.layout_classifier = LayoutClassifier()

    def classify(self, raw_text: str, word_map: List[Dict[str, Any]]) -> Tuple[DocumentType, float]:
        # 1. Fetch parameters from config
        class_config = self.config.get("classification", {})
        threshold = class_config.get("confidence_threshold", 0.70)
        weights = class_config.get("weights", {"keyword": 0.40, "regex": 0.40, "layout": 0.20})

        w_keyword = weights.get("keyword", 0.40)
        w_regex = weights.get("regex", 0.40)
        w_layout = weights.get("layout", 0.20)

        # 2. Get individual classifications
        kw_doc, kw_score = self.keyword_classifier.classify(raw_text, word_map)
        rg_doc, rg_score = self.regex_classifier.classify(raw_text, word_map)
        ly_doc, ly_score = self.layout_classifier.classify(raw_text, word_map)

        logger.info(f"Classifier individual results: Keyword={kw_doc.value}({kw_score:.2f}), "
                    f"Regex={rg_doc.value}({rg_score:.2f}), Layout={ly_doc.value}({ly_score:.2f})")

        # If a strict or soft ID number regex matched with confidence >= 0.70, trust it directly.
        # This prevents diluting reliable regex matches (like Aadhaar/PAN formats) when other classifiers are silent.
        if rg_score >= 0.70 and rg_doc != DocumentType.UNKNOWN:
            logger.info(f"Ensemble bypassed: Reliable regex match for {rg_doc.value} ({rg_score:.2f})")
            return rg_doc, rg_score

        # 3. Aggregate scores with normalized weights for active classifiers
        scores = {
            DocumentType.AADHAAR: 0.0,
            DocumentType.PAN: 0.0,
            DocumentType.PASSPORT: 0.0,
            DocumentType.DRIVING_LICENCE: 0.0
        }
        total_weight = 0.0
        if kw_doc != DocumentType.UNKNOWN:
            total_weight += w_keyword
        if rg_doc != DocumentType.UNKNOWN:
            total_weight += w_regex
        if ly_doc != DocumentType.UNKNOWN:
            total_weight += w_layout

        if total_weight > 0.0:
            if kw_doc != DocumentType.UNKNOWN:
                scores[kw_doc] += (w_keyword / total_weight) * kw_score
            if rg_doc != DocumentType.UNKNOWN:
                scores[rg_doc] += (w_regex / total_weight) * rg_score
            if ly_doc != DocumentType.UNKNOWN:
                scores[ly_doc] += (w_layout / total_weight) * ly_score
        else:
            # If all are UNKNOWN, no score can exceed 0.0
            pass

        # Find best score
        best_doc = DocumentType.UNKNOWN
        best_score = 0.0

        for doc_type, score in scores.items():
            if score > best_score:
                best_score = score
                best_doc = doc_type

        logger.info(f"Ensemble raw decision: {best_doc.value} with aggregated score {best_score:.2f} (Threshold: {threshold:.2f})")

        # 4. Enforce threshold
        if best_score < threshold:
            logger.warning(f"Ensemble score {best_score:.2f} is below threshold {threshold:.2f}. Classifying as UNKNOWN.")
            return DocumentType.UNKNOWN, best_score

        return best_doc, best_score
