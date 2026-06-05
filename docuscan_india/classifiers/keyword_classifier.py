import re
from typing import Tuple, List, Dict, Any
from classifiers.base_classifier import BaseClassifier
from utils.document_packet import DocumentType

class KeywordClassifier(BaseClassifier):
    def __init__(self):
        # Specific keywords that uniquely identify each document type
        self.keywords = {
            DocumentType.AADHAAR: [
                r"unique\s+identification",
                r"uidai",
                r"aadhaar",
                r"भारत\s+सरकार",
                r"विशिष्ट\s+पहचान\s+प्राधिकरण",
                r"mera\s+aadhaar",
                r"yob\s*:",
                r"male\s+/\s+female",
                r"enrollment\s+no"
            ],
            DocumentType.PAN: [
                r"income\s+tax",
                r"permanent\s+account\s+card",
                r"permanent\s+account\s+number",
                r"आयकर\s+विभाग",
                r"pan\s+card",
                r"father\'s\s+name",
                r"govt\.\s+of\s+india"
            ],
            DocumentType.PASSPORT: [
                r"republic\s+of\s+india",
                r"passport\s+no",
                r"भारत\s+गणराज्य",
                r"passport",
                r"nationality",
                r"place\s+of\s+birth",
                r"type\s+code",
                r"sur\s*name",
                r"given\s+name"
            ],
            DocumentType.DRIVING_LICENCE: [
                r"driving\s+licence",
                r"driving\s+license",
                r"licence\s+to\s+drive",
                r"licensing\s+authority",
                r"transport\s+department",
                r"class\s+of\s+vehicle",
                r"lmv",
                r"mcwg",
                r"badge\s+no"
            ]
        }

    def classify(self, raw_text: str, word_map: List[Dict[str, Any]]) -> Tuple[DocumentType, float]:
        if not raw_text:
            return DocumentType.UNKNOWN, 0.0

        text_lower = raw_text.lower()
        scores: Dict[DocumentType, float] = {}

        for doc_type, regex_list in self.keywords.items():
            matches = 0
            for regex in regex_list:
                if re.search(regex, text_lower, re.IGNORECASE):
                    matches += 1
            # Calculate fraction of matched keywords
            scores[doc_type] = matches / len(regex_list) if regex_list else 0.0

        # Find the document type with the highest score
        best_doc = DocumentType.UNKNOWN
        best_score = 0.0

        for doc_type, score in scores.items():
            if score > best_score:
                best_score = score
                best_doc = doc_type

        # If best score is negligible, classify as UNKNOWN
        if best_score < 0.15:
            return DocumentType.UNKNOWN, 0.0

        return best_doc, best_score
